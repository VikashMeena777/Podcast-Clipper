#!/usr/bin/env python3
"""
Podcast Clipper - Extracts 10 viral shorts from podcasts
=========================================================
Uses Groq Whisper (FREE) for transcription
Uses Groq LLaMA (FREE) for segment scoring + metadata
Uses FFmpeg for video processing
Supports Hindi + English podcasts
"""

import os
import sys
import json
import asyncio
import argparse
import subprocess
import tempfile
import shutil
import requests
from pathlib import Path
from typing import List, Dict
import re

# Groq API
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_API_URL = "https://api.groq.com/openai/v1"

# Processing config
MAX_CLIP_DURATION = 60
MIN_CLIP_DURATION = 30  # Minimum 30 seconds for better context
NUM_CLIPS = 10
WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL = "llama-3.3-70b-versatile"

# Subtitle styling
SUBTITLE_FONT = "Arial"
SUBTITLE_FONTSIZE = 50
SUBTITLE_COLOR = "&H00FFFFFF"
SUBTITLE_OUTLINE = "&H00000000"


def download_youtube(url: str, output_path: str) -> str:
    """Download YouTube video using yt-dlp."""
    print(f"[1/7] Downloading from YouTube...")
    
    cmd = [
        'yt-dlp',
        '--js-runtimes', 'deno',
        '--remote-components', 'ejs:github',
        '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
        '--merge-output-format', 'mp4',
        '-o', output_path,
    ]
    
    # Add cookies if available
    cookies_file = os.environ.get('YOUTUBE_COOKIES_FILE', '')
    if cookies_file and os.path.exists(cookies_file):
        cmd.extend(['--cookies', cookies_file])
        print(f"  Using cookies file: {cookies_file}")
    
    # Add user agent to look like a real browser
    cmd.extend([
        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        '--sleep-interval', '1',
        '--max-sleep-interval', '3',
    ])
    
    cmd.append(url)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  yt-dlp stderr: {result.stderr}")
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")
    return output_path


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video for Whisper transcription."""
    print(f"[2/7] Extracting audio...")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'mp3',
        '-ar', '16000',
        '-ac', '1',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Audio: {size_mb:.1f} MB")
    return output_path


async def transcribe_audio(audio_path: str, language: str = "hi") -> Dict:
    """Transcribe audio - tries local Whisper first, then Groq API."""
    print(f"[3/7] Transcribing with Whisper (language: {language})...")
    
    # Try local faster-whisper first (FREE, no limits!)
    try:
        return await transcribe_local_whisper(audio_path, language)
    except Exception as e:
        print(f"  Local Whisper failed: {e}")
        print(f"  Falling back to Groq API...")
        return await transcribe_groq_api(audio_path, language)


async def transcribe_local_whisper(audio_path: str, language: str = "hi") -> Dict:
    """Transcribe using local faster-whisper (FREE, no limits!)"""
    print(f"  Using local faster-whisper (FREE)...")
    
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("faster-whisper not installed")
    
    # Use small model for speed on GitHub Actions (no GPU)
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    segments_list = []
    full_text = []
    
    # Transcribe
    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        word_timestamps=False
    )
    
    for segment in segments:
        segments_list.append({
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip()
        })
        full_text.append(segment.text.strip())
    
    print(f"  Transcribed: {len(segments_list)} segments (local)")
    
    return {
        'text': ' '.join(full_text),
        'segments': segments_list
    }


async def transcribe_groq_api(audio_path: str, language: str = "hi") -> Dict:
    """Transcribe using Groq Whisper API (fallback)."""
    print(f"  Using Groq Whisper API...")
    
    # Check file size - Groq has 25MB limit
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"  Audio file size: {file_size_mb:.1f} MB")
    
    # If file is too large, split it
    if file_size_mb > 24:
        print(f"  Audio too large, splitting into chunks...")
        return await transcribe_audio_chunked(audio_path, language)
    
    url = f"{GROQ_API_URL}/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    # Retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(audio_path, 'rb') as f:
                files = {
                    'file': (os.path.basename(audio_path), f, 'audio/mp3')
                }
                data = {
                    'model': WHISPER_MODEL,
                    'language': language,
                    'response_format': 'verbose_json',
                    'timestamp_granularities[]': 'segment'
                }
                
                response = requests.post(url, headers=headers, files=files, data=data, timeout=300)
                
                if response.status_code in [429, 500]:
                    wait_time = 30 * (attempt + 1)
                    print(f"  Error {response.status_code}, waiting {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                result = response.json()
                
            print(f"  Transcribed: {len(result.get('segments', []))} segments (Groq)")
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}: {e}")
                import time
                time.sleep(10)
            else:
                raise
    
    raise RuntimeError("Groq transcription failed")


async def transcribe_audio_chunked(audio_path: str, language: str = "hi") -> Dict:
    """Split audio and transcribe in chunks for long podcasts."""
    import time
    
    # Get audio duration
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
           '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())
    
    # Split into ~10 minute chunks
    chunk_duration = 600
    num_chunks = int(total_duration / chunk_duration) + 1
    
    print(f"  Total duration: {total_duration/60:.1f} min, splitting into {num_chunks} chunks")
    
    all_segments = []
    
    with tempfile.TemporaryDirectory() as chunk_dir:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = os.path.join(chunk_dir, f"chunk_{i:02d}.mp3")
            
            cmd = [
                'ffmpeg', '-y', '-ss', str(start_time),
                '-i', audio_path, '-t', str(chunk_duration),
                '-acodec', 'mp3', '-ar', '16000', '-ac', '1',
                chunk_path
            ]
            subprocess.run(cmd, capture_output=True, text=True)
            
            if not os.path.exists(chunk_path):
                continue
                
            print(f"  Transcribing chunk {i+1}/{num_chunks}...")
            
            # Use local whisper for chunks
            try:
                chunk_result = await transcribe_local_whisper(chunk_path, language)
            except:
                chunk_result = await transcribe_groq_api(chunk_path, language)
            
            for segment in chunk_result.get('segments', []):
                segment['start'] += start_time
                segment['end'] += start_time
                all_segments.append(segment)
            
            if i < num_chunks - 1:
                time.sleep(2)
    
    return {
        'text': ' '.join(s.get('text', '') for s in all_segments),
        'segments': all_segments
    }


def format_transcript_with_timestamps(whisper_result: Dict) -> str:
    """Format transcript with timestamps for LLM analysis."""
    lines = []
    for segment in whisper_result.get('segments', []):
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()
        lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")
    
    return "\n".join(lines)


async def find_viral_segments(transcript: str) -> List[Dict]:
    """Use Groq LLaMA to find top 10 viral segments. Handles large transcripts by chunking."""
    print(f"[4/7] Analyzing for viral segments...")
    
    # Check transcript size - Groq has token limits, use small chunks for reliability
    MAX_CHUNK_CHARS = 15000  # Small chunks to avoid 413 errors
    
    if len(transcript) > MAX_CHUNK_CHARS:
        print(f"  Transcript too large ({len(transcript)} chars), analyzing in chunks...")
        return await find_viral_segments_chunked(transcript, MAX_CHUNK_CHARS)
    
    return await analyze_transcript_chunk(transcript)


async def find_viral_segments_chunked(transcript: str, chunk_size: int) -> List[Dict]:
    """Split transcript into chunks and find best segments from each."""
    lines = transcript.split('\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for line in lines:
        if current_size + len(line) > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(line)
        current_size += len(line) + 1
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    print(f"  Split into {len(chunks)} chunks")
    
    all_segments = []
    
    for i, chunk in enumerate(chunks):
        print(f"  Analyzing chunk {i+1}/{len(chunks)}...")
        try:
            chunk_segments = await analyze_transcript_chunk(chunk)
            all_segments.extend(chunk_segments)
        except Exception as e:
            print(f"    Chunk {i+1} failed: {e}")
            continue
    
    # Sort by viral score and return top 10
    all_segments.sort(key=lambda x: x.get('viral_score', 0), reverse=True)
    print(f"  Found {len(all_segments)} total segments, selecting top 10")
    
    return all_segments[:NUM_CLIPS]


async def analyze_transcript_chunk(transcript: str) -> List[Dict]:
    """Analyze a single transcript chunk for viral segments."""
    
    prompt = f"""You are a viral content expert. Find clips that will BLOW UP on TikTok/Reels/Shorts.

TRANSCRIPT:
{transcript}

⚠️ STRICT REQUIREMENTS - A good clip MUST have ALL of these:

1. **STANDALONE** - A random viewer with ZERO context must understand 100% of what's being said. 
   - ❌ REJECT: "As I was saying...", "Like I mentioned...", "So yeah, that's why..."
   - ❌ REJECT: References to earlier topics, people not introduced, inside jokes
   - ✅ ACCEPT: Complete story with beginning/middle/end, universal truth, standalone advice

2. **STRONG OPENING** - First 3 seconds must HOOK the viewer:
   - ✅ "Here's what nobody tells you about..."
   - ✅ "The biggest mistake people make is..."
   - ✅ "I'm going to share something that changed my life..."
   - ❌ "...and so then I..." (mid-sentence start)

3. **COMPLETE THOUGHT** - The clip must have a SATISFYING ENDING:
   - ✅ Story has punchline/conclusion
   - ✅ Advice is fully explained
   - ❌ Gets cut off mid-point
   - ❌ Ends with "so..." or trails off

4. **EMOTIONAL PUNCH** - Must trigger strong reaction:
   - Surprising revelation
   - Controversial opinion  
   - Inspiring story
   - Funny moment
   - Mind-blowing fact

CONTENT TYPES:
- "insight" - Universal wisdom anyone can apply
- "story" - Complete anecdote with clear lesson
- "opinion" - Bold, debatable take
- "motivation" - Empowering message
- "humor" - Genuinely funny standalone moment

DURATION: 30-60 seconds (sweet spot: 45 seconds)

OUTPUT (JSON only):
{{"segments": [
  {{
    "start_time": 125.0,
    "end_time": 170.0,
    "duration": 45,
    "viral_score": 42,
    "type": "insight",
    "hook_line": "Here's what nobody tells you about success...",
    "context": "Universal advice about mindset that anyone can apply"
  }}
]}}

Find TOP 5 clips. Return ONLY valid JSON."""

    url = f"{GROQ_API_URL}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=120)
    response.raise_for_status()
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    
    parsed = json.loads(content)
    segments = parsed.get('segments', [])
    
    return segments


def cut_clip(video_path: str, start: float, duration: float, output_path: str) -> str:
    """Cut a clip from the video with proper sync."""
    # Put -ss AFTER -i for accurate seeking (slower but synced)
    # Add small buffer before and after
    actual_start = max(0, start - 0.5)
    actual_duration = duration + 1
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-ss', str(actual_start),
        '-t', str(actual_duration),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
        '-c:a', 'aac', '-b:a', '192k',
        '-avoid_negative_ts', 'make_zero',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FFmpeg cut error: {result.stderr[:200]}")
        raise RuntimeError(f"Failed to cut clip: {result.stderr}")
    
    return output_path


def generate_ass_subtitles(segments: List[Dict], clip_start: float, clip_duration: float, output_path: str):
    """Generate ASS subtitles with word-by-word display (2-3 words at a time)."""
    
    # Use Noto Sans for Hindi/Hinglish support
    font_name = "Noto Sans Devanagari"
    font_size = 100  # Very large font for mobile reels
    
    # ASS Style parameters:
    # Alignment: 5 = center-middle of screen
    # MarginV: 600 = position in lower-middle area
    # Outline: 6 = thick black outline
    # Shadow: 4 = drop shadow for depth
    
    ass_content = f"""[Script Info]
Title: Podcast Clip Subtitles
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,4,2,50,50,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    clip_end = clip_start + clip_duration
    
    # Collect all words with their timestamps
    all_words = []
    
    for seg in segments:
        seg_start = seg.get('start', 0)
        seg_end = seg.get('end', 0)
        text = seg.get('text', '').strip()
        
        if not text:
            continue
        
        # Check if segment overlaps with clip
        if seg_end < clip_start or seg_start > clip_end:
            continue
        
        # Split segment text into words
        words = text.split()
        if not words:
            continue
        
        # Calculate time per word in this segment
        seg_duration = seg_end - seg_start
        time_per_word = seg_duration / len(words) if words else 0
        
        for i, word in enumerate(words):
            word_start = seg_start + (i * time_per_word)
            word_end = seg_start + ((i + 1) * time_per_word)
            
            # Only include words within clip bounds
            if word_end >= clip_start and word_start <= clip_end:
                all_words.append({
                    'text': word,
                    'start': word_start,
                    'end': word_end
                })
    
    # Group words into chunks of 2-3 words
    words_per_chunk = 3
    
    for i in range(0, len(all_words), words_per_chunk):
        chunk = all_words[i:i + words_per_chunk]
        if not chunk:
            continue
        
        # Get timing for this chunk
        chunk_start = chunk[0]['start']
        chunk_end = chunk[-1]['end']
        
        # Adjust times relative to clip start
        rel_start = max(0, chunk_start - clip_start)
        rel_end = min(clip_duration, chunk_end - clip_start)
        
        if rel_end <= rel_start:
            continue
        
        # Format times for ASS
        start_str = format_ass_time(rel_start)
        end_str = format_ass_time(rel_end)
        
        # Combine words and clean text
        chunk_text = ' '.join(w['text'] for w in chunk)
        clean_text = chunk_text.replace('\\', '').replace('{', '').replace('}', '')
        clean_text = clean_text.replace('\n', ' ').strip().upper()
        
        ass_content += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{clean_text}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    return output_path


def format_ass_time(seconds: float) -> str:
    """Format seconds to ASS time (H:MM:SS.CC)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def add_subtitles_and_blur_background(clip_path: str, subtitle_path: str, output_path: str) -> str:
    """Add subtitles and create 9:16 with blurred background (full video, not cropped)."""
    print(f"    Adding subtitles + blurred background...")
    
    # Escape subtitle path for FFmpeg filter
    sub_path_escaped = subtitle_path.replace('\\', '/').replace(':', '\\:')
    
    # Complex filter for blurred background effect:
    # 1. Scale original to fit 9:16 (with letterboxing to keep aspect)
    # 2. Create blurred background from same video scaled to fill
    # 3. Overlay original on top of blurred background
    # 4. Add subtitles
    
    filter_complex = (
        # Background: scale to fill 1080x1920 and blur heavily
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:5[bg];"
        # Foreground: scale to fit within 1080x1920 keeping aspect ratio
        "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
        # Overlay foreground on blurred background (centered)
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[video];"
        # Add subtitles
        f"[video]ass='{sub_path_escaped}'[out]"
    )
    
    cmd = [
        'ffmpeg', '-y',
        '-i', clip_path,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-map', '0:a',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '22',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FFmpeg error with subtitles, trying without...")
        print(f"    Error: {result.stderr[:300]}")
        
        # Fallback: without subtitles
        filter_complex_no_sub = (
            "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=30:5[bg];"
            "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2[out]"
        )
        
        cmd = [
            'ffmpeg', '-y',
            '-i', clip_path,
            '-filter_complex', filter_complex_no_sub,
            '-map', '[out]',
            '-map', '0:a',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'aac', '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, text=True)
    
    return output_path


async def generate_metadata(segment: Dict, podcast_title: str) -> Dict:
    """Generate title, caption, hashtags using Groq LLaMA."""
    
    prompt = f"""You are a viral social media expert creating metadata for podcast clips.

CLIP INFO:
- Podcast: {podcast_title}
- Hook: {segment.get('hook_line', '')}
- Type: {segment.get('type', 'opinion')}
- Context: {segment.get('context', '')}

Generate metadata for this clip. Be catchy and engaging!

OUTPUT FORMAT (JSON only):
{{
  "title": "Catchy title under 60 chars for Reels/Shorts",
  "caption": "Instagram caption with hook, 2-3 lines, include CTA",
  "hashtags": ["#podcast", "#viral", "...25 total hashtags"],
  "youtube_description": "150-200 word SEO description"
}}

For Hindi content, use Hinglish (Roman Hindi) in captions.
Return ONLY JSON, no markdown."""

    url = f"{GROQ_API_URL}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    
    return json.loads(content)


def upload_to_drive(file_path: str, folder: str = "Podcast Clips") -> str:
    """Upload file to Google Drive using rclone."""
    print(f"    Uploading to Google Drive...")
    
    filename = os.path.basename(file_path)
    remote_path = f"vk889900:{folder}/{filename}"
    
    cmd = ['rclone', 'copyto', file_path, remote_path, '-v']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    rclone warning: {result.stderr[:100]}")
        return f"local://{file_path}"
    
    # Get file ID
    cmd = ['rclone', 'lsjson', remote_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        try:
            info = json.loads(result.stdout)
            if info and len(info) > 0:
                file_id = info[0].get('ID', '')
                if file_id:
                    return f"https://drive.google.com/file/d/{file_id}/view"
        except:
            pass
    
    return remote_path


async def process_podcast(podcast_url: str, podcast_title: str, language: str = "hi") -> List[Dict]:
    """Main processing function."""
    print("=" * 60)
    print("PODCAST CLIPPER (FREE)")
    print("=" * 60)
    print(f"Title: {podcast_title}")
    print(f"Language: {language}")
    
    clips_data = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Download video
        video_path = str(temp_path / "podcast.mp4")
        download_youtube(podcast_url, video_path)
        
        # 2. Extract audio
        audio_path = str(temp_path / "audio.mp3")
        extract_audio(video_path, audio_path)
        
        # 3. Transcribe
        whisper_result = await transcribe_audio(audio_path, language)
        transcript = format_transcript_with_timestamps(whisper_result)
        
        # Save transcript
        with open(str(temp_path / "transcript.txt"), 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # 4. Find viral segments
        segments = await find_viral_segments(transcript)
        
        # Get raw segments from whisper for subtitles
        whisper_segments = whisper_result.get('segments', [])
        
        if not segments:
            raise RuntimeError("No viral segments found!")
        
        # 5-6. Process each clip
        print(f"[5/7] Cutting {len(segments)} clips...")
        
        for i, segment in enumerate(segments):
            clip_num = i + 1
            print(f"\n  Clip {clip_num}/{len(segments)} (score: {segment.get('viral_score', 0)})")
            
            start = segment['start_time']
            duration = min(segment.get('duration', 45), MAX_CLIP_DURATION)
            duration = max(duration, MIN_CLIP_DURATION)
            
            # Cut clip
            raw_clip_path = str(temp_path / f"raw_clip_{clip_num:02d}.mp4")
            cut_clip(video_path, start, duration, raw_clip_path)
            
            # Generate subtitles from actual transcript segments (not just hook_line)
            subtitle_path = str(temp_path / f"subtitles_{clip_num:02d}.ass")
            generate_ass_subtitles(whisper_segments, start, duration, subtitle_path)
            
            # Add subtitles and blurred background
            final_clip_path = str(temp_path / f"clip_{clip_num:02d}_final.mp4")
            add_subtitles_and_blur_background(raw_clip_path, subtitle_path, final_clip_path)
            
            # Generate metadata
            print(f"    Generating metadata...")
            metadata = await generate_metadata(segment, podcast_title)
            
            # Upload to Drive
            safe_title = ''.join(c for c in podcast_title[:20] if c.isalnum() or c == ' ').replace(' ', '_')
            upload_filename = f"clip_{clip_num:02d}_{safe_title}.mp4"
            final_upload_path = str(temp_path / upload_filename)
            shutil.copy(final_clip_path, final_upload_path)
            
            video_url = upload_to_drive(final_upload_path)
            
            # Compile clip data
            clip_data = {
                "clip_number": clip_num,
                "start_time": start,
                "end_time": start + duration,
                "duration": duration,
                "viral_score": segment.get('viral_score', 0),
                "viral_type": segment.get('type', 'opinion'),
                "hook_line": segment.get('hook_line', ''),
                "title": metadata.get('title', f'Clip {clip_num}'),
                "caption": metadata.get('caption', ''),
                "hashtags": metadata.get('hashtags', []),
                "youtube_description": metadata.get('youtube_description', ''),
                "video_url": video_url
            }
            
            clips_data.append(clip_data)
            print(f"    ✓ Clip {clip_num} complete!")
        
        print(f"\n[6/7] All clips processed!")
        print(f"[7/7] Preparing callback data...")
    
    return clips_data


def main():
    parser = argparse.ArgumentParser(description='Extract viral clips from podcast')
    parser.add_argument('--payload', required=True, help='JSON payload')
    args = parser.parse_args()
    
    payload = json.loads(args.payload)
    
    podcast_url = payload['podcast_url']
    podcast_title = payload.get('title', 'Podcast')
    language = payload.get('language', 'hi')
    callback_url = payload.get('callback_url', '')
    row_id = payload.get('row_id', '')  # For matching in Google Sheet
    
    # Process
    clips = asyncio.run(process_podcast(podcast_url, podcast_title, language))
    
    # Callback to n8n
    if callback_url:
        print(f"\nSending callback to n8n...")
        callback_data = {
            "status": "success",
            "podcast_title": podcast_title,
            "row_id": row_id,  # Include for Google Sheet matching
            "clips_count": len(clips),
            "clips": clips
        }
        
        try:
            response = requests.post(callback_url, json=callback_data, timeout=30)
            print(f"  Callback sent: {response.status_code}")
        except Exception as e:
            print(f"  Callback failed: {e}")
    
    # Output for GitHub Actions
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"clips_count={len(clips)}\n")
            f.write(f"status=success\n")
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print(f"Generated {len(clips)} viral clips")
    print("=" * 60)
    
    return clips


if __name__ == '__main__':
    main()
