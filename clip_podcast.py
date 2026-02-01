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
MIN_CLIP_DURATION = 20
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
        '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
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
    """Transcribe audio using Groq Whisper API (FREE)."""
    print(f"[3/7] Transcribing with Whisper (language: {language})...")
    
    url = f"{GROQ_API_URL}/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
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
        
        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        
        result = response.json()
        
    print(f"  Transcribed: {len(result.get('segments', []))} segments")
    return result


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
    """Use Groq LLaMA to find top 10 viral segments."""
    print(f"[4/7] Analyzing for viral segments...")
    
    prompt = f"""You are a viral content expert specializing in podcast clips for TikTok/Reels/Shorts.

Analyze this podcast transcript and find the TOP 10 most viral-worthy segments.

TRANSCRIPT (with timestamps):
{transcript}

SCORING CRITERIA (rate each 1-10):
1. **Shock Value** - Unexpected revelation, controversial take, or surprising fact
2. **Humor** - Genuinely funny moment, witty comeback, or comedic timing
3. **Strong Opinion** - Bold claim, hot take, or passionate argument
4. **Emotional Peak** - Inspiring story, vulnerable moment, or intense emotion
5. **Quotability** - Memorable phrase that people will share

For each segment found, return:
- start_time: float (seconds) - extract from timestamp
- end_time: float (seconds) - should be 20-60 seconds after start
- duration: 20-60 seconds only
- viral_score: total out of 50
- type: "humor" | "shock" | "opinion" | "emotional" | "quotable"
- hook_line: First sentence that grabs attention
- context: Why this segment is viral-worthy

OUTPUT FORMAT (JSON only, no markdown):
{{"segments": [
  {{
    "start_time": 125.5,
    "end_time": 165.2,
    "duration": 39.7,
    "viral_score": 42,
    "type": "shock",
    "hook_line": "Wait, he actually said that?",
    "context": "Guest reveals controversial industry secret"
  }}
]}}

RULES:
- Each segment MUST be 20-60 seconds
- Leave 2-second padding at start/end
- Avoid segments that need context from earlier
- Order by viral_score descending (highest first)
- Return exactly 10 segments

Return ONLY the JSON, no other text."""

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
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    
    parsed = json.loads(content)
    segments = parsed.get('segments', [])
    
    print(f"  Found: {len(segments)} viral segments")
    return segments[:NUM_CLIPS]


def cut_clip(video_path: str, start: float, duration: float, output_path: str) -> str:
    """Cut a clip from the video."""
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(max(0, start - 1)),  # 1s before for safety
        '-i', video_path,
        '-t', str(duration + 2),  # 2s buffer
        '-c', 'copy',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try with re-encoding if stream copy fails
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(max(0, start - 1)),
            '-i', video_path,
            '-t', str(duration + 2),
            '-c:v', 'libx264', '-preset', 'fast',
            '-c:a', 'aac',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, text=True)
    
    return output_path


def generate_ass_subtitles(text: str, duration: float, output_path: str):
    """Generate ASS subtitles with word-by-word animation."""
    
    # Split into words
    words = text.split()
    words_per_line = 5
    time_per_word = duration / len(words) if words else 1
    
    ass_content = f"""[Script Info]
Title: Podcast Clip Subtitles
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{SUBTITLE_FONT},{SUBTITLE_FONTSIZE},{SUBTITLE_COLOR},&H000000FF,{SUBTITLE_OUTLINE},&H00000000,-1,0,0,0,100,100,0,0,1,4,0,2,50,50,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    current_time = 0.5  # Start 0.5s in
    
    for i in range(0, len(words), words_per_line):
        chunk = words[i:i + words_per_line]
        chunk_text = ' '.join(chunk).upper()
        
        start_time = current_time
        end_time = current_time + (time_per_word * len(chunk))
        
        # Format times
        start_str = format_ass_time(start_time)
        end_str = format_ass_time(end_time)
        
        # Clean text for ASS
        clean_text = chunk_text.replace('\\', '').replace('{', '').replace('}', '')
        
        ass_content += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{clean_text}\n"
        
        current_time = end_time
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)
    
    return output_path


def format_ass_time(seconds: float) -> str:
    """Format seconds to ASS time (H:MM:SS.CC)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def add_subtitles_and_crop(clip_path: str, subtitle_path: str, output_path: str) -> str:
    """Add subtitles and crop to 9:16 vertical format."""
    print(f"    Adding subtitles + vertical crop...")
    
    # Escape subtitle path for FFmpeg
    sub_path_escaped = subtitle_path.replace('\\', '/').replace(':', '\\:')
    
    cmd = [
        'ffmpeg', '-y',
        '-i', clip_path,
        '-vf', f"crop=ih*9/16:ih,scale=1080:1920,ass='{sub_path_escaped}'",
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-movflags', '+faststart',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FFmpeg warning: {result.stderr[:200]}")
        # Try without subtitles if ASS fails
        cmd = [
            'ffmpeg', '-y',
            '-i', clip_path,
            '-vf', "crop=ih*9/16:ih,scale=1080:1920",
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
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
            
            # Generate subtitles
            subtitle_path = str(temp_path / f"subtitles_{clip_num:02d}.ass")
            hook_text = segment.get('hook_line', 'Watch this!')
            generate_ass_subtitles(hook_text, duration, subtitle_path)
            
            # Add subtitles and crop
            final_clip_path = str(temp_path / f"clip_{clip_num:02d}_final.mp4")
            add_subtitles_and_crop(raw_clip_path, subtitle_path, final_clip_path)
            
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
    
    # Process
    clips = asyncio.run(process_podcast(podcast_url, podcast_title, language))
    
    # Callback to n8n
    if callback_url:
        print(f"\nSending callback to n8n...")
        callback_data = {
            "status": "success",
            "podcast_title": podcast_title,
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
