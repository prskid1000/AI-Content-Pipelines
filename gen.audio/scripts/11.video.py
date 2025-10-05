import os
import sys
import subprocess
import random
import wave
from pathlib import Path
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

# Configuration
SHORTS_VIDEO_COUNT = 5  # Number of shorts videos to create
SHORTS_VIDEO_LENGTH = 30  # Length of each shorts video in seconds


def find_thumbnail(output_dir: str) -> str | None:
    """Find a thumbnail image in the output directory.

    Prefers PNG but falls back to common formats.
    """
    candidates = [
        os.path.join(output_dir, "thumbnail.png"),
        os.path.join(output_dir, "thumbnail.jpg"),
        os.path.join(output_dir, "thumbnail.jpeg"),
        os.path.join(output_dir, "thumbnail.webp"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def find_shorts_thumbnails(output_dir: str) -> list[str]:
    """Find all shorts thumbnail images in the output directory."""
    shorts_thumbnails = []
    for i in range(1, SHORTS_VIDEO_COUNT + 1):
        candidates = [
            os.path.join(output_dir, f"thumbnail.shorts.v{i}.png"),
            os.path.join(output_dir, f"thumbnail.shorts.v{i}.jpg"),
            os.path.join(output_dir, f"thumbnail.shorts.v{i}.jpeg"),
            os.path.join(output_dir, f"thumbnail.shorts.v{i}.webp"),
        ]
        for path in candidates:
            if os.path.exists(path):
                shorts_thumbnails.append(path)
                break
    return shorts_thumbnails

def get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / float(sample_rate)
            return duration
    except Exception as e:
        print(f"ERROR: Failed to get audio duration: {e}")
        return 0.0

def get_random_audio_segments(audio_path: str, num_segments: int, segment_length: float) -> list[tuple[float, float]]:
    """Get random start times for audio segments."""
    try:
        total_duration = get_audio_duration(audio_path)
        if total_duration <= segment_length:
            print(f"WARNING: Audio duration ({total_duration:.2f}s) is shorter than segment length ({segment_length}s)")
            return [(0.0, total_duration)]
        
        # Calculate maximum start time to ensure segment fits
        max_start_time = total_duration - segment_length
        
        segments = []
        for _ in range(num_segments):
            start_time = random.uniform(0.0, max_start_time)
            end_time = start_time + segment_length
            segments.append((start_time, end_time))
        
        return segments
    except Exception as e:
        print(f"ERROR: Failed to generate random segments: {e}")
        return []


def build_ffmpeg_cmd(image_path: str, audio_path: str, output_path: str, start_time: float = 0.0, duration: float = None) -> list[str]:
    """Construct a YouTube-friendly ffmpeg command for a constant-image video.

    Settings target YouTube upload recommendations: H.264 High@4.2 in MP4,
    yuv420p 8-bit, AAC-LC at 48 kHz, with faststart and a GOP of 2s at 30 fps.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-loop", "1",                 # loop single image as video frames
        "-i", image_path,
        "-i", audio_path,
    ]
    
    # Add audio trimming if specified
    if start_time > 0.0:
        cmd.extend(["-ss", str(start_time)])
    
    if duration:
        cmd.extend(["-t", str(duration)])
    
    cmd.extend([
        # Video encoding (YouTube recommended)
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level:v", "4.2",
        "-tune", "stillimage",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-r", "30",
        "-g", "60",                    # 2s GOP at 30fps
        "-movflags", "+faststart",    # moov at start for better streaming
        # Audio encoding (YouTube recommended)
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        # Stop when shortest input ends (the audio)
        "-shortest",
        output_path,
    ])
    
    return cmd


def create_shorts_videos(output_dir: str, audio_path: str) -> int:
    """Create multiple shorts videos with random audio segments."""
    # Find shorts thumbnails
    shorts_thumbnails = find_shorts_thumbnails(output_dir)
    if len(shorts_thumbnails) < SHORTS_VIDEO_COUNT:
        print(f"ERROR: Found only {len(shorts_thumbnails)} shorts thumbnails, need {SHORTS_VIDEO_COUNT}")
        print("Expected: thumbnail.short.v1.png through thumbnail.short.v5.png")
        return 1
    
    # Get random audio segments
    audio_segments = get_random_audio_segments(audio_path, SHORTS_VIDEO_COUNT, SHORTS_VIDEO_LENGTH)
    if len(audio_segments) < SHORTS_VIDEO_COUNT:
        print(f"ERROR: Failed to generate {SHORTS_VIDEO_COUNT} audio segments")
        return 1
    
    print(f"Creating {SHORTS_VIDEO_COUNT} shorts videos...")
    print(f"Each video will be {SHORTS_VIDEO_LENGTH} seconds long")
    
    success_count = 0
    for i in range(SHORTS_VIDEO_COUNT):
        thumbnail_path = shorts_thumbnails[i]
        start_time, end_time = audio_segments[i]
        output_path = os.path.join(output_dir, f"shorts.v{i+1}.mp4")
        
        print(f"\nCreating shorts video {i+1}/{SHORTS_VIDEO_COUNT}:")
        print(f"  Thumbnail: {os.path.basename(thumbnail_path)}")
        print(f"  Audio segment: {start_time:.2f}s - {end_time:.2f}s")
        print(f"  Output: {os.path.basename(output_path)}")
        
        cmd = build_ffmpeg_cmd(thumbnail_path, audio_path, output_path, start_time, SHORTS_VIDEO_LENGTH)
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode == 0:
                print(f"  ✓ Success: {output_path}")
                success_count += 1
            else:
                print(f"  ✗ Failed: {result.stdout}")
        except FileNotFoundError:
            print("  ✗ ERROR: ffmpeg not found. Please install ffmpeg and ensure it is on PATH.")
            return 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    print(f"\nCompleted: {success_count}/{SHORTS_VIDEO_COUNT} shorts videos created")
    return 0 if success_count == SHORTS_VIDEO_COUNT else 1

def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    output_dir = os.path.join(project_root, "output")

    # Check for regular thumbnail
    image_path = find_thumbnail(output_dir)
    if not image_path:
        print("ERROR: thumbnail image not found in output/. Expected one of: thumbnail.png/.jpg/.jpeg/.webp")
        return 1

    audio_path = os.path.join(output_dir, "final.wav")
    if not os.path.exists(audio_path):
        print("ERROR: audio file not found at output/final.wav")
        return 1

    # Create regular video
    output_path = os.path.join(output_dir, "final.mp4")
    cmd = build_ffmpeg_cmd(image_path, audio_path, output_path)
    print("Creating regular video...")
    print("Running:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Please install ffmpeg and ensure it is on PATH.")
        return 1

    if result.returncode != 0:
        print("ffmpeg failed with output:\n" + result.stdout)
        return result.returncode

    print(f"Regular video written to {output_path}")
    
    # Create shorts videos
    return create_shorts_videos(output_dir, audio_path)


if __name__ == "__main__":
    sys.exit(main())


