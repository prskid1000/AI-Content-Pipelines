import os
import sys
import subprocess
from pathlib import Path
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)


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


def build_ffmpeg_cmd(image_path: str, audio_path: str, output_path: str) -> list[str]:
    """Construct a YouTube-friendly ffmpeg command for a constant-image video.

    Settings target YouTube upload recommendations: H.264 High@4.2 in MP4,
    yuv420p 8-bit, AAC-LC at 48 kHz, with faststart and a GOP of 2s at 30 fps.
    """
    return [
        "ffmpeg",
        "-y",
        "-loop", "1",                 # loop single image as video frames
        "-i", image_path,
        "-i", audio_path,
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
    ]


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    output_dir = os.path.join(project_root, "output")

    image_path = find_thumbnail(output_dir)
    if not image_path:
        print("ERROR: thumbnail image not found in output/. Expected one of: thumbnail.png/.jpg/.jpeg/.webp")
        return 1

    audio_path = os.path.join(output_dir, "final.wav")
    if not os.path.exists(audio_path):
        print("ERROR: audio file not found at output/final.wav")
        return 1

    output_path = os.path.join(output_dir, "final.mp4")

    cmd = build_ffmpeg_cmd(image_path, audio_path, output_path)
    print("Running:", " ".join(cmd))

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Please install ffmpeg and ensure it is on PATH.")
        return 1

    if result.returncode != 0:
        print("ffmpeg failed with output:\n" + result.stdout)
        return result.returncode

    print(f"Video written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


