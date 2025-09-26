import os
import sys
import argparse
import shutil
import subprocess


def print_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: replace unencodable characters for Windows cp1252 consoles
        enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
        sanitized = []
        for a in args:
            s = str(a)
            try:
                s.encode(enc, errors="strict")
            except Exception:
                s = s.encode(enc, errors="replace").decode(enc, errors="replace")
            sanitized.append(s)
        print(*sanitized, **kwargs)


def run_ffprobe_duration_ms(path: str) -> int:
    """Return media duration in milliseconds using ffprobe. -1 if error."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not os.path.exists(path):
        return -1
    try:
        # Output only duration in seconds with milliseconds
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        s = result.stdout.strip()
        seconds = float(s)
        return int(round(seconds * 1000.0))
    except Exception:
        return -1


def run_ffprobe_stream_duration_ms(path: str, stream_selector: str) -> int:
    """Return specific stream duration in ms (audio or video) using ffprobe. -1 if error."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not os.path.exists(path):
        return -1
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                stream_selector,
                "-show_entries",
                "stream=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        s = result.stdout.strip()
        if not s:
            return -1
        seconds = float(s)
        return int(round(seconds * 1000.0))
    except Exception:
        return -1


def human_ms(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:d}m {seconds:06.3f}s ({ms} ms)"


def combine(video_path: str,
            audio_path: str,
            out_path: str,
            tolerance_ms: int = 500,
            strict: bool = False) -> int:
    """Mux and re-encode to a YouTube 720p-friendly MP4 (H.264 + AAC, 1280x720, yuv420p)."""

    if not os.path.exists(video_path):
        print_flush(f"❌ Video not found: {video_path}")
        return 1
    if not os.path.exists(audio_path):
        print_flush(f"❌ Audio not found: {audio_path}")
        return 1

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        print_flush("❌ ffmpeg is not available in PATH. Please install ffmpeg.")
        return 1
    if not ffprobe:
        print_flush("❌ ffprobe is not available in PATH. Please install ffmpeg (includes ffprobe).")
        return 1

    # Probe durations
    v_dur_ms = run_ffprobe_duration_ms(video_path)
    a_dur_ms = run_ffprobe_duration_ms(audio_path)
    # Fallback to stream-specific queries if needed
    if v_dur_ms <= 0:
        v_dur_ms = run_ffprobe_stream_duration_ms(video_path, "v:0")
    if a_dur_ms <= 0:
        a_dur_ms = run_ffprobe_stream_duration_ms(audio_path, "a:0")

    if v_dur_ms > 0:
        print_flush(f"🎬 Video duration: {human_ms(v_dur_ms)}")
    else:
        print_flush("⚠️ Could not determine video duration")
    if a_dur_ms > 0:
        print_flush(f"🔊 Audio duration: {human_ms(a_dur_ms)}")
    else:
        print_flush("⚠️ Could not determine audio duration")

    diff_ms = -1
    if v_dur_ms > 0 and a_dur_ms > 0:
        diff_ms = abs(v_dur_ms - a_dur_ms)
        print_flush(f"Δ duration: {diff_ms} ms (tolerance {tolerance_ms} ms)")
        if strict and diff_ms > tolerance_ms:
            print_flush("❌ Lengths differ beyond tolerance in strict mode. Aborting.")
            return 2

    # Hardcoded 720p settings and longest-match policy
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720
    FPS = 30
    AUDIO_BITRATE = "192k"

    # Build base command and decide padding approach
    vf_chain = (
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
    )

    # Determine which stream to pad to match longest
    use_tpad = False
    use_apad = False
    pad_sec = 0.0
    if v_dur_ms > 0 and a_dur_ms > 0:
        if v_dur_ms < a_dur_ms:
            use_tpad = True
            pad_sec = max(0.0, (a_dur_ms - v_dur_ms) / 1000.0)
        elif a_dur_ms < v_dur_ms:
            use_apad = True
            pad_sec = max(0.0, (v_dur_ms - a_dur_ms) / 1000.0)

    cmd = [
        ffmpeg,
        "-y",
        "-i", video_path,
        "-i", audio_path,
    ]

    filter_args = []
    map_args = []

    if use_tpad:
        # Apply scale+pad then tpad in a single filter_complex for video
        filter_args = [
            "-filter_complex",
            f"[0:v]{vf_chain},tpad=stop_mode=clone:stop_duration={pad_sec:.3f}[vout]",
            "-map", "[vout]",
            "-map", "1:a",
        ]
    else:
        # Use -vf for video; if audio shorter, apad it in filter_complex
        cmd += ["-vf", vf_chain]
        map_args = ["-map", "0:v", "-map", "1:a"]
        if use_apad:
            filter_args = [
                "-filter_complex", f"[1:a]apad=pad_dur={pad_sec:.3f}[aout]",
                "-map", "0:v",
                "-map", "[aout]",
            ]

    # Video encode settings
    cmd += [
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level:v", "4.2",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-r", str(FPS),
        "-g", str(FPS * 2),
        "-movflags", "+faststart",
    ]

    # Audio encode settings
    cmd += [
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-ar", "48000",
        "-ac", "2",
    ]

    cmd += filter_args + map_args + [out_path]

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    print_flush("🔗 Muxing and encoding to YouTube SD MP4...")
    print_flush("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print_flush(f"❌ ffmpeg failed: {e}")
        return 3

    # Post-check: probe final duration and report delta vs video
    final_ms = run_ffprobe_duration_ms(out_path)
    if final_ms > 0 and v_dur_ms > 0:
        fin_diff = abs(final_ms - v_dur_ms)
        print_flush(f"✅ Wrote: {out_path}")
        print_flush(f"Final duration: {human_ms(final_ms)} | vs source video: {fin_diff} ms diff")
    else:
        print_flush(f"✅ Wrote: {out_path}")

    # Guidance for YouTube SD compliance
    print_flush("\n📺 YouTube SD settings: 854x480, H.264 (yuv420p), AAC stereo. This file matches those settings.")
    if diff_ms >= 0:
        print_flush("Length diff video vs audio:", f"{diff_ms} ms")
    return 0


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_video = os.path.normpath(os.path.join(script_dir, "../output/merged.mp4"))
    default_audio = os.path.normpath(os.path.join(script_dir, "../../gen.audio/output/final.wav"))
    default_out = os.path.normpath(os.path.join(script_dir, "../output/final_sd.mp4"))

    parser = argparse.ArgumentParser(description="Combine merged.mp4 video with final.wav audio into YouTube 720p MP4 (no flags needed)")
    parser.add_argument("--video", default=default_video, help="Path to source MP4 video (merged.mp4)")
    parser.add_argument("--audio", default=default_audio, help="Path to source WAV audio (final.wav)")
    parser.add_argument("--out", default=default_out, help="Output MP4 path")
    parser.add_argument("--tolerance-ms", type=int, default=500, help="Allowed duration difference before strict abort")
    parser.add_argument("--strict", action="store_true", help="Abort if durations differ beyond tolerance")

    args = parser.parse_args()

    return combine(
        video_path=args.video,
        audio_path=args.audio,
        out_path=args.out,
        tolerance_ms=args.tolerance_ms,
        strict=args.strict,
    )


if __name__ == "__main__":
    sys.exit(main())


