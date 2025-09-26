import os
import sys
import argparse
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)


def human_ms(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:d}m {seconds:06.3f}s ({ms} ms)"


def combine_with_torchaudio(story_path: str, sfx_path: str, out_path: str, tolerance_ms: int, strict: bool) -> int:
    import torch
    import torchaudio
    from torchaudio.transforms import Resample

    if not os.path.exists(story_path):
        print(f"❌ Story file not found: {story_path}")
        return 1
    if not os.path.exists(sfx_path):
        print(f"❌ SFX file not found: {sfx_path}")
        return 1

    story_waveform, story_sr = torchaudio.load(story_path)  # shape: [C, N]
    sfx_waveform, sfx_sr = torchaudio.load(sfx_path)

    story_channels, story_num_samples = story_waveform.shape
    sfx_channels, sfx_num_samples = sfx_waveform.shape

    story_len_ms = int(round(story_num_samples * 1000.0 / story_sr))
    sfx_len_ms = int(round(sfx_num_samples * 1000.0 / sfx_sr))
    diff_ms = abs(story_len_ms - sfx_len_ms)

    print(f"Story: {human_ms(story_len_ms)} @ {story_sr} Hz, {story_channels} ch")
    print(f"SFX  : {human_ms(sfx_len_ms)} @ {sfx_sr} Hz, {sfx_channels} ch")
    print(f"Δ length: {diff_ms} ms (tolerance {tolerance_ms} ms)")

    if strict and diff_ms > tolerance_ms:
        print("❌ Lengths differ beyond tolerance in strict mode. Aborting.")
        return 2

    # Resample SFX to story sample rate if needed
    if sfx_sr != story_sr:
        resampler = Resample(orig_freq=sfx_sr, new_freq=story_sr)
        sfx_waveform = resampler(sfx_waveform)
        sfx_sr = story_sr
        sfx_num_samples = sfx_waveform.shape[1]

    # Match channel count to story
    if sfx_channels != story_channels:
        if sfx_channels == 1 and story_channels > 1:
            sfx_waveform = sfx_waveform.expand(story_channels, -1)
        elif sfx_channels > 1 and story_channels == 1:
            sfx_waveform = sfx_waveform.mean(dim=0, keepdim=True)
        else:
            # General case: repeat or average to match
            if sfx_channels < story_channels:
                repeats = (story_channels + sfx_channels - 1) // sfx_channels
                sfx_waveform = sfx_waveform.repeat(repeats, 1)[:story_channels, :]
            else:
                sfx_waveform = sfx_waveform[:story_channels, :]
        sfx_channels = story_channels

    # Pad/trim SFX to exactly story length in samples
    if sfx_waveform.shape[1] < story_num_samples:
        pad_len = story_num_samples - sfx_waveform.shape[1]
        pad = torch.zeros((story_channels, pad_len), dtype=sfx_waveform.dtype)
        sfx_waveform = torch.cat([sfx_waveform, pad], dim=1)
    elif sfx_waveform.shape[1] > story_num_samples:
        sfx_waveform = sfx_waveform[:, :story_num_samples]

    # Mix: keep story dominant, add SFX at -6 dB by default
    # -6 dB ≈ 0.501187; use 0.5 for simplicity
    mixed = story_waveform + (0.5 * sfx_waveform)

    # Prevent clipping
    mixed = torch.clamp(mixed, min=-1.0, max=1.0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, mixed, sample_rate=story_sr, encoding="PCM_S", bits_per_sample=16)

    print(f"✅ Saved: {out_path}")
    print(f"Final length (story-locked): {human_ms(story_len_ms)}")
    return 0


def combine_with_pydub(story_path: str, sfx_path: str, out_path: str, tolerance_ms: int, strict: bool) -> int:
    from pydub import AudioSegment

    if not os.path.exists(story_path):
        print(f"❌ Story file not found: {story_path}")
        return 1
    if not os.path.exists(sfx_path):
        print(f"❌ SFX file not found: {sfx_path}")
        return 1

    story = AudioSegment.from_file(story_path)
    sfx = AudioSegment.from_file(sfx_path)

    story_len_ms = len(story)
    sfx_len_ms = len(sfx)
    diff_ms = abs(story_len_ms - sfx_len_ms)

    print(f"Story: {human_ms(story_len_ms)} @ {story.frame_rate} Hz, {story.channels} ch")
    print(f"SFX  : {human_ms(sfx_len_ms)} @ {sfx.frame_rate} Hz, {sfx.channels} ch")
    print(f"Δ length: {diff_ms} ms (tolerance {tolerance_ms} ms)")

    if strict and diff_ms > tolerance_ms:
        print("❌ Lengths differ beyond tolerance in strict mode. Aborting.")
        return 2

    # Convert SFX channel count to match story for proper overlay
    if sfx.channels != story.channels:
        sfx = sfx.set_channels(story.channels)

    # Overlay SFX onto story at t=0
    mixed = story.overlay(sfx)

    # Force final length to exactly match story
    if len(mixed) != story_len_ms:
        mixed = mixed[:story_len_ms]

    # Ensure properties track story
    mixed = mixed.set_frame_rate(story.frame_rate).set_channels(story.channels).set_sample_width(story.sample_width)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mixed.export(out_path, format="wav")

    print(f"✅ Saved: {out_path}")
    print(f"Final length (story-locked): {human_ms(story_len_ms)}")
    return 0


def main() -> int:
    # Resolve defaults relative to the repository root so the script works from any CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_story = os.path.join(project_root, "output", "story.wav")
    default_sfx = os.path.join(project_root, "output", "sfx.wav")
    default_out = os.path.join(project_root, "output", "final.wav")

    parser = argparse.ArgumentParser(
        description="Combine narration (story.wav) with SFX (sfx.wav) into final.wav of story length"
    )
    parser.add_argument("--story", default=default_story, help="Path to story narration WAV")
    parser.add_argument("--sfx", default=default_sfx, help="Path to SFX WAV")
    parser.add_argument("--out", default=default_out, help="Output WAV path")
    parser.add_argument("--tolerance-ms", type=int, default=100, help="Allowed length difference before strict mode aborts")
    parser.add_argument("--strict", action="store_true", help="Abort if lengths differ beyond tolerance")

    args = parser.parse_args()

    # Prefer torchaudio for robust resampling/channel handling if available
    try:
        import torchaudio  # noqa: F401
        return combine_with_torchaudio(args.story, args.sfx, args.out, args.tolerance_ms, args.strict)
    except Exception as e:
        print(f"ℹ️  Falling back to pydub mix ({e})")
        try:
            return combine_with_pydub(args.story, args.sfx, args.out, args.tolerance_ms, args.strict)
        except Exception as e2:
            print(f"❌ Failed to combine with pydub: {e2}")
            return 3


if __name__ == "__main__":
    sys.exit(main())


