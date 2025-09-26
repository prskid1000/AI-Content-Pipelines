import os
import re
import sys
import time
import shutil
import subprocess
from typing import List, Dict, Optional, Tuple


def print_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


class VideoFromScenes:
    """Create per-scene videos from 2.timeline.script.txt which contains both durations and scene information.

    Inputs (relative to this script by default):
    - ../../gen.audio/input/2.timeline.script.txt → lines: "<duration>: scene_name = description, actor = dialogue" or "<duration> : ..." (silence)
    - ../output/scene/                           → still images; file name equals scene name (e.g., scene_1.1.png)

    Output:
    - ../output/video/scene_<id>.mp4 → one file per scene, duration equals the paired duration
    - For silence segments (marked with "..."), uses the previous scene's duration
    - Scene information is extracted directly from the timeline script
    """

    def __init__(self,
                 durations_file: Optional[str] = None,
                 scenes_file: Optional[str] = None,
                 scenes_image_dir: Optional[str] = None,
                 output_video_dir: Optional[str] = None,
                 fps: int = 24,
                 resume: bool = True,
                 force: bool = False,
                 start_index: int = 0,
                 min_ok_size_bytes: int = 1024,
                 target_width: int = 854,
                 target_height: int = 480):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.durations_file = durations_file or os.path.normpath(os.path.join(script_dir, "../../gen.audio/input/2.timeline.script.txt"))
        # Note: scenes_file parameter kept for backward compatibility but no longer used
        self.scenes_image_dir = scenes_image_dir or os.path.normpath(os.path.join(script_dir, "../output/scene"))
        self.output_video_dir = output_video_dir or os.path.normpath(os.path.join(script_dir, "../output/video"))
        self.fps = fps
        self.resume = resume
        self.force = force
        self.start_index = max(0, int(start_index))
        self.min_ok_size_bytes = max(0, int(min_ok_size_bytes))
        self.target_width = int(target_width)
        self.target_height = int(target_height)
        # Root output (for merged file)
        self.output_root_dir = os.path.normpath(os.path.join(script_dir, "../output"))
        os.makedirs(self.output_root_dir, exist_ok=True)

        os.makedirs(self.output_video_dir, exist_ok=True)

        # Build a quick index of images available
        self.available_images_lower = set()
        if os.path.isdir(self.scenes_image_dir):
            for fname in os.listdir(self.scenes_image_dir):
                self.available_images_lower.add(fname.lower())
        
        # Track silence block processing
        self._current_silence_block_count = 0
        self._current_silence_position = 0

    # ---------- IO helpers ----------
    def read_timeline_data(self) -> Tuple[List[float], List[Dict[str, str]]]:
        """Read durations and scene information from 2.timeline.script.txt.
        
        Returns:
            Tuple of (durations, scenes) where scenes contain scene_name, scene_id, description
        
        Format: duration: scene_name = description, actor = dialogue
        For silence segments (marked with "..."), uses the previous scene's duration.
        """
        durations: List[float] = []
        scenes: List[Dict[str, str]] = []
        last_valid_duration = 0.0  # For handling silence segments
        last_scene_name = ""  # For handling silence segments
        
        try:
            # Read all lines first to avoid file position issues
            with open(self.durations_file, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            
            for line_num, line in enumerate(all_lines, 1):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                
                # Split on first colon to get duration
                parts = line.split(":", 1)
                if len(parts) < 2:
                    continue
                    
                duration_str = parts[0].strip()
                rest = parts[1].strip()
                
                try:
                    duration = float(duration_str)
                    
                    # Check if this is a silence segment (marked with dots only)
                    if rest and all(c == '.' for c in rest):
                        # Smart silence handling: try to interpolate scene from context
                        silence_scene = self._determine_silence_scene_with_distribution(
                            line_num, duration, last_scene_name, all_lines, line_num - 1, durations, scenes
                        )
                        
                        durations.append(duration)
                        scenes.append(silence_scene)
                        print_flush(f"🔇 Silence segment {line_num}: using {silence_scene['scene_name']} duration {duration:.3f}s")
                    else:
                        # Regular scene with dialogue - extract scene info
                        scene_info = self._parse_scene_from_timeline(rest)
                        if scene_info:
                            durations.append(duration)
                            scenes.append(scene_info)
                            last_valid_duration = duration
                            last_scene_name = scene_info["scene_name"]
                        else:
                            print_flush(f"⚠️ Could not parse scene info from line {line_num}: {line}")
                            continue
                        
                except ValueError:
                    # Skip malformed lines
                    print_flush(f"⚠️ Skipping malformed line {line_num}: {line}")
                    continue
                        
        except FileNotFoundError:
            print_flush(f"❌ Timeline script file not found: {self.durations_file}")
            return [], []
        except Exception as e:
            print_flush(f"❌ Error reading timeline script file: {e}")
            return [], []
            
        print_flush(f"📋 Loaded {len(durations)} timeline entries from {self.durations_file}")
        return durations, scenes
    
    def _determine_silence_scene_with_distribution(self, line_num: int, duration: float, last_scene_name: str, 
                                all_lines: List[str], current_line_idx: int, existing_durations: List[float], existing_scenes: List[Dict[str, str]]) -> Dict[str, str]:
        """Smart silence scene determination with distribution for multiple adjacent silence segments."""
        
        # Peek ahead to find next non-silence scene and count silence segments
        next_scene_name = None
        silence_count = 1  # Current silence segment
        
        # Look ahead from current position
        for i in range(current_line_idx + 1, len(all_lines)):
            next_line = all_lines[i].strip()
            if not next_line or ":" not in next_line:
                continue
                
            parts = next_line.split(":", 1)
            if len(parts) < 2:
                continue
                
            rest = parts[1].strip()
            if rest and all(c == '.' for c in rest):
                silence_count += 1  # Count additional silence segments
            else:
                # Found next non-silence line
                next_scene_info = self._parse_scene_from_timeline(rest)
                if next_scene_info:
                    next_scene_name = next_scene_info["scene_name"]
                break
        
        # Update silence block tracking
        if silence_count != self._current_silence_block_count:
            # New silence block detected
            self._current_silence_block_count = silence_count
            self._current_silence_position = 0
        else:
            # Continue in same silence block
            self._current_silence_position += 1
        
        # Decision logic for silence scene
        if last_scene_name and next_scene_name:
            # We have both previous and next scenes
            last_scene_id = self._extract_scene_id(last_scene_name) if last_scene_name.startswith("scene_") else last_scene_name
            next_scene_id = self._extract_scene_id(next_scene_name) if next_scene_name.startswith("scene_") else next_scene_name
            
            # Try to interpolate or distribute
            chosen_scene, chosen_id, reason = self._distribute_silence_scene(
                last_scene_name, last_scene_id, next_scene_name, next_scene_id, silence_count, self._current_silence_position
            )
        elif next_scene_name:
            # Only next scene available - use it
            chosen_scene = next_scene_name
            chosen_id = self._extract_scene_id(next_scene_name)
            reason = f"anticipated {next_scene_name}"
        elif last_scene_name:
            # Only previous scene available - use it
            chosen_scene = last_scene_name
            chosen_id = self._extract_scene_id(last_scene_name)
            reason = f"continued from {last_scene_name}"
        else:
            # No context - use default
            chosen_scene = "scene_silence"
            chosen_id = "silence"
            reason = "no context available"
        
        return {
            "scene_name": chosen_scene,
            "scene_id": chosen_id,
            "description": f"[Silence segment - {reason}]",
            "original_line": f"{duration}: ..."
        }
    
    def _interpolate_scene_id(self, scene_id1: str, scene_id2: str) -> Optional[str]:
        """Interpolate scene ID between two scenes (e.g., scene_1.4 → scene_1.6 = scene_1.5)."""
        try:
            # Parse scene IDs like "1.4", "1.6"
            if "." in scene_id1 and "." in scene_id2:
                parts1 = scene_id1.split(".")
                parts2 = scene_id2.split(".")
                
                if len(parts1) == 2 and len(parts2) == 2:
                    chapter1, scene1 = int(parts1[0]), int(parts1[1])
                    chapter2, scene2 = int(parts2[0]), int(parts2[1])
                    
                    # Only interpolate if same chapter and gap of exactly 2 (e.g., 1.4 → 1.6)
                    if chapter1 == chapter2 and scene2 - scene1 == 2:
                        # Interpolate the middle scene (gap of 1)
                        middle_scene = scene1 + 1
                        return f"{chapter1}.{middle_scene}"
            
            return None
        except (ValueError, IndexError):
            return None
    
    def _distribute_silence_scene(self, last_scene_name: str, last_scene_id: str, 
                                 next_scene_name: str, next_scene_id: str, 
                                 silence_count: int, silence_position: int) -> Tuple[str, str, str]:
        """Distribute silence scenes between previous and next scenes."""
        
        # Case 1: Single silence with exact gap of 2 (e.g., 1.4 ... 1.6) -> interpolate to 1.5
        if silence_count == 1:
            interpolated_scene = self._interpolate_scene_id(last_scene_id, next_scene_id)
            if interpolated_scene:
                return f"scene_{interpolated_scene}", interpolated_scene, f"interpolated {interpolated_scene} between {last_scene_name} → {next_scene_name}"
            else:
                return last_scene_name, self._extract_scene_id(last_scene_name), f"continued from {last_scene_name}"
        
        # Case 2: Multiple silence segments - split in half between previous and next scenes
        else:
            # Split consecutive silence segments: first half → previous, second half → next
            mid_point = silence_count // 2
            
            # We need a way to track which silence segment this is within the block
            # Since we can't easily track position in current architecture, we'll use line_num as proxy
            # This is a simplified approach - ideally we'd track the silence block position
            
            if silence_position < mid_point:
                # First half - assign to previous scene
                return last_scene_name, self._extract_scene_id(last_scene_name), f"first half of {silence_count} silence block → {last_scene_name}"
            else:
                # Second half - assign to next scene
                return next_scene_name, self._extract_scene_id(next_scene_name), f"second half of {silence_count} silence block → {next_scene_name}"
    
    def combine_adjacent_scenes(self, durations: List[float], scenes: List[Dict[str, str]]) -> Dict[str, Dict[str, any]]:
        """Combine adjacent same scenes into single large durations.
        
        Returns:
            Dict mapping scene_name to combined scene info with total duration
        """
        if not durations or not scenes or len(durations) != len(scenes):
            return {}
        
        combined_scenes: Dict[str, Dict[str, any]] = {}
        scene_order: List[str] = []  # Track order of first appearance
        
        for i, (duration, scene) in enumerate(zip(durations, scenes)):
            scene_name = scene.get("scene_name", f"scene_{i+1}")
            
            if scene_name not in combined_scenes:
                # First occurrence of this scene
                combined_scenes[scene_name] = {
                    "scene_name": scene_name,
                    "scene_id": scene.get("scene_id", scene_name),
                    "description": scene.get("description", ""),
                    "total_duration": duration,
                    "first_occurrence": i,
                    "occurrences": [i]
                }
                scene_order.append(scene_name)
            else:
                # Additional occurrence - add to total duration
                combined_scenes[scene_name]["total_duration"] += duration
                combined_scenes[scene_name]["occurrences"].append(i)
        
        print_flush(f"🔗 Combined {len(durations)} timeline entries into {len(combined_scenes)} unique scenes")
        
        # Check if each scene appears only in contiguous blocks
        for scene_name, info in combined_scenes.items():
            occurrences = info["occurrences"]
            if len(occurrences) > 1:
                # Check if all occurrences are contiguous
                for i in range(1, len(occurrences)):
                    if occurrences[i] != occurrences[i-1] + 1:
                        print_flush(f"❌ Scene '{scene_name}' appears non-contiguously at indices {occurrences}")
                        print_flush("❌ Each scene must appear in contiguous blocks only. Exiting.")
                        return {}
        
        # Log combined durations
        for scene_name in scene_order:
            info = combined_scenes[scene_name]
            print_flush(f"📋 {scene_name}: {info['total_duration']:.3f}s ({len(info['occurrences'])} segments)")
        
        return combined_scenes
    
    def _extract_scene_id(self, scene_name: str) -> str:
        """Extract scene ID from scene name (e.g., 'scene_1.1' -> '1.1')."""
        if scene_name.startswith("scene_"):
            return scene_name[6:]  # Remove "scene_" prefix
        return scene_name
    
    def _parse_scene_from_timeline(self, rest: str) -> Optional[Dict[str, str]]:
        """Parse scene information from timeline script line.
        
        Format: scene_name = description, actor = dialogue
        """
        if " = " not in rest:
            return None
            
        # Split on " = " to separate scene_name from description
        parts = rest.split(" = ", 1)
        if len(parts) < 2:
            return None
            
        scene_name = parts[0].strip()
        description_part = parts[1].strip()
        
        # Extract just the visual description (before the first comma with actor)
        if ", " in description_part:
            # Find the last ", actor_name = " pattern to separate description from dialogue
            last_comma_idx = description_part.rfind(", ")
            if last_comma_idx > 0:
                # Check if this looks like an actor assignment
                after_comma = description_part[last_comma_idx + 2:]
                if " = " in after_comma:
                    description = description_part[:last_comma_idx].strip()
                else:
                    description = description_part
            else:
                description = description_part
        else:
            description = description_part
        
        return {
            "scene_name": scene_name,
            "scene_id": self._extract_scene_id(scene_name),
            "description": description,
            "original_line": f"{scene_name} = {description_part}"
        }


    # ---------- Image location ----------
    def find_image_for_scene(self, scene_name: str, scene_id: str) -> Optional[str]:
        """Find image using fixed naming: scene_{id}.png inside scenes folder."""
        filename = f"scene_{scene_id}.png" if scene_id else f"{scene_name}.png"
        full_path = os.path.join(self.scenes_image_dir, filename)
        if os.path.exists(full_path):
            return full_path
        return None

    # ---------- Video export ----------
    def export_video(self, image_path: str, duration: float, output_path: str) -> bool:
        """Export a still image to an MP4 video of the given duration using ffmpeg only."""
        duration = max(0.01, float(duration))

        # ffmpeg
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            print_flush("❌ ffmpeg is not available in PATH. Please install ffmpeg.")
            return False

        # Letterbox to target resolution with preserved aspect ratio
        vf_chain = (
            f"scale={self.target_width}:{self.target_height}:force_original_aspect_ratio=decrease,"
            f"pad={self.target_width}:{self.target_height}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
        )

        cmd = [
            ffmpeg_path,
            "-y",
            "-loop", "1",
            "-i", image_path,
            "-t", f"{duration:.3f}",
            "-vf", vf_chain,
            "-r", str(self.fps),
            "-c:v", "libx264",
            output_path,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            print_flush(f"❌ ffmpeg failed for {os.path.basename(image_path)}: {e}")
            return False


    def merge_videos_ffmpeg(self, segments: List[str], output_path: str) -> bool:
        """Merge a list of mp4 segments into one mp4 using ffmpeg concat demuxer.

        We re-encode the final output to ensure compatibility.
        """
        if not segments:
            return False

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            print_flush("❌ ffmpeg is not available in PATH. Please install ffmpeg.")
            return False

        # Build a concat list file
        list_path = os.path.join(self.output_video_dir, "concat_list.txt")
        try:
            with open(list_path, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(f"file '{seg.replace('\\\\', '/').replace('\\', '/')}'\n")
        except Exception as e:
            print_flush(f"❌ Could not write concat list: {e}")
            return False

        cmd = [
            ffmpeg_path,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-r", str(self.fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path,
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.CalledProcessError as e:
            print_flush(f"❌ ffmpeg merge failed: {e}")
            return False

    # ---------- Main workflow ----------
    def run(self) -> int:
        print_flush("🚀 Building per-scene videos from timeline script...")
        print_flush(f"📁 Timeline:  {self.durations_file}")
        print_flush(f"🖼️  Images:    {self.scenes_image_dir}")
        print_flush(f"🎬 Output:    {self.output_video_dir}")
        print_flush(f"⬆️  Resolution: {self.target_width}x{self.target_height}")

        durations, scenes = self.read_timeline_data()

        if not durations or not scenes:
            return 1

        if len(durations) != len(scenes):
            print_flush(f"⚠️ Count mismatch: timeline entries={len(durations)} vs scenes={len(scenes)}. Proceeding with min count.")

        # Combine adjacent same scenes into single large durations
        combined_scenes = self.combine_adjacent_scenes(durations, scenes)
        if not combined_scenes:
            print_flush("❌ Failed to combine scenes or scenes appear non-contiguously.")
            return 1

        created_count = 0
        skipped_existing_count = 0
        missing_images: List[str] = []
        scene_video_paths: List[str] = []  # Final sequence for merging (in original order)
        merged_duration_seconds: float = 0.0

        start = time.time()
        
        # Generate videos for unique scenes with their combined durations
        print_flush("\n🎬 Generating videos for unique scenes with combined durations...")
        scene_video_map: Dict[str, str] = {}  # Map scene_name -> video_file_path
        
        for scene_name, scene_info in combined_scenes.items():
            scene_id = scene_info.get("scene_id", scene_name)
            total_duration = scene_info["total_duration"]
            
            image_path = self.find_image_for_scene(scene_name=scene_name, scene_id=scene_id)
            if not image_path:
                missing_images.append(scene_name)
                print_flush(f"❌ Missing image for {scene_name} (id={scene_id})")
                continue

            # Create video file for this scene with combined duration
            out_name = f"{scene_name}.mp4"
            out_path = os.path.join(self.output_video_dir, out_name)

            # Resume logic: skip if output exists and looks valid, unless forced
            if (not self.force) and self.resume and os.path.exists(out_path):
                try:
                    size = os.path.getsize(out_path)
                except OSError:
                    size = 0
                if size >= self.min_ok_size_bytes:
                    print_flush(f"⏭️  Skipping {out_name} (exists, size={size} bytes)")
                    skipped_existing_count += 1
                    scene_video_map[scene_name] = out_path
                    continue

            print_flush(f"🎞️  Rendering {out_name} duration={total_duration:.3f}s from {os.path.basename(image_path)}")
            ok = self.export_video(image_path=image_path, duration=total_duration, output_path=out_path)
            if ok:
                created_count += 1
                scene_video_map[scene_name] = out_path
        
        # Build final sequence in original timeline order
        print_flush(f"\n🔗 Building final sequence from {len(scene_video_map)} unique scenes...")
        processed_scenes = set()  # Track which scenes we've already added to avoid duplicates
        
        for idx in range(len(durations)):
            scene = scenes[idx]
            scene_name = scene.get("scene_name") or f"scene_{idx+1}"
            
            # Only add each scene once (first occurrence)
            if scene_name in processed_scenes:
                continue
                
            if scene_name in scene_video_map:
                scene_video_paths.append(scene_video_map[scene_name])
                merged_duration_seconds += combined_scenes[scene_name]["total_duration"]
                processed_scenes.add(scene_name)
                print_flush(f"📋 Added {scene_name} to sequence (duration: {combined_scenes[scene_name]['total_duration']:.3f}s)")
            else:
                print_flush(f"❌ No video available for {scene_name}")
                merged_duration_seconds += combined_scenes.get(scene_name, {}).get("total_duration", 0.0)

        elapsed = time.time() - start
        print_flush("\n📊 VIDEO SUMMARY:")
        print_flush(f"   • Total timeline entries: {len(durations)}")
        print_flush(f"   • Unique scenes:         {len(combined_scenes)}")
        print_flush(f"   • Videos created:        {created_count}")
        print_flush(f"   • Skipped existing:      {skipped_existing_count}")
        print_flush(f"   • Missing images:        {len(missing_images)}")
        print_flush(f"   • Final segments:        {len(scene_video_paths)}")
        if missing_images:
            print_flush("   • Missing for scenes:")
            for name in missing_images[:15]:
                print_flush(f"     - {name}")
            if len(missing_images) > 15:
                print_flush(f"     ... and {len(missing_images) - 15} more")
        print_flush(f"   • FPS:               {self.fps}")
        print_flush(f"   • Time:              {elapsed:.2f}s")

        # Merge into a single video in ../output/merged.mp4
        merged_out = os.path.join(self.output_root_dir, "merged.mp4")
        if scene_video_paths:
            print_flush(f"\n🔗 Merging {len(scene_video_paths)} unique scene videos → {merged_out}")
            merged_ok = self.merge_videos_ffmpeg(scene_video_paths, merged_out)
            if merged_ok:
                print_flush(f"✅ Merged video created: {merged_out}")
                print_flush(f"⏱️  Merged duration (sum of combined scenes): {merged_duration_seconds:.3f}s ({merged_duration_seconds/60:.2f} min)")
            else:
                print_flush("❌ Failed to create merged video")

        processed = created_count + skipped_existing_count
        return 0 if processed > 0 else 1


def main() -> int:
    # CLI: [timeline_script] [scenes_dir] [output_dir] [fps] [--force] [--no-resume] [--start-index N]
    import argparse

    parser = argparse.ArgumentParser(description="Create per-scene videos from timeline script and scene images")
    parser.add_argument("durations", nargs="?", default=None, help="Path to 2.timeline.script.txt")
    parser.add_argument("scenes_dir", nargs="?", default=None, help="Directory with scene images")
    parser.add_argument("output_dir", nargs="?", default=None, help="Directory to write videos")
    parser.add_argument("fps", nargs="?", type=int, default=24, help="Frames per second")
    parser.add_argument("--force", action="store_true", help="Overwrite existing videos")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume (do not skip existing videos)")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this scene index (0-based)")
    parser.add_argument("--min-ok-size", type=int, default=1024, help="Minimum file size to consider existing video valid")

    args = parser.parse_args()

    job = VideoFromScenes(
        durations_file=args.durations,
        scenes_image_dir=args.scenes_dir,
        output_video_dir=args.output_dir,
        fps=args.fps,
        resume=not args.no_resume,
        force=args.force,
        start_index=args.start_index,
        min_ok_size_bytes=args.min_ok_size,
    )
    return job.run()


if __name__ == "__main__":
    sys.exit(main())


