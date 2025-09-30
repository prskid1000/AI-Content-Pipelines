import os
import json
import time
import shutil
import re
import argparse
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Feature flags
ENABLE_RESUMABLE_MODE = True
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them

# Video configuration constants
VIDEO_WIDTH = 1024
VIDEO_HEIGHT = 576
FRAMES_PER_SECOND = 24

ENABLE_MOTION = True
ENABLE_SCENE = True
ENABLE_LOCATION = True  # Set to True to replace {{loc_1}} with location descriptions from 3.location.txt

ART_STYLE = "Anime"


class ResumableState:
    """Manages resumable state for expensive video animation operations."""
    
    def __init__(self, checkpoint_dir: str, script_name: str, force_start: bool = False):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.checkpoint_dir / f"{script_name}.state.json"
        
        # If force_start is True, remove existing checkpoint and start fresh
        if force_start and self.state_file.exists():
            try:
                self.state_file.unlink()
                print("Force start enabled - removed existing checkpoint")
            except Exception as ex:
                print(f"WARNING: Failed to remove checkpoint for force start: {ex}")
        
        self.state = self._load_state()
    
    def _load_state(self) -> dict:
        """Load state from checkpoint file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as ex:
                print(f"WARNING: Failed to load checkpoint file: {ex}")
        
        return {
            "videos": {
                "completed": [],
                "results": {}
            }
        }
    
    def _save_state(self):
        """Save current state to checkpoint file."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as ex:
            print(f"WARNING: Failed to save checkpoint: {ex}")
    
    def is_video_complete(self, scene_name: str) -> bool:
        """Check if video animation is complete."""
        return scene_name in self.state["videos"]["completed"]
    
    def get_video_result(self, scene_name: str) -> dict:
        """Get video animation result."""
        return self.state["videos"]["results"].get(scene_name, {})
    
    def set_video_result(self, scene_name: str, result: dict):
        """Set video animation result and mark as complete."""
        self.state["videos"]["results"][scene_name] = result
        if scene_name not in self.state["videos"]["completed"]:
            self.state["videos"]["completed"].append(scene_name)
        self._save_state()
    
    def cleanup(self):
        """Clean up tracking files based on configuration setting."""
        try:
            if CLEANUP_TRACKING_FILES and self.state_file.exists():
                self.state_file.unlink()
                print("All operations completed successfully - tracking files cleaned up")
            else:
                print("All operations completed successfully - tracking files preserved")
        except Exception as ex:
            print(f"WARNING: Error in cleanup: {ex}")
    
    def validate_and_cleanup_results(self) -> int:
        """Validate that all completed video files actually exist and clean up missing entries.
        
        Returns:
            int: Number of entries cleaned up (removed from completed list)
        """
        cleaned_count = 0
        videos_to_remove = []
        
        # Check each completed video
        for scene_name in self.state["videos"]["completed"]:
            result = self.state["videos"]["results"].get(scene_name, {})
            file_paths = result.get('paths', [])
            
            # Check if all video files actually exist
            all_exist = True
            if not file_paths:
                all_exist = False
            else:
                for file_path in file_paths:
                    if not file_path or not os.path.exists(file_path):
                        all_exist = False
                        break
            
            if not all_exist:
                print(f"Precheck: File missing for {scene_name} - marking as not completed")
                videos_to_remove.append(scene_name)
                cleaned_count += 1
        
        # Remove invalid entries
        for scene_name in videos_to_remove:
            if scene_name in self.state["videos"]["completed"]:
                self.state["videos"]["completed"].remove(scene_name)
            if scene_name in self.state["videos"]["results"]:
                del self.state["videos"]["results"][scene_name]
        
        # Save cleaned state if any changes were made
        if cleaned_count > 0:
            self._save_state()
            print(f"Precheck: Cleaned up {cleaned_count} invalid entries from checkpoint")
        
        return cleaned_count
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        completed = len(self.state["videos"]["completed"])
        total = len(self.state["videos"]["results"]) + len([k for k in self.state["videos"]["results"].keys() if k not in self.state["videos"]["completed"]])
        
        return f"Progress: Videos({completed}/{total})"


def print_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

class VideoAnimator:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188/"):
        self.comfyui_url = comfyui_url
        # ComfyUI saves videos under this folder
        self.comfyui_output_folder = "../../ComfyUI/output"
        # ComfyUI input folder where we need to copy scene images
        self.comfyui_input_folder = "../../ComfyUI/input"
        # Scene images directory
        self.scene_images_dir = "../../gen.image/output/scene"
        # Final destination for videos
        self.final_output_dir = "../output/animation"
        # Directory for storing extracted frames
        self.frames_output_dir = "../output/frames"
        # Timeline script file with durations (single source of truth)
        self.timeline_file = "../../gen.audio/input/2.timeline.script.txt"
        # Animation workflow file
        self.workflow_file = "../workflow/animate.json"
        # Character data file
        self.character_file = "../../gen.image/input/3.character.txt"
        # Location data file
        self.location_file = "../../gen.image/input/3.location.txt"
        # Motion data file
        self.motion_file = "../input/2.motion.txt"

        # Time estimation tracking
        self.processing_times = []
        self.start_time = None

        # Create output directories
        os.makedirs(self.final_output_dir, exist_ok=True)
        os.makedirs(self.frames_output_dir, exist_ok=True)
        os.makedirs(self.comfyui_input_folder, exist_ok=True)

        # Track silence block processing
        self._current_silence_block_count = 0
        self._current_silence_position = 0

    def estimate_remaining_time(self, current_scene: int, total_scenes: int, scene_processing_time: float = None, scene_description: str = None, duration: float = None) -> str:
        """Estimate remaining time based on processing history and content characteristics"""
        if not self.processing_times:
            return "No data available"
        
        # Calculate base average processing time per scene using ALL previous entries
        avg_time_per_scene = sum(self.processing_times) / len(self.processing_times)
        
        # If we have current scene processing time, include it in the calculation
        if scene_processing_time:
            # Use all previous entries plus current entry for more accurate estimation
            all_times = self.processing_times + [scene_processing_time]
            estimated_time_per_scene = sum(all_times) / len(all_times)
        else:
            estimated_time_per_scene = avg_time_per_scene
        
        # Apply content-based adjustments if we have scene description and duration
        if scene_description and duration is not None:
            word_count = len(scene_description.split())
            char_count = len(scene_description)
            
            # Calculate complexity factor based on content characteristics
            # More complex scenes with more characters and longer durations take longer
            word_factor = 1.0 + (word_count - 20) * 0.02  # Base 20 words, +2% per word over/under
            char_factor = 1.0 + (char_count - 100) * 0.001  # Base 100 chars, +0.1% per char over/under
            
            # Duration factor - longer videos take more time to generate
            duration_factor = 1.0 + (duration - 5.0) * 0.1  # Base 5 seconds, +10% per second over/under
            
            # Check for complexity indicators
            complexity_factor = 1.0
            if any(word in scene_description.lower() for word in ['complex', 'detailed', 'multiple', 'many', 'crowd', 'action', 'motion']):
                complexity_factor = 1.4  # 40% longer for complex scenes
            elif any(word in scene_description.lower() for word in ['simple', 'basic', 'single', 'minimal', 'static']):
                complexity_factor = 0.8  # 20% shorter for simple scenes
            
            # Combine factors (cap at reasonable bounds)
            complexity_factor = min(2.5, max(0.5, (word_factor + char_factor + duration_factor) / 3 * complexity_factor))
            estimated_time_per_scene *= complexity_factor
        
        remaining_scenes = total_scenes - current_scene
        estimated_remaining_seconds = remaining_scenes * estimated_time_per_scene
        
        # Convert to human readable format
        if estimated_remaining_seconds < 60:
            return f"~{estimated_remaining_seconds:.0f}s"
        elif estimated_remaining_seconds < 3600:
            minutes = estimated_remaining_seconds / 60
            return f"~{minutes:.1f}m"
        else:
            hours = estimated_remaining_seconds / 3600
            return f"~{hours:.1f}h"
    
    def format_processing_time(self, processing_time: float) -> str:
        """Format processing time in human readable format"""
        if processing_time < 60:
            return f"{processing_time:.1f}s"
        elif processing_time < 3600:
            minutes = processing_time / 60
            return f"{minutes:.1f}m"
        else:
            hours = processing_time / 3600
            return f"{hours:.1f}h"

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
            with open(self.timeline_file, "r", encoding="utf-8") as f:
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
            print_flush(f"❌ Timeline script file not found: {self.timeline_file}")
            return [], []
        except Exception as e:
            print_flush(f"❌ Error reading timeline script file: {e}")
            return [], []
            
        print_flush(f"📋 Loaded {len(durations)} timeline entries from {self.timeline_file}")
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
            
            if silence_position < mid_point:
                # First half - assign to previous scene
                return last_scene_name, self._extract_scene_id(last_scene_name), f"first half of {silence_count} silence block → {last_scene_name}"
            else:
                # Second half - assign to next scene
                return next_scene_name, self._extract_scene_id(next_scene_name), f"second half of {silence_count} silence block → {next_scene_name}"
    
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

    def _read_character_data(self) -> dict[str, str]:
        """Parse character data from input file."""
        characters = {}
        try:
            with open(self.character_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            entries = [entry.strip() for entry in content.split('\n\n') if entry.strip()]
            for entry in entries:
                match = re.match(r'\(\(([^)]+)\)\):\s*(.+)', entry, re.DOTALL)
                if match:
                    characters[match.group(1).strip()] = match.group(2).strip()
        except Exception as e:
            print_flush(f"ERROR: Failed to read character data: {e}")
        return characters

    def _read_motion_data(self) -> dict[str, str]:
        """Parse motion data from input file.
        
        Returns:
            Dict mapping motion IDs to motion descriptions
        """
        motions = {}
        try:
            with open(self.motion_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            for line in lines:
                # Parse format: (motion_1.1) ((character)) {motion description}; ((character2)) {motion description}.
                match = re.match(r'\(([^)]+)\)\s*(.+)', line)
                if match:
                    motion_id = match.group(1).strip()
                    motion_content = match.group(2).strip()
                    motions[motion_id] = motion_content
                    
        except FileNotFoundError:
            print_flush(f"⚠️ Motion file not found: {self.motion_file}")
        except Exception as e:
            print_flush(f"ERROR: Failed to read motion data: {e}")
        return motions

    def _read_location_data(self) -> dict[str, str]:
        """Parse location data from input file."""
        locations = {}
        try:
            with open(self.location_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            entries = [entry.strip() for entry in content.split('\n\n') if entry.strip()]
            for entry in entries:
                # Match format: {{loc_id}} description...
                match = re.match(r'\{\{([^}]+)\}\}\s*(.+)', entry, re.DOTALL)
                if match:
                    locations[match.group(1).strip()] = match.group(2).strip()
        except Exception as e:
            print_flush(f"ERROR: Failed to read location data: {e}")
        return locations

    def _get_character_details(self, character_names: list[str], characters_data: dict[str, str]) -> str:
        """Get character details text for the given character names."""
        if not character_names or not characters_data:
            return ""
        details = [f"{char} WITH {{{characters_data[char]}}}." 
                  for char in character_names if char in characters_data]
        return "\n".join(details)

    def _replace_location_references(self, scene_description: str, locations_data: dict[str, str]) -> str:
        """Replace {{loc_id}} references with actual location descriptions."""
        if not ENABLE_LOCATION or not locations_data:
            return scene_description
        
        def replace_func(match):
            full_match = match.group(0)
            # Try to match {{loc_id, description}} or {{loc_id}}
            if ',' in full_match:
                # Full format: {{loc_id, description}} - replace with full description from file
                loc_id = match.group(1).strip()
                if loc_id in locations_data:
                    return f"{{{{{locations_data[loc_id]}}}}}"
                return full_match
            else:
                # Simple reference: {{loc_id}} - replace with description
                loc_id = match.group(1).strip()
                if loc_id in locations_data:
                    return f"{{{{{locations_data[loc_id]}}}}}"
                return full_match
        
        # Replace {{loc_id}} patterns with location descriptions
        result = re.sub(r'\{\{([^}]+)\}\}', replace_func, scene_description)
        return result

    def _extract_characters_from_scene(self, scene_description: str) -> list[str]:
        """Extract character names from scene description."""
        character_matches = re.findall(r'\(\(([^)]+)\)\)(?!\))', scene_description)
        seen = set()
        unique_characters = []
        for char in character_matches:
            char = char.strip()
            if char and char not in seen:
                seen.add(char)
                unique_characters.append(char)
        return unique_characters

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
        
        # Log combined durations
        for scene_name in scene_order:
            info = combined_scenes[scene_name]
            print_flush(f"📋 {scene_name}: {info['total_duration']:.3f}s ({len(info['occurrences'])} segments)")
        
        return combined_scenes

    def _calculate_frame_count(self, duration: float) -> int:
        """Calculate number of frames based on duration and frame rate."""
        return max(1, int(duration * FRAMES_PER_SECOND))
    
    def _calculate_video_chunks(self, duration: float) -> List[Tuple[str, int]]:
        """Calculate video chunks for long durations.
        
        Returns:
            List of (chunk_filename, frame_count) tuples
        """
        total_frames = self._calculate_frame_count(duration)
        max_frames_per_chunk = 8 * FRAMES_PER_SECOND + 1
        
        if total_frames <= max_frames_per_chunk:
            # Single video
            return [("", total_frames)]
        
        # Multiple chunks needed
        chunks = []
        remaining_frames = total_frames
        chunk_number = 1
        
        while remaining_frames > 0:
            chunk_frames = min(remaining_frames, max_frames_per_chunk)
            chunks.append((f"_{chunk_number}", chunk_frames))
            remaining_frames -= chunk_frames
            chunk_number += 1
        
        return chunks

    def _copy_saved_frames_to_output(self, chunk_scene_id: str) -> None:
        """Copy saved frames from ComfyUI output to output/frames directory with proper naming.
        
        Args:
            chunk_scene_id: The chunk scene ID (e.g., "scene_1.1_1")
        """
        try:
            frame_prefix = f"{chunk_scene_id}_frame"
            
            if not os.path.isdir(self.comfyui_output_folder):
                return
            
            # Find all frames with the prefix
            matching_frames = []
            for root, _, files in os.walk(self.comfyui_output_folder):
                for name in files:
                    if name.startswith(frame_prefix) and name.endswith('.png'):
                        full_path = os.path.join(root, name)
                        try:
                            mtime = os.path.getmtime(full_path)
                            matching_frames.append((full_path, name, mtime))
                        except OSError:
                            continue
            
            if not matching_frames:
                return
            
            # Sort by modification time to get frames in order
            matching_frames.sort(key=lambda x: x[2])
            
            # Copy frames to output/frames directory with proper naming
            for frame_idx, (source_path, original_name, mtime) in enumerate(matching_frames):
                # Create proper naming: scene_1.1_chunk_id.frame_001.png
                frame_number = frame_idx + 1
                new_filename = f"{chunk_scene_id}.frame_{frame_number:03d}.png"
                dest_path = os.path.join(self.frames_output_dir, new_filename)
                
                try:
                    shutil.copy2(source_path, dest_path)
                except Exception as e:
                    print(f"⚠️ Failed to copy frame {original_name}: {e}")
            
            print(f"✅ Copied {len(matching_frames)} frames for {chunk_scene_id}")
            
        except Exception as e:
            print(f"❌ Error copying saved frames: {e}")

    def _find_last_saved_frame(self, scene_id: str, chunk_suffix: str = "") -> Optional[str]:
        """Find the last saved frame from ComfyUI output for a specific scene and chunk.
        
        Args:
            scene_id: The scene ID (e.g., "scene_1.1")
            chunk_suffix: The chunk suffix (e.g., "_1", "_2", etc.)
            
        Returns:
            Path to the last saved frame, or None if not found
        """
        try:
            frame_prefix = f"{scene_id}{chunk_suffix}_frame"
            
            if not os.path.isdir(self.comfyui_output_folder):
                return None
            
            # Find all frames with the prefix
            matching_frames = []
            for root, _, files in os.walk(self.comfyui_output_folder):
                for name in files:
                    if name.startswith(frame_prefix) and name.endswith('.png'):
                        full_path = os.path.join(root, name)
                        try:
                            mtime = os.path.getmtime(full_path)
                            matching_frames.append((full_path, mtime))
                        except OSError:
                            continue
            
            if not matching_frames:
                return None
            
            # Sort by modification time and get the latest
            matching_frames.sort(key=lambda x: x[1], reverse=True)
            latest_frame = matching_frames[0][0]
            
            return latest_frame
            
        except Exception as e:
            print(f"❌ Error finding last saved frame: {e}")
            return None

    def _extract_last_frame(self, video_path: str, output_image_path: str) -> bool:
        """Extract the last frame from a video file using ffmpeg.
        
        Args:
            video_path: Path to input video file
            output_image_path: Path where to save the extracted frame
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First get video duration and frame count using ffprobe
            duration_cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
            
            if duration_result.returncode != 0:
                print(f"⚠️ Could not get video duration, using fallback method...")
                # Fallback to simple method
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', '-0.1',  # Seek to 0.1 seconds before the end
                    '-vframes', '1',
                    '-vsync', '0',  # Disable frame dropping/duplication for accuracy
                    '-y',  # Overwrite output file
                    output_image_path
                ]
            else:
                # Calculate exact timestamp for last frame
                duration = float(duration_result.stdout.strip())
                # Calculate timestamp for the last frame (subtract one frame duration)
                frame_duration = 1.0 / FRAMES_PER_SECOND
                last_frame_time = duration - frame_duration
                
                # Ensure we don't go negative
                last_frame_time = max(0.0, last_frame_time)
                
                print(f"📹 Video duration: {duration:.3f}s, extracting frame at {last_frame_time:.3f}s")
                
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', str(last_frame_time),  # Seek to exact last frame time
                    '-vframes', '1',
                    '-vsync', '0',  # Disable frame dropping/duplication for accuracy
                    '-y',  # Overwrite output file
                    output_image_path
                ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✅ Extracted last frame: {os.path.basename(output_image_path)}")
                return True
            else:
                # Fallback: try extracting the first frame if last frame fails
                print(f"⚠️ Last frame extraction failed, trying first frame as fallback...")
                fallback_cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-vframes', '1',
                    '-y',
                    output_image_path
                ]
                
                fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=60)
                
                if fallback_result.returncode == 0:
                    print(f"✅ Extracted fallback frame: {os.path.basename(output_image_path)}")
                    return True
                else:
                    print(f"❌ FFmpeg error extracting frame: {result.stderr}")
                    print(f"❌ Fallback also failed: {fallback_result.stderr}")
                    return False
                
        except subprocess.TimeoutExpired:
            print("❌ FFmpeg timeout - frame extraction took too long")
            return False
        except FileNotFoundError:
            print("❌ FFmpeg not found - please install ffmpeg to extract frames")
            return False
        except Exception as e:
            print(f"❌ Error extracting frame: {e}")
            return False

    def _merge_video_chunks(self, chunk_paths: List[str], output_path: str) -> bool:
        """Merge video chunks into a single video file using ffmpeg, then remove first frame and duplicate second frame.
        
        Args:
            chunk_paths: List of chunk file paths in order
            output_path: Final merged video path
            
        Returns:
            True if successful, False otherwise
        """
        if not chunk_paths:
            return False
            
        if len(chunk_paths) == 1:
            # Only one chunk, process it to remove first frame and duplicate second
            return self._process_video_remove_first_frame(chunk_paths[0], output_path)
        
        try:
            # Create a temporary file list for ffmpeg
            temp_list_file = os.path.join(self.final_output_dir, f"temp_merge_list_{int(time.time())}.txt")
            
            with open(temp_list_file, 'w', encoding='utf-8') as f:
                for chunk_path in chunk_paths:
                    # Use absolute paths and escape backslashes for Windows
                    abs_path = os.path.abspath(chunk_path).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")
            
            # Create temporary merged file
            temp_merged_path = os.path.join(self.final_output_dir, f"temp_merged_{int(time.time())}.mp4")
            
            # Use ffmpeg to concatenate videos
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', temp_list_file,
                '-c', 'copy',  # Copy streams without re-encoding for speed
                '-y',  # Overwrite output file
                temp_merged_path
            ]
            
            print(f"Merging {len(chunk_paths)} video chunks...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Clean up temporary file
            try:
                os.remove(temp_list_file)
            except OSError:
                pass
            
            if result.returncode == 0:
                print(f"✅ Successfully merged chunks, now processing to remove first frame...")
                # Now process the merged video to remove first frame and duplicate second
                success = self._process_video_remove_first_frame(temp_merged_path, output_path)
                
                # Clean up temporary merged file
                try:
                    os.remove(temp_merged_path)
                except OSError:
                    pass
                    
                return success
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ FFmpeg timeout - merging took too long")
            return False
        except FileNotFoundError:
            print("❌ FFmpeg not found - please install ffmpeg to merge video chunks")
            print("   Download from: https://ffmpeg.org/download.html")
            print("   Or install via: winget install ffmpeg")
            return False
        except Exception as e:
            print(f"❌ Error merging video chunks: {e}")
            return False

    def _process_video_remove_first_frame(self, input_path: str, output_path: str) -> bool:
        """Remove first frame and duplicate the new first frame to maintain timing.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            frame_duration = 1.0 / FRAMES_PER_SECOND
            
            # Step 1: Remove first frame (trim from second frame onwards)
            trimmed_path = os.path.join(self.final_output_dir, f"temp_trimmed_{int(time.time())}.mp4")
            
            trim_cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ss', str(frame_duration),  # Skip first frame
                '-c', 'copy',
                '-y',
                trimmed_path
            ]
            
            print(f"Removing first frame...")
            result = subprocess.run(trim_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"❌ Failed to remove first frame: {result.stderr}")
                return False
            
            # Step 2: Extract the first frame of the trimmed video
            first_frame_path = os.path.join(self.final_output_dir, f"temp_first_frame_{int(time.time())}.png")
            
            extract_cmd = [
                'ffmpeg',
                '-i', trimmed_path,
                '-vframes', '1',
                '-y',
                first_frame_path
            ]
            
            print(f"Extracting first frame of trimmed video...")
            result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"❌ Failed to extract first frame: {result.stderr}")
                # Fallback: just use trimmed video
                shutil.move(trimmed_path, output_path)
                return True
            
            # Step 3: Create duplicate of the first frame
            duplicate_path = os.path.join(self.final_output_dir, f"temp_duplicate_{int(time.time())}.mp4")
            
            duplicate_cmd = [
                'ffmpeg',
                '-loop', '1',
                '-i', first_frame_path,
                '-t', str(frame_duration),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-r', str(FRAMES_PER_SECOND),
                '-y',
                duplicate_path
            ]
            
            print(f"Creating duplicate of first frame...")
            result = subprocess.run(duplicate_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"❌ Failed to create duplicate frame: {result.stderr}")
                # Fallback: just use trimmed video
                shutil.move(trimmed_path, output_path)
                return True
            
            # Step 4: Concatenate duplicate + trimmed video
            concat_list = os.path.join(self.final_output_dir, f"temp_concat_{int(time.time())}.txt")
            with open(concat_list, 'w') as f:
                f.write(f"file '{os.path.abspath(duplicate_path).replace(chr(92), chr(47))}'\n")
                f.write(f"file '{os.path.abspath(trimmed_path).replace(chr(92), chr(47))}'\n")
            
            final_cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list,
                '-c', 'copy',
                '-y',
                output_path
            ]
            
            print(f"Concatenating duplicate frame with trimmed video...")
            result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=60)
            
            # Clean up temporary files
            for temp_file in [trimmed_path, first_frame_path, duplicate_path, concat_list]:
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
            
            if result.returncode == 0:
                print(f"✅ Successfully processed video: {os.path.basename(output_path)}")
                return True
            else:
                print(f"❌ FFmpeg concatenation error: {result.stderr}")
                # Final fallback: use trimmed video
                shutil.move(trimmed_path, output_path)
                return True
                
        except Exception as e:
            print(f"❌ Error processing video: {e}")
            return False

    def _find_frame_file(self, scene_id: str, frame_number: int) -> Optional[str]:
        """Find a specific frame file for a scene.
        
        Args:
            scene_id: The scene ID (e.g., "scene_1.1")
            frame_number: The frame number to find (1-based)
            
        Returns:
            Path to the frame file, or None if not found
        """
        try:
            if not os.path.exists(self.frames_output_dir):
                return None
            
            # Look for frame files with the scene ID
            frame_files = []
            for filename in os.listdir(self.frames_output_dir):
                if filename.startswith(scene_id) and filename.endswith('.png'):
                    # Extract frame number from filename (e.g., "scene_1.1.frame_002.png" -> 2)
                    frame_match = re.search(r'\.frame_(\d+)\.png$', filename)
                    if frame_match:
                        file_frame_num = int(frame_match.group(1))
                        frame_files.append((file_frame_num, filename))
            
            # Sort by frame number
            frame_files.sort(key=lambda x: x[0])
            
            # Find the requested frame number
            for file_frame_num, filename in frame_files:
                if file_frame_num == frame_number:
                    return os.path.join(self.frames_output_dir, filename)
            
            print(f"⚠️ Frame {frame_number} not found for scene {scene_id}")
            return None
            
        except Exception as e:
            print(f"❌ Error finding frame file: {e}")
            return None

    def _get_available_scene_images(self) -> dict[str, str]:
        """Get all available scene images."""
        scene_images = {}
        if not os.path.exists(self.scene_images_dir):
            print(f"ERROR: Scene images directory not found: {self.scene_images_dir}")
            return scene_images
        
        for filename in os.listdir(self.scene_images_dir):
            if filename.endswith('.png'):
                scene_id = filename[:-4]  # Remove .png extension
                scene_images[scene_id] = os.path.join(self.scene_images_dir, filename)
        
        return scene_images

    def _find_image_for_scene(self, scene_name: str, scene_id: str) -> Optional[str]:
        """Find image using fixed naming: scene_{id}.png inside scenes folder."""
        filename = f"scene_{scene_id}.png" if scene_id else f"{scene_name}.png"
        full_path = os.path.join(self.scene_images_dir, filename)
        if os.path.exists(full_path):
            return full_path
        return None

    def _get_animation_prompt(self) -> str:
        """Get the master animation prompt."""
        return """ Use the style of {ART_STYLE}.""".format(ART_STYLE=ART_STYLE)

    def _get_negative_prompt(self) -> str:
        """Get the negative prompt for animation."""
        return "worst quality, low quality, blurry, distortion, artifacts, noisy,logo,text, words, letters, writing, caption, subtitle, title, label, watermark, text, extra limbs, extra fingers, bad anatomy, poorly drawn face, asymmetrical features, plastic texture, uncanny valley"

    def _load_base_workflow(self) -> dict:
        """Load the base animation workflow."""
        try:
            with open(self.workflow_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load workflow: {e}")
            return {}

    def _copy_image_to_comfyui(self, scene_id: str, image_path: str) -> str | None:
        """Copy scene image to ComfyUI input directory."""
        try:
            filename = f"{scene_id}.png"
            dest_path = os.path.join(self.comfyui_input_folder, filename)
            shutil.copy2(image_path, dest_path)
            print(f"Copied scene image to ComfyUI input: {dest_path}")
            return filename
        except Exception as e:
            print(f"ERROR: Failed to copy image {image_path}: {e}")
            return None

    def _build_animation_workflow(self, scene_id: str, scene_description: str, image_filename: str, duration: float = None, characters_data: dict[str, str] = None, motion_data: dict[str, str] = None, locations_data: dict[str, str] = None, frame_count: int = None) -> dict:
        """Build animation workflow with scene image and description."""
        workflow = self._load_base_workflow()
        if not workflow:
            return {}

        # Get prompts
        animation_prompt = self._get_animation_prompt()
        negative_prompt = self._get_negative_prompt()
        
        # Extract characters from scene description
        character_names = self._extract_characters_from_scene(scene_description)
        
        
        # Replace location references if location data is available
        processed_scene_description = scene_description
        if locations_data:
            processed_scene_description = self._replace_location_references(scene_description, locations_data)

        # Combine animation prompt with scene description
        full_prompt = f"{animation_prompt}"

        if ENABLE_SCENE:
            full_prompt += f"\n\nSCENE DESCRIPTION:\n{processed_scene_description}"
        
        # Add character details if available
        if character_names and characters_data:
            character_details = self._get_character_details(character_names, characters_data)
            if character_details:
                full_prompt += f"\n\nCHARACTER DESCRIPTION:\n{character_details}"
                print_flush(f"🎭 Added character details for: {', '.join(character_names)}")
        
        # Add motion prompts if available
        if motion_data and ENABLE_MOTION:
            # Try to find matching motion for this scene
            # Extract scene ID (remove "scene_" prefix if present)
            clean_scene_id = self._extract_scene_id(scene_id)
            scene_motion_id = f"motion_{clean_scene_id}"
            if scene_motion_id in motion_data:
                motion_prompt = motion_data[scene_motion_id]
                full_prompt += f"\n\nMOTION PROMPT:\n{motion_prompt}"
                print_flush(f"🎬 Added motion prompt for {clean_scene_id}")
            else:
                print_flush(f"⚠️ No motion data found for scene {clean_scene_id} (looking for {scene_motion_id})")
        
        # Use provided frame count or calculate from duration
        if frame_count is None:
            if duration is not None:
                frame_count = self._calculate_frame_count(duration)
            else:
                print(f"Duration is not provided for {scene_id}")
                frame_count = 121  # Default fallback
        
        # Update workflow nodes
        # Set positive prompt (node 6)
        workflow["6"]["inputs"]["text"] = full_prompt
        
        # Debug: Show final prompt structure
        print_flush(f"📝 Final animation prompt for {scene_id}:")
        print_flush(f"   Scene chars: {len(character_names)} found: {character_names}")
        print_flush(f"   Prompt length: {len(full_prompt)} chars")
        
        # Set negative prompt (node 7)
        workflow["7"]["inputs"]["text"] = negative_prompt
        
        # Set input image (node 1206)
        workflow["1206"]["inputs"]["image"] = image_filename
        
        # Set output filename prefix (node 1336)
        workflow["1336"]["inputs"]["filename_prefix"] = f"{scene_id}_animate"
        
        # Set save image filename prefix (node 2000)
        workflow["2000"]["inputs"]["filename_prefix"] = f"{scene_id}_frame"
        
        # Update video dimensions and frame settings
        if "1241" in workflow:  # LTXVConditioning node
            workflow["1241"]["inputs"]["frame_rate"] = float(FRAMES_PER_SECOND)
        
        if "1336" in workflow:  # Video Combine node
            workflow["1336"]["inputs"]["frame_rate"] = FRAMES_PER_SECOND
        
        if "1338" in workflow:  # LTXVBaseSampler node
            workflow["1338"]["inputs"]["width"] = VIDEO_WIDTH
            workflow["1338"]["inputs"]["height"] = VIDEO_HEIGHT
            workflow["1338"]["inputs"]["num_frames"] = frame_count
        
        return workflow

    def _generate_video(self, scene_id: str, scene_description: str, image_path: str, duration: float = None, characters_data: dict[str, str] = None, motion_data: dict[str, str] = None, locations_data: dict[str, str] = None, resumable_state=None) -> List[str] | None:
        """Generate video(s) from a scene image using ComfyUI with frame continuity between chunks.
        
        Returns:
            List of generated video file paths, or None if failed
        """
        try:
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_video_complete(scene_id):
                cached_result = resumable_state.get_video_result(scene_id)
                if cached_result and cached_result.get('paths'):
                    # Check if all cached video files still exist
                    all_exist = all(os.path.exists(path) for path in cached_result['paths'])
                    if all_exist:
                        print(f"Using cached video: {scene_id}")
                        return cached_result['paths']
                    else:
                        print(f"Cached video files missing, regenerating: {scene_id}")
                elif cached_result:
                    print(f"Cached result missing, regenerating: {scene_id}")
            
            print(f"Generating video for scene: {scene_id}")
            if duration is not None:
                frame_count = self._calculate_frame_count(duration)
                print(f"Duration: {duration:.2f}s, Frames: {frame_count}")
            
            # Calculate video chunks
            chunks = self._calculate_video_chunks(duration) if duration else [("", 97)]
            print(f"Video chunks: {len(chunks)}")
            
            # Copy initial image to ComfyUI input
            image_filename = self._copy_image_to_comfyui(scene_id, image_path)
            if not image_filename:
                return None

            generated_videos = []
            current_input_image = image_filename  # Track the current input image for continuity
            
            for chunk_idx, (chunk_suffix, chunk_frames) in enumerate(chunks):
                chunk_scene_id = f"{scene_id}{chunk_suffix}"
                print(f"Generating chunk: {chunk_scene_id} ({chunk_frames} frames)")
                
                # For subsequent chunks, use the last frame of the previous chunk as input
                if chunk_idx > 0 and generated_videos:
                    # Try to find the last saved frame from the previous chunk
                    previous_chunk_suffix = chunks[chunk_idx - 1][0]  # Get previous chunk suffix
                    last_saved_frame = self._find_last_saved_frame(scene_id, previous_chunk_suffix)
                    
                    if last_saved_frame:
                        # Copy the last saved frame to ComfyUI input folder for processing
                        last_frame_filename = f"{chunk_scene_id}_last_frame.png"
                        comfyui_frame_path = os.path.join(self.comfyui_input_folder, last_frame_filename)
                        shutil.copy2(last_saved_frame, comfyui_frame_path)
                        
                        # Also copy to frames output directory for reference
                        frames_output_path = os.path.join(self.frames_output_dir, last_frame_filename)
                        shutil.copy2(last_saved_frame, frames_output_path)
                        
                        # Update the input image for this chunk
                        current_input_image = last_frame_filename
                        print(f"🔄 Using last saved frame from previous chunk as input for {chunk_scene_id}")
                        print(f"📁 Frame stored: {frames_output_path}")
                    else:
                        print(f"⚠️ Failed to find last saved frame from previous chunk, using original image")
                        current_input_image = image_filename
                
                # Build workflow for this chunk
                workflow = self._build_animation_workflow(chunk_scene_id, scene_description, current_input_image, duration, characters_data, motion_data, locations_data, chunk_frames)
                if not workflow:
                    print(f"ERROR: Failed to build workflow for {chunk_scene_id}")
                    continue

                # Print workflow summary
                self._print_workflow_summary(workflow, f"Animation: {chunk_scene_id}")

                # Submit to ComfyUI
                resp = requests.post(f"{self.comfyui_url}prompt", json={"prompt": workflow}, timeout=300)
                if resp.status_code != 200:
                    print(f"ERROR: ComfyUI API error: {resp.status_code} {resp.text}")
                    continue
                    
                prompt_id = resp.json().get("prompt_id")
                if not prompt_id:
                    print("ERROR: No prompt ID returned from ComfyUI")
                    continue

                # Wait for completion
                print(f"Waiting for video generation completion (prompt_id: {prompt_id})...")
                while True:
                    h = requests.get(f"{self.comfyui_url}history/{prompt_id}")
                    if h.status_code == 200 and prompt_id in h.json():
                        status = h.json()[prompt_id].get("status", {})
                        if status.get("exec_info", {}).get("queue_remaining", 0) == 0:
                            time.sleep(3)  # Give it a moment to finish saving
                            break
                    time.sleep(3)  # Video generation takes longer

                # Copy saved frames to output/frames directory with proper naming
                self._copy_saved_frames_to_output(chunk_scene_id)

                # Find and copy generated video
                generated_video = self._find_newest_output_with_prefix(f"{chunk_scene_id}_animate")
                if not generated_video:
                    print(f"ERROR: Could not find generated video for {chunk_scene_id}")
                    continue

                final_path = os.path.join(self.final_output_dir, f"{chunk_scene_id}.mp4")
                shutil.copy2(generated_video, final_path)
                print(f"Saved video: {final_path}")
                generated_videos.append(final_path)
            
            # Note: Frame files are preserved in output/frames/ for debugging and inspection
            
            # If we have multiple chunks, merge them into a single video
            if len(generated_videos) > 1:
                print(f"🔗 Merging {len(generated_videos)} video chunks for {scene_id}...")
                merged_path = os.path.join(self.final_output_dir, f"{scene_id}.mp4")
                if self._merge_video_chunks(generated_videos, merged_path):
                    # Clean up individual chunk files after successful merge
                    for chunk_path in generated_videos:
                        try:
                            os.remove(chunk_path)
                            print(f"🗑️ Cleaned up chunk: {os.path.basename(chunk_path)}")
                        except OSError as e:
                            print(f"⚠️ Could not remove chunk {os.path.basename(chunk_path)}: {e}")
                    
                    # Save to checkpoint if resumable mode enabled
                    if resumable_state:
                        result = {
                            'paths': [merged_path],
                            'scene_id': scene_id,
                            'scene_description': scene_description,
                            'duration': duration,
                            'frame_count': frame_count
                        }
                        resumable_state.set_video_result(scene_id, result)
                    
                    return [merged_path]  # Return the merged file
                else:
                    print(f"⚠️ Failed to merge chunks for {scene_id}, keeping individual chunks")
                    
                    # Save to checkpoint if resumable mode enabled
                    if resumable_state:
                        result = {
                            'paths': generated_videos,
                            'scene_id': scene_id,
                            'scene_description': scene_description,
                            'duration': duration,
                            'frame_count': frame_count
                        }
                        resumable_state.set_video_result(scene_id, result)
                    
                    return generated_videos  # Return individual chunks if merge failed
            
            # Save to checkpoint if resumable mode enabled (single video case)
            if resumable_state and generated_videos:
                result = {
                    'paths': generated_videos,
                    'scene_id': scene_id,
                    'scene_description': scene_description,
                    'duration': duration,
                    'frame_count': frame_count
                }
                resumable_state.set_video_result(scene_id, result)
            
            return generated_videos if generated_videos else None

        except Exception as e:
            print(f"ERROR: Failed to generate video for {scene_id}: {e}")
            return None

    def _find_newest_output_with_prefix(self, prefix: str) -> str | None:
        """Find the newest generated video with the given prefix."""
        if not os.path.isdir(self.comfyui_output_folder):
            return None
        
        latest, latest_mtime = None, -1.0
        video_exts = {".mp4", ".avi", ".mov", ".webm"}
        for root, _, files in os.walk(self.comfyui_output_folder):
            for name in files:
                if name.startswith(prefix) and any(name.lower().endswith(ext) for ext in video_exts):
                    full_path = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(full_path)
                        if mtime > latest_mtime:
                            latest_mtime, latest = mtime, full_path
                    except OSError:
                        continue
        return latest

    def _get_completed_videos(self) -> set[str]:
        """Get scene IDs that have already been converted to video."""
        if not os.path.exists(self.final_output_dir):
            return set()
        return {f[:-4] for f in os.listdir(self.final_output_dir) if f.endswith('.mp4')}

    def _validate_scene_contiguity(self, durations: List[float], scenes: List[Dict[str, str]]) -> bool:
        """Pre-check: Validate that same scenes appear only in contiguous blocks."""
        if not durations or not scenes or len(durations) != len(scenes):
            return True  # Skip validation if data is invalid
        
        # Track scene occurrences
        scene_occurrences: Dict[str, List[int]] = {}
        
        for i, scene in enumerate(scenes):
            scene_name = scene.get("scene_name", f"scene_{i+1}")
            if scene_name not in scene_occurrences:
                scene_occurrences[scene_name] = []
            scene_occurrences[scene_name].append(i)
        
        # Check contiguity for each scene
        non_contiguous_scenes = []
        
        for scene_name, occurrences in scene_occurrences.items():
            if len(occurrences) > 1:
                # Check if all occurrences are contiguous
                for i in range(1, len(occurrences)):
                    if occurrences[i] != occurrences[i-1] + 1:
                        non_contiguous_scenes.append({
                            "scene": scene_name,
                            "occurrences": occurrences
                        })
                        break
        
        if non_contiguous_scenes:
            print_flush("❌ VALIDATION FAILED: Non-contiguous scene occurrences detected!")
            print_flush("   Each scene must appear in contiguous blocks only.")
            print_flush("   Non-contiguous scenes found:")
            
            for item in non_contiguous_scenes[:5]:  # Show first 5
                scene = item["scene"]
                occurrences = item["occurrences"]
                print_flush(f"   - {scene}: appears at timeline positions {occurrences}")
            
            if len(non_contiguous_scenes) > 5:
                print_flush(f"   ... and {len(non_contiguous_scenes) - 5} more scenes")
            
            print_flush("\n   This prevents proper scene combination and video generation.")
            print_flush("   Please fix the timeline script so identical scenes are grouped together.")
            return False
        
        print_flush("✅ Scene contiguity validation passed")
        return True

    def animate_all_scenes(self, force_regenerate: bool = False, resumable_state=None) -> dict[str, str]:
        """Convert unique scenes to videos using timeline script as single source of truth."""
        print_flush("🎬 Starting scene animation process...")
        print_flush(f"📁 Timeline:  {self.timeline_file}")
        print_flush(f"🖼️  Images:    {self.scene_images_dir}")
        print_flush(f"🎥 Output:    {self.final_output_dir}")
        
        
        # Read timeline data
        durations, scenes = self.read_timeline_data()
        if not durations or not scenes:
            return {}
        
        # Read character data
        characters_data = self._read_character_data()
        if characters_data:
            print_flush(f"📖 Loaded {len(characters_data)} character descriptions")
        else:
            print_flush("⚠️ No character data found - animations will use scene descriptions only")
        
        # Read motion data
        motion_data = self._read_motion_data()
        if motion_data:
            print_flush(f"🎬 Loaded {len(motion_data)} motion prompts")
        else:
            print_flush("⚠️ No motion data found - animations will use basic motion prompts only")
        
        # Read location data
        locations_data = self._read_location_data() if ENABLE_LOCATION else {}
        if ENABLE_LOCATION and locations_data:
            print_flush(f"📍 Loaded {len(locations_data)} location descriptions")
        elif ENABLE_LOCATION:
            print_flush("⚠️ Location info enabled but no location data found")
        
        if len(durations) != len(scenes):
            print_flush(f"⚠️ Count mismatch: timeline entries={len(durations)} vs scenes={len(scenes)}. Proceeding with min count.")
        
        # Combine adjacent same scenes into single large durations
        combined_scenes = self.combine_adjacent_scenes(durations, scenes)
        if not combined_scenes:
            print_flush("❌ Failed to combine scenes or scenes appear non-contiguously.")
            return {}

        # Get available scene images
        scene_images = self._get_available_scene_images()
        
        # Find scenes that have both description and image
        available_scenes = {}
        missing_images = []
        
        for scene_name, scene_info in combined_scenes.items():
            scene_id = scene_info.get("scene_id", scene_name)
            
            # Look for scene image
            image_path = self._find_image_for_scene(scene_name, scene_id)
            if image_path:
                available_scenes[scene_name] = {
                    **scene_info,
                    "image_path": image_path
                }
            else:
                missing_images.append(scene_name)
        
        if missing_images:
            print_flush(f"⚠️ Missing images for {len(missing_images)} scenes:")
            for name in missing_images[:10]:
                print_flush(f"   - {name}")
            if len(missing_images) > 10:
                print_flush(f"   ... and {len(missing_images) - 10} more")
        
        if not available_scenes:
            print_flush("❌ No scenes have both descriptions and images")
            return {}

        # Use resumable state if available, otherwise fall back to file-based checking
        if resumable_state:
            # Run precheck to validate file existence and clean up invalid entries
            cleaned_count = resumable_state.validate_and_cleanup_results()
            if cleaned_count > 0:
                print(f"Precheck completed: {cleaned_count} invalid entries removed from checkpoint")
            
            completed_videos = set()
            for scene_name in available_scenes.keys():
                if resumable_state.is_video_complete(scene_name):
                    completed_videos.add(scene_name)
        else:
            completed_videos = self._get_completed_videos()
        
        if not force_regenerate and completed_videos:
            print_flush(f"Found {len(completed_videos)} completed videos: {sorted(completed_videos)}")

        scenes_to_process = {name: info for name, info in available_scenes.items() 
                           if force_regenerate or name not in completed_videos}

        if not scenes_to_process:
            print_flush("All scenes already converted to video!")
            return {}

        print_flush(f"Processing {len(scenes_to_process)} unique scenes, skipped {len(completed_videos)}")
        print_flush("=" * 60)

        results = {}
        # Initialize start time for time estimation
        self.start_time = time.time()
        
        print_flush(f"\n📊 VIDEO ANIMATION PROGRESS")
        print_flush("=" * 120)
        print_flush(f"{'Scene':<6} {'Description':<50} {'Duration':<10} {'Status':<15} {'Time':<10} {'ETA':<10}")
        print_flush("-" * 120)
        
        for i, (scene_name, scene_info) in enumerate(scenes_to_process.items(), 1):
            scene_start_time = time.time()
            duration = scene_info["total_duration"]
            description = scene_info["description"]
            image_path = scene_info["image_path"]
            frame_count = self._calculate_frame_count(duration)
            
            eta = self.estimate_remaining_time(i, len(scenes_to_process), scene_description=description, duration=duration)
            print_flush(f"{i:<6} {description[:50]:<50} {duration:<10.2f} {'PROCESSING':<15} {'--':<10} {eta:<10}")
            
            output_paths = self._generate_video(scene_name, description, image_path, duration, characters_data, motion_data, locations_data, resumable_state)
            
            scene_processing_time = time.time() - scene_start_time
            self.processing_times.append(scene_processing_time)
            
            if output_paths:
                eta = self.estimate_remaining_time(i, len(scenes_to_process), scene_processing_time, description, duration)
                if len(output_paths) == 1:
                    results[scene_name] = output_paths[0]
                    print_flush(f"{i:<6} {description[:50]:<50} {duration:<10.2f} {'✅ COMPLETED':<15} {self.format_processing_time(scene_processing_time):<10} {eta:<10}")
                    print_flush(f"✅ Generated video: {scene_name}")
                else:
                    # Multiple chunks generated
                    results[scene_name] = output_paths
                    print_flush(f"{i:<6} {description[:50]:<50} {duration:<10.2f} {'✅ COMPLETED':<15} {self.format_processing_time(scene_processing_time):<10} {eta:<10}")
                    print_flush(f"✅ Generated {len(output_paths)} video chunks: {scene_name}")
                    for j, path in enumerate(output_paths, 1):
                        print_flush(f"   Chunk {j}: {os.path.basename(path)}")
            else:
                eta = self.estimate_remaining_time(i, len(scenes_to_process), scene_processing_time, description, duration)
                print_flush(f"{i:<6} {description[:50]:<50} {duration:<10.2f} {'❌ FAILED':<15} {self.format_processing_time(scene_processing_time):<10} {eta:<10}")
                print_flush(f"❌ Failed: {scene_name}")
        
        elapsed = time.time() - self.start_time
        
        print_flush("\n📊 ANIMATION SUMMARY:")
        print_flush(f"   • Total timeline entries: {len(durations)}")
        print_flush(f"   • Unique scenes:         {len(combined_scenes)}")
        print_flush(f"   • Scenes with images:    {len(available_scenes)}")
        # Count total videos (including chunks)
        total_videos = sum(len(v) if isinstance(v, list) else 1 for v in results.values())
        print_flush(f"   • Videos created:        {total_videos} ({len(results)} scenes)")
        print_flush(f"   • Skipped existing:      {len(completed_videos)}")
        print_flush(f"   • Missing images:        {len(missing_images)}")
        print_flush(f"   • Time:                  {elapsed:.2f}s")

        return results

    def _print_workflow_summary(self, workflow: dict, title: str) -> None:
        """Print a comprehensive workflow summary showing the flow to sampler inputs."""
        print(f"\n🔗 WORKFLOW SUMMARY: {title}")
        
        # Find the main sampler node (LTXVBaseSampler)
        sampler_node = None
        sampler_id = None
        for node_id, node in workflow.items():
            if node.get("class_type") == "LTXVBaseSampler":
                sampler_node = node
                sampler_id = node_id
                break
        
        if sampler_node:
            inputs = sampler_node.get("inputs", {})
            print(f"   📊 LTXVBaseSampler({sampler_id}) - Core Parameters:")
            print(f"      📐 Dimensions: {inputs.get('width', 'N/A')}x{inputs.get('height', 'N/A')}")
            print(f"      🎬 Frames: {inputs.get('num_frames', 'N/A')}")
            print(f"      💪 Strength: {inputs.get('strength', 'N/A')}")
            print(f"      🎯 Crop: {inputs.get('crop', 'N/A')}")
            print(f"      📊 CRF: {inputs.get('crf', 'N/A')}")
            print(f"      🔄 Blur: {inputs.get('blur', 'N/A')}")
            
            # Trace input flows
            self._trace_input_flow(workflow, "model", inputs.get("model", [None, 0])[0], inputs.get("model", [None, 0])[1], sampler_id)
            self._trace_input_flow(workflow, "vae", inputs.get("vae", [None, 0])[0], inputs.get("vae", [None, 0])[1], sampler_id)
            self._trace_input_flow(workflow, "guider", inputs.get("guider", [None, 0])[0], inputs.get("guider", [None, 0])[1], sampler_id)
            self._trace_input_flow(workflow, "sampler", inputs.get("sampler", [None, 0])[0], inputs.get("sampler", [None, 0])[1], sampler_id)
            self._trace_input_flow(workflow, "sigmas", inputs.get("sigmas", [None, 0])[0], inputs.get("sigmas", [None, 0])[1], sampler_id)
            self._trace_input_flow(workflow, "noise", inputs.get("noise", [None, 0])[0], inputs.get("noise", [None, 0])[1], sampler_id)
            self._trace_input_flow(workflow, "optional_cond_images", inputs.get("optional_cond_images", [None, 0])[0], inputs.get("optional_cond_images", [None, 0])[1], sampler_id)
        
        print("   " + "="*50)
    
    def _trace_input_flow(self, workflow: dict, input_name: str, source_node_id: str, source_output: int, sampler_id: str) -> None:
        """Dynamically trace the flow from source to sampler input using backward tracing."""
        if source_node_id not in workflow:
            print(f"   ❌ {input_name}: Source node {source_node_id} not found")
            return
            
        print(f"\n   🔗 {input_name.upper()} FLOW:")
        # Use backward tracing to build the complete path
        path_data = []
        self._trace_node_backwards_with_storage(workflow, source_node_id, sampler_id, 0, path_data, input_name)
        # Print the path in reverse order (source to target)
        self._print_reverse_path(workflow, path_data, sampler_id)
    
    def _trace_node_backwards_with_storage(self, workflow: dict, node_id: str, target_id: str, depth: int, path_data: list, specific_input: str = None) -> None:
        """Recursively trace backwards through the workflow graph and store path data."""
        if node_id not in workflow:
            return
            
        node = workflow[node_id]
        node_type = node.get("class_type", "Unknown")
        node_inputs = node.get("inputs", {})
        
        # Store current node data
        node_data = {
            "node_id": node_id,
            "node_type": node_type,
            "node_inputs": node_inputs,
            "depth": depth
        }
        path_data.append(node_data)
        
        # Continue tracing backwards for specific input or all inputs
        if specific_input and specific_input in node_inputs:
            # Trace only the specific input
            input_value = node_inputs[specific_input]
            if isinstance(input_value, list) and len(input_value) >= 2:
                upstream_node_id = input_value[0]
                if upstream_node_id in workflow and upstream_node_id != node_id:  # Avoid infinite loops
                    self._trace_node_backwards_with_storage(workflow, upstream_node_id, target_id, depth + 1, path_data)
        else:
            # Trace all inputs (original behavior)
            for input_name, input_value in node_inputs.items():
                if isinstance(input_value, list) and len(input_value) >= 2:
                    upstream_node_id = input_value[0]
                    if upstream_node_id in workflow and upstream_node_id != node_id:  # Avoid infinite loops
                        self._trace_node_backwards_with_storage(workflow, upstream_node_id, target_id, depth + 1, path_data)
    
    def _print_reverse_path(self, workflow: dict, path_data: list, sampler_id: str) -> None:
        """Print the stored path data in reverse order (source to target)."""
        if not path_data:
            print("      ❌ No path found")
            return
        
        # Reverse the path data to show source → target
        reversed_path = list(reversed(path_data))
        
        for i, node_data in enumerate(reversed_path):
            node_id = node_data["node_id"]
            node_type = node_data["node_type"]
            node_inputs = node_data["node_inputs"]
            depth = node_data["depth"]
            
            # Indent based on position in reversed path
            indent = "      " + "   " * i
            
            if i == 0:
                # First node (source)
                print(f"{indent}📤 {node_type}({node_id})")
            elif i == len(reversed_path) - 1:
                # Last node (target/sampler)
                print(f"{indent}📥 {node_type}({node_id})")
            else:
                # Middle nodes
                print(f"{indent}⬇️  {node_type}({node_id})")
            
            # Show node parameters
            self._show_node_parameters(node_type, node_inputs, indent + "   ")
    
    def _show_node_parameters(self, node_type: str, node_inputs: dict, indent: str) -> None:
        """Show relevant parameters for a node type."""
        if node_type == "UnetLoaderGGUF":
            print(f"{indent}🤖 Model: {node_inputs.get('unet_name', 'N/A')}")
            print(f"{indent}📱 Device: {node_inputs.get('device', 'cuda')}")
            
        elif node_type == "LoraLoader":
            print(f"{indent}🎨 LoRA: {node_inputs.get('lora_name', 'N/A')}")
            print(f"{indent}💪 Model Strength: {node_inputs.get('strength_model', 'N/A')}")
            print(f"{indent}📝 CLIP Strength: {node_inputs.get('strength_clip', 'N/A')}")
            
        elif node_type == "CLIPTextEncode":
            text = node_inputs.get("text", "")
            if len(text) > 80:
                text = text[:80] + "..."
            print(f"{indent}📝 Text: {text}")
            
        elif node_type == "LoadImage":
            print(f"{indent}🖼️ Image: {node_inputs.get('image', 'N/A')}")
            
        elif node_type == "LTXVConditioning":
            print(f"{indent}🎬 Frame Rate: {node_inputs.get('frame_rate', 'N/A')}")
            
        elif node_type == "STGGuiderAdvanced":
            print(f"{indent}🎯 CFG Threshold: {node_inputs.get('skip_steps_sigma_threshold', 'N/A')}")
            print(f"{indent}🔄 CFG Rescale: {node_inputs.get('cfg_star_rescale', 'N/A')}")
            
        elif node_type == "RandomNoise":
            print(f"{indent}🎲 Noise Seed: {node_inputs.get('noise_seed', 'N/A')}")
            
        elif node_type == "StringToFloatList":
            print(f"{indent}📝 String: {node_inputs.get('string', 'N/A')}")
            
        elif node_type == "FloatToSigmas":
            print(f"{indent}📊 Float List: Connected")
            
        elif node_type == "Set VAE Decoder Noise":
            print(f"{indent}⏰ Timestep: {node_inputs.get('timestep', 'N/A')}")
            print(f"{indent}📏 Scale: {node_inputs.get('scale', 'N/A')}")
            print(f"{indent}🌱 Seed: {node_inputs.get('seed', 'N/A')}")
            
        elif node_type == "KSamplerSelect":
            print(f"{indent}🎲 Sampler: {node_inputs.get('sampler_name', 'N/A')}")
            
        elif node_type == "VHS_VideoCombine":
            print(f"{indent}🎬 Frame Rate: {node_inputs.get('frame_rate', 'N/A')}")
            print(f"{indent}🔄 Loop Count: {node_inputs.get('loop_count', 'N/A')}")
            print(f"{indent}💾 Prefix: {node_inputs.get('filename_prefix', 'N/A')}")
            print(f"{indent}🎥 Format: {node_inputs.get('format', 'N/A')}")
            print(f"{indent}📹 Pixel Format: {node_inputs.get('pix_fmt', 'N/A')}")
            print(f"{indent}📊 CRF: {node_inputs.get('crf', 'N/A')}")
            
        elif node_type == "SaveImage":
            print(f"{indent}💾 Filename: {node_inputs.get('filename_prefix', 'N/A')}")
            
        elif node_type == "VAELoader":
            print(f"{indent}🔄 VAE: {node_inputs.get('vae_name', 'N/A')}")
            
        elif node_type == "CLIPLoader":
            print(f"{indent}📖 Type: CLIPLoader")
            print(f"{indent}📁 Clip: {node_inputs.get('clip_name', 'N/A')}")
            print(f"{indent}⚙️ type: {node_inputs.get('type', 'N/A')}")
            print(f"{indent}⚙️ device: {node_inputs.get('device', 'N/A')}")
            
        # Show any other relevant parameters
        for key, value in node_inputs.items():
            if key not in ['model', 'clip', 'vae', 'pixels', 'samples', 'image', 'text', 'lora_name', 'strength_model', 'strength_clip', 'model_name', 'device', 'width', 'height', 'num_frames', 'strength', 'crop', 'crf', 'blur', 'frame_rate', 'skip_steps_sigma_threshold', 'cfg_star_rescale', 'noise_seed', 'string', 'timestep', 'scale', 'sampler_name', 'loop_count', 'filename_prefix', 'format', 'pix_fmt', 'vae_name', 'clip_name', 'type']:
                if isinstance(value, (str, int, float, bool)) and len(str(value)) < 50:
                    print(f"{indent}⚙️ {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert scene images to animated videos using ComfyUI and timeline script")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration of all videos")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")
    args = parser.parse_args()
    
    animator = VideoAnimator()
    
    # Initialize resumable state if enabled
    resumable_state = None
    if ENABLE_RESUMABLE_MODE:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.normpath(os.path.join(base_dir, "../output/tracking"))
        script_name = Path(__file__).stem  # Automatically get script name without .py extension
        resumable_state = ResumableState(checkpoint_dir, script_name, args.force_start)
        print_flush(f"Resumable mode enabled - checkpoint directory: {checkpoint_dir}")
        if resumable_state.state_file.exists():
            print_flush(f"Found existing checkpoint: {resumable_state.state_file}")
            print_flush(resumable_state.get_progress_summary())
        else:
            print_flush("No existing checkpoint found - starting fresh")
    
    # Process all unique scenes from timeline
    results = animator.animate_all_scenes(force_regenerate=args.force, resumable_state=resumable_state)
    
    if results:
        total_videos = sum(len(v) if isinstance(v, list) else 1 for v in results.values())
        print_flush(f"\n✅ Successfully generated {total_videos} animated videos from {len(results)} scenes:")
        for scene_name, paths in results.items():
            if isinstance(paths, list):
                print_flush(f"  {scene_name}: {len(paths)} chunks")
                for i, path in enumerate(paths, 1):
                    print_flush(f"    Chunk {i}: {path}")
            else:
                print_flush(f"  {scene_name}: {paths}")
        
        # Clean up checkpoint files if resumable mode was used and everything completed successfully
        if resumable_state:
            print_flush("All operations completed successfully")
            print_flush("Final progress:", resumable_state.get_progress_summary())
            resumable_state.cleanup()
        
        return 0
    else:
        print_flush("\n⚠️ No new videos generated")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
