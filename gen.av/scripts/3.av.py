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
WORKFLOW_SUMMARY_ENABLED = False  # Set to True to enable workflow summary printing

# Video configuration constants
VIDEO_WIDTH = 1024
VIDEO_HEIGHT = 576
FRAMES_PER_SECOND = 24
CHUNK_SIZE = 5  # Maximum seconds per chunk (5 seconds max for AV)

# Feature flags
ENABLE_SCENE = False  # Set to True to add scene prompts from 3.scene.txt
ENABLE_LOCATION_IN_SCENE = False  # Set to True to add location prompts from 3.location.txt
ENABLE_CHARACTER_IN_MOTION = False  # Set to True to add character prompts from 2.character.txt in motion prompts

USE_SUMMARY_TEXT = False  # Set to True to use summary text

# Words to speech ratio: seconds per word
WORDS_TO_SPEECH_RATIO = 0.15  # 0.15 seconds per word (approximately 6.67 words per second)

# LoRA Switch Configuration (controls LoRA chain in movie.json workflow)
# Each switch controls a specific part of the LoRA chain
ENABLE_SWITCH_279_286 = True   # Node 279:286 - Controls depth-control LoRA (ltx-2-19b-ic-lora-depth-control.safetensors)
ENABLE_SWITCH_279_288 = True   # Node 279:288 - Controls canny-control LoRA (ltx-2-19b-ic-lora-canny-control.safetensors)
ENABLE_SWITCH_279_289 = True   # Node 279:289 - Controls pose-control LoRA (ltx-2-19b-ic-lora-pose-control.safetensors)
ENABLE_SWITCH_279_290 = True   # Node 279:290 - Controls detailer LoRA (ltx-2-19b-ic-lora-detailer.safetensors)
ENABLE_SWITCH_279_291 = True   # Node 279:291 - Final switch for first CFGGuider (279:239)
ENABLE_SWITCH_279_292 = True   # Node 279:292 - Final switch for second CFGGuider (279:252)

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
            },
            "chunks": {
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
    
    def is_chunk_complete(self, chunk_id: str) -> bool:
        """Check if a chunk is complete."""
        return chunk_id in self.state.get("chunks", {}).get("completed", [])
    
    def get_chunk_result(self, chunk_id: str) -> dict:
        """Get chunk result."""
        return self.state.get("chunks", {}).get("results", {}).get(chunk_id, {})
    
    def set_chunk_result(self, chunk_id: str, result: dict):
        """Set chunk result and mark as complete."""
        if "chunks" not in self.state:
            self.state["chunks"] = {"completed": [], "results": {}}
        self.state["chunks"]["results"][chunk_id] = result
        if chunk_id not in self.state["chunks"]["completed"]:
            self.state["chunks"]["completed"].append(chunk_id)
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
        """Validate that all completed video/chunk files actually exist and clean up missing entries.
        
        Returns:
            int: Number of entries cleaned up (removed from completed list)
        """
        cleaned_count = 0
        videos_to_remove = []
        chunks_to_remove = []
        
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
        
        # Check each completed chunk
        chunks_state = self.state.get("chunks", {})
        for chunk_id in chunks_state.get("completed", []):
            result = chunks_state.get("results", {}).get(chunk_id, {})
            file_path = result.get('path')
            
            # Check if chunk file actually exists
            if not file_path or not os.path.exists(file_path):
                print(f"Precheck: Chunk file missing for {chunk_id} - marking as not completed")
                chunks_to_remove.append(chunk_id)
                cleaned_count += 1
        
        # Remove invalid video entries
        for scene_name in videos_to_remove:
            if scene_name in self.state["videos"]["completed"]:
                self.state["videos"]["completed"].remove(scene_name)
            if scene_name in self.state["videos"]["results"]:
                del self.state["videos"]["results"][scene_name]
        
        # Remove invalid chunk entries
        if "chunks" in self.state:
            for chunk_id in chunks_to_remove:
                if chunk_id in self.state["chunks"]["completed"]:
                    self.state["chunks"]["completed"].remove(chunk_id)
                if chunk_id in self.state["chunks"]["results"]:
                    del self.state["chunks"]["results"][chunk_id]
        
        # Save cleaned state if any changes were made
        if cleaned_count > 0:
            self._save_state()
            print(f"Precheck: Cleaned up {cleaned_count} invalid entries from checkpoint")
        
        return cleaned_count
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        completed = len(self.state["videos"]["completed"])
        total = len(self.state["videos"]["results"]) + len([k for k in self.state["videos"]["results"].keys() if k not in self.state["videos"]["completed"]])
        
        chunks_completed = len(self.state.get("chunks", {}).get("completed", []))
        chunks_total = len(self.state.get("chunks", {}).get("results", {}))
        
        return f"Progress: Videos({completed}/{total}), Chunks({chunks_completed}/{chunks_total})"


def print_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

class AVVideoGenerator:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188/"):
        self.comfyui_url = comfyui_url
        # ComfyUI saves videos under this folder
        self.comfyui_output_folder = "../../ComfyUI/output"
        # ComfyUI input folder where we need to copy scene images
        self.comfyui_input_folder = "../../ComfyUI/input"
        # Scene images directory
        self.scene_images_dir = "../../gen.image/output/scene"
        # Final destination for videos
        self.final_output_dir = "../output"
        # Directory for storing extracted frames
        self.frames_output_dir = "../output/frames"
        # Story input file (instead of timeline)
        self.story_file = "../input/1.story.txt"
        # Movie workflow file
        self.workflow_file = "../workflow/movie.json"
        # Character data file
        self.character_file = "../../gen.image/input/3.character.txt" if USE_SUMMARY_TEXT else "../../gen.image/input/2.character.txt"
        # Location data file
        self.location_file = "../../gen.image/input/3.location.txt" if USE_SUMMARY_TEXT else "../../gen.image/input/2.location.txt"
        # Motion data file
        self.motion_file = "../input/2.motion.txt"

        # Time estimation tracking
        self.processing_times = []
        self.start_time = None

        # Create output directories
        os.makedirs(self.final_output_dir, exist_ok=True)
        os.makedirs(self.frames_output_dir, exist_ok=True)
        os.makedirs(self.comfyui_input_folder, exist_ok=True)

    def estimate_remaining_time(self, current_scene: int, total_scenes: int, scene_processing_time: float = None, scene_description: str = None, duration: float = None) -> str:
        """Estimate remaining time based on duration and words - hybrid approach for video generation"""
        
        # Calculate total video duration and words if we have scene data
        total_video_duration = getattr(self, 'total_video_duration', 0)
        total_video_words = getattr(self, 'total_video_words', 0)
        
        # For first scene with no data, provide a reasonable initial estimate
        if not self.processing_times and scene_processing_time is None:
            if total_video_duration > 0 and total_video_words > 0:
                # Initial estimate: assume 30x realtime for video generation (30 seconds of video takes 900 seconds to generate)
                duration_estimate = total_video_duration * 30
                # Also consider text complexity: assume 50 words per minute for video processing
                words_estimate = (total_video_words / 50) * 60
                # Use the higher of the two estimates
                estimated_remaining_seconds = max(duration_estimate, words_estimate)
            elif total_video_duration > 0:
                # Duration-based estimate only
                estimated_remaining_seconds = total_video_duration * 30
            elif total_video_words > 0:
                # Words-based estimate only
                estimated_remaining_seconds = (total_video_words / 50) * 60
            else:
                # Fallback: assume 300 seconds per scene (5 minutes for video generation)
                remaining_scenes = total_scenes - current_scene
                estimated_remaining_seconds = remaining_scenes * 300.0
            
            return self._format_time_with_confidence(estimated_remaining_seconds, confidence="low")
        
        # Calculate actual generation speed from completed scenes
        if scene_processing_time and scene_description and duration is not None:
            # Store video processing data for better estimation
            if not hasattr(self, 'video_processing_data'):
                self.video_processing_data = []
            self.video_processing_data.append({
                'duration': duration,
                'words': len(scene_description.split()),
                'processing_time': scene_processing_time,
                'duration_ratio': scene_processing_time / duration if duration > 0 else 1.0,
                'words_per_minute': (len(scene_description.split()) / scene_processing_time * 60) if scene_processing_time > 0 else 50
            })
            
            # Use hybrid estimation if we have video processing data
            if hasattr(self, 'video_processing_data') and self.video_processing_data:
                # Calculate average duration ratio (processing_time / video_duration)
                total_duration_processed = sum(data['duration'] for data in self.video_processing_data)
                total_processing_time = sum(data['processing_time'] for data in self.video_processing_data)
                
                if total_duration_processed > 0:
                    actual_duration_ratio = total_processing_time / total_duration_processed
                    
                    # Estimate remaining video duration
                    duration_processed_so_far = sum(data['duration'] for data in self.video_processing_data)
                    remaining_duration = total_video_duration - duration_processed_so_far
                    
                    # Calculate remaining time based on actual duration ratio
                    duration_based_estimate = remaining_duration * actual_duration_ratio
                    
                    # Also calculate words-based estimate
                    total_words_processed = sum(data['words'] for data in self.video_processing_data)
                    if total_processing_time > 0:
                        actual_wpm = total_words_processed / (total_processing_time / 60)
                        words_processed_so_far = sum(data['words'] for data in self.video_processing_data)
                        remaining_words = total_video_words - words_processed_so_far
                        words_based_estimate = (remaining_words / actual_wpm) * 60 if actual_wpm > 0 else duration_based_estimate
                        
                        # Use the higher of the two estimates
                        estimated_remaining_seconds = max(duration_based_estimate, words_based_estimate)
                    else:
                        estimated_remaining_seconds = duration_based_estimate
                    
                    # Determine confidence based on data points
                    confidence = "low"
                    if len(self.video_processing_data) >= 5:
                        confidence = "high"
                    elif len(self.video_processing_data) >= 3:
                        confidence = "medium"
                    
                    return self._format_time_with_confidence(estimated_remaining_seconds, confidence)
        
        # Fallback to scene-based estimation if no video data available
        if scene_processing_time:
            all_times = self.processing_times + [scene_processing_time]
        else:
            all_times = self.processing_times
        
        # Simple average of recent processing times
        estimated_time_per_scene = sum(all_times) / len(all_times)
        remaining_scenes = total_scenes - current_scene
        estimated_remaining_seconds = remaining_scenes * estimated_time_per_scene
        
        # Determine confidence level
        confidence = "low"
        if len(all_times) >= 5:
            confidence = "high"
        elif len(all_times) >= 3:
            confidence = "medium"
        
        return self._format_time_with_confidence(estimated_remaining_seconds, confidence)
    
    def _format_time_with_confidence(self, seconds: float, confidence: str = "medium") -> str:
        """Format time with confidence indicator"""
        # Format the time part
        if seconds < 60:
            time_str = f"~{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            time_str = f"~{minutes:.1f}m"
        else:
            hours = seconds / 3600
            time_str = f"~{hours:.1f}h"
        
        # Add confidence indicator
        if confidence == "high":
            return f"{time_str} (✓)"
        elif confidence == "medium":
            return f"{time_str} (~)"
        else:  # low
            return f"{time_str} (?)"
    
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

    def read_story_data(self) -> Tuple[List[float], List[Dict[str, str]]]:
        """Read and parse story file to extract scenes with dialogue and calculate durations.
        
        Returns:
            Tuple of (durations, scenes) where scenes contain scene_name, scene_id, description, dialogue
        """
        durations: List[float] = []
        scenes: List[Dict[str, str]] = []
        
        try:
            with open(self.story_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print_flush(f"❌ Story file not found: {self.story_file}")
            return [], []
        except Exception as e:
            print_flush(f"❌ Error reading story file: {e}")
            return [], []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for dialogue lines (format: [speaker] dialogue text...)
            if line.startswith('[') and ']' in line:
                # Extract dialogue
                dialogue_end = line.find(']')
                speaker = line[1:dialogue_end]
                dialogue_text = line[dialogue_end + 1:].strip()
                
                # Count words in dialogue
                word_count = len(dialogue_text.split())
                duration = word_count * WORDS_TO_SPEECH_RATIO
                
                # Look for the next scene line
                j = i + 1
                scene_id = None
                scene_description = None
                
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line.startswith('(scene_'):
                        # Extract scene ID and description
                        scene_match = re.match(r'\(scene_(\d+\.\d+)\)\s*(.+)', next_line)
                        if scene_match:
                            scene_id = scene_match.group(1)
                            scene_description = scene_match.group(2).strip()
                        break
                    j += 1
                
                if scene_id and scene_description:
                    scene_name = f"scene_{scene_id}"
                    # Build description from dialogue + scene description
                    full_description = f"{dialogue_text}\n{scene_description}"
                    
                    durations.append(duration)
                    scenes.append({
                        "scene_name": scene_name,
                        "scene_id": scene_id,
                        "description": full_description,
                        "dialogue": dialogue_text,
                        "speaker": speaker
                    })
                
                i = j + 1 if scene_id else i + 1
            else:
                i += 1
        
        print_flush(f"📋 Loaded {len(durations)} scenes with dialogue from {self.story_file}")
        return durations, scenes

    def _read_character_data(self) -> dict[str, str]:
        """Parse character data from input file.
        
        Format: Each character on a single line as: ((character_name)): description
        """
        characters = {}
        try:
            with open(self.character_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            # Split on single newlines since each character is on one line
            entries = [entry.strip() for entry in content.split('\n') if entry.strip()]
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
        """Parse location data from input file.
        
        Format: Each location on a single line as: {{loc_id}} description
        """
        locations = {}
        try:
            with open(self.location_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            # Split on single newlines since each location is on one line
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            for line in lines:
                # Match format: {{loc_id}} description...
                match = re.match(r'\{\{([^}]+)\}\}\s*(.+)', line)
                if match:
                    locations[match.group(1).strip()] = match.group(2).strip()
        except Exception as e:
            print_flush(f"ERROR: Failed to read location data: {e}")
        return locations

    def _replace_location_references(self, scene_description: str, locations_data: dict[str, str]) -> str:
        """Replace {{loc_id}} references with location descriptions inline.
        
        Format: {{loc_forest}} -> scene which should look like, {detailed description}, as background
        """
        if not ENABLE_LOCATION_IN_SCENE or not locations_data:
            return scene_description
        
        def replace_func(match):
            full_match = match.group(0)
            # Extract loc_id (handle both {{loc_id}} and {{loc_id, description}} formats)
            content = match.group(1).strip()
            loc_id = content.split(',')[0].strip() if ',' in content else content
            
            if loc_id in locations_data:
                # Replace with inline location description
                return f"\nSCENE, {locations_data[loc_id]}, as background of the entire illustration."
            else:
                # Location not found, keep original
                return full_match

        # Replace {{loc_id}} patterns with location descriptions
        result = re.sub(r'\{\{([^}]+)\}\}', replace_func, scene_description)
        return result

    def _replace_motion_character_references(self, motion_prompt: str, characters_data: dict[str, str]) -> str:
        """Replace ((character_name)) references in motion prompts with character descriptions.
        
        Format: ((Alice)) {walks} -> Character which should look like, {description}, as one of the main characters {walks}
        """
        if not ENABLE_CHARACTER_IN_MOTION or not characters_data:
            return motion_prompt
        
        def replace_motion_char_func(match):
            full_match = match.group(0)
            char_name = match.group(1).strip()
            
            if char_name in characters_data:
                # Replace with inline character description for motion
                return f"\nCHARACTER, {char_name}, which looks like, {characters_data[char_name]},"
            else:
                # Character not found, keep original
                return full_match
        
        # Replace ((character_name)) patterns in motion prompts
        result = re.sub(r'\(\(([^)]+)\)\)', replace_motion_char_func, motion_prompt)
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
        max_frames_per_chunk = CHUNK_SIZE * FRAMES_PER_SECOND + 1
        
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
        """Find the last saved frame from ComfyUI output for a specific scene and chunk."""
        try:
            frame_prefix = f"{scene_id}{chunk_suffix}.frame_"
            print(f"🔍 Frame prefix: {frame_prefix}")
            
            if not os.path.isdir(self.frames_output_dir):
                print(f"❌ Frames directory not found: {self.frames_output_dir}")
                return None
            
            # Find all matching frames and extract frame numbers
            matching_frames = []
            for root, _, files in os.walk(self.frames_output_dir):
                for name in files:
                    if name.startswith(frame_prefix) and name.endswith('.png'):
                        frame_match = re.search(r'.frame_(\d+)\.png$', name)
                        if frame_match:
                            frame_number = int(frame_match.group(1))
                            full_path = os.path.join(root, name)
                            matching_frames.append((full_path, frame_number))
            
            if not matching_frames:
                return None

            print(f"🔍 Matching frames: {len(matching_frames)}")
            
            # Get the frame with highest number
            latest_frame = max(matching_frames, key=lambda x: x[1])
            print_flush(f"🔍 Found last frame: {os.path.basename(latest_frame[0])} (frame #{latest_frame[1]})")
            return latest_frame[0]
            
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
                print_flush(f"⚠️ Could not get video duration, using fallback method...")
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
                
                print_flush(f"📹 Video duration: {duration:.3f}s, extracting frame at {last_frame_time:.3f}s")
                
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
                print_flush(f"✅ Extracted last frame: {os.path.basename(output_image_path)}")
                return True
            else:
                print_flush(f"❌ Failed to extract last frame: {result.stderr}")
                return False
                
        except Exception as e:
            print_flush(f"❌ Error extracting last frame: {e}")
            return False

    def _find_image_for_scene(self, scene_name: str, scene_id: str) -> Optional[str]:
        """Find image using fixed naming: scene_{id}.png inside scenes folder."""
        filename = f"scene_{scene_id}.png" if scene_id else f"{scene_name}.png"
        full_path = os.path.join(self.scene_images_dir, filename)
        if os.path.exists(full_path):
            return full_path
        return None

    def _get_negative_prompt(self) -> str:
        """Get the negative prompt for animation."""
        return "worst quality, low quality, blurry, distortion, artifacts, noisy,logo,text, words, letters, writing, caption, subtitle, title, label, watermark, text, extra limbs, extra fingers, bad anatomy, poorly drawn face, asymmetrical features, plastic texture, uncanny valley"

    def _get_positive_prompt(self) -> str:
        """Get the positive prompt for animation."""
        return "Shot taken with Fixed Camera Position, Camera Angle and Camera Focus."

    def _load_base_workflow(self) -> dict:
        """Load the base movie workflow."""
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

    def _build_av_workflow(self, chunk_scene_id: str, scene_description: str, image_filename: str, duration: float = None, characters_data: dict[str, str] = None, motion_data: dict[str, str] = None, locations_data: dict[str, str] = None, frame_count: int = None) -> dict:
        """Build AV workflow with scene image and description.
        
        Args:
            chunk_scene_id: The chunk scene ID (e.g., "scene_1.1" or "scene_1.1_2" for chunk 2)
            scene_description: Visual description of the scene (includes dialogue)
            image_filename: Input image filename in ComfyUI input folder
            duration: Duration of video in seconds
            characters_data: Character descriptions for inline replacement
            motion_data: Motion prompts for scenes
            locations_data: Location descriptions for inline replacement
            frame_count: Number of frames to generate (overrides duration)
        """
        workflow = self._load_base_workflow()
        if not workflow:
            return {}

        # Get prompts - build from scene description (which includes dialogue)
        negative_prompt = self._get_negative_prompt()
        positive_prompt = self._get_positive_prompt() + "\n" + (scene_description if ENABLE_SCENE else scene_description)

        if locations_data:
            positive_prompt = self._replace_location_references(positive_prompt, locations_data)
            print_flush(f"📍 Replaced location references with descriptions")
        
        # Add motion prompts if available (with inline character replacements)
        if motion_data:
            # Try to find matching motion for this scene
            clean_scene_id = self._extract_scene_id(chunk_scene_id)
            # Remove only the LAST underscore+number (chunk suffix) if it exists
            if '_' in clean_scene_id and clean_scene_id.split('_')[-1].isdigit():
                clean_scene_id = clean_scene_id.rsplit('_', 1)[0]
            scene_motion_id = f"motion_{clean_scene_id}"
            if scene_motion_id in motion_data:
                motion_prompt = motion_data[scene_motion_id]
                # Replace character references in motion prompt with descriptions
                if characters_data:
                    motion_prompt = self._replace_motion_character_references(motion_prompt, characters_data)
                    print_flush(f"🎬 Added motion prompt with inline character replacements for {clean_scene_id}")
                else:
                    print_flush(f"🎬 Added motion prompt for {clean_scene_id}")
                positive_prompt += f"\n{motion_prompt}"
            else:
                print_flush(f"⚠️ No motion data found for scene {clean_scene_id} (looking for {scene_motion_id})")
        
        # Use provided frame count or calculate from duration
        if frame_count is None:
            if duration is not None:
                frame_count = self._calculate_frame_count(duration)
            else:
                print(f"Duration is not provided for {chunk_scene_id}")
                frame_count = 121  # Default fallback

        print_flush(f"Full prompt: {positive_prompt}")
        
        # Update workflow nodes for movie.json
        # Node "204" is the text input (PrimitiveStringMultiline) - set to positive prompt
        if "204" in workflow:
            workflow["204"]["inputs"]["value"] = positive_prompt
        
        # Node "228" is the image input (LoadImage)
        if "228" in workflow:
            workflow["228"]["inputs"]["image"] = image_filename
        
        # Node "279:277" is the length (PrimitiveInt) - frame count
        if "279:277" in workflow:
            workflow["279:277"]["inputs"]["value"] = frame_count
        
        # Node "279:270" is the frame rate (PrimitiveFloat) - keep at 24
        if "279:270" in workflow:
            workflow["279:270"]["inputs"]["value"] = float(FRAMES_PER_SECOND)
        
        # Node "104" is the video output (SaveVideo) - set filename prefix
        if "104" in workflow:
            workflow["104"]["inputs"]["filename_prefix"] = f"video/AV-{chunk_scene_id}"
        
        # Configure LoRA switches (6 switches total)
        # Node "279:286" - Depth Control LoRA switch
        if "279:286" in workflow:
            workflow["279:286"]["inputs"]["switch"] = ENABLE_SWITCH_279_286
        
        # Node "279:288" - Canny Control LoRA switch
        if "279:288" in workflow:
            workflow["279:288"]["inputs"]["switch"] = ENABLE_SWITCH_279_288
        
        # Node "279:289" - Pose Control LoRA switch
        if "279:289" in workflow:
            workflow["279:289"]["inputs"]["switch"] = ENABLE_SWITCH_279_289
        
        # Node "279:290" - Detailer LoRA switch
        if "279:290" in workflow:
            workflow["279:290"]["inputs"]["switch"] = ENABLE_SWITCH_279_290
        
        # Node "279:291" - Final switch for first CFGGuider (279:239)
        if "279:291" in workflow:
            workflow["279:291"]["inputs"]["switch"] = ENABLE_SWITCH_279_291
        
        # Node "279:292" - Final switch for second CFGGuider (279:252)
        if "279:292" in workflow:
            workflow["279:292"]["inputs"]["switch"] = ENABLE_SWITCH_279_292
        
        return workflow

    def _extract_scene_id(self, chunk_scene_id: str) -> str:
        """Extract scene ID from chunk scene ID, removing 'scene_' prefix if present."""
        # Remove "scene_" prefix if present
        if chunk_scene_id.startswith("scene_"):
            return chunk_scene_id[6:]  # Remove "scene_" (6 chars)
        return chunk_scene_id

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
                
                # Check if this chunk is already complete
                if resumable_state and resumable_state.is_chunk_complete(chunk_scene_id):
                    cached_chunk = resumable_state.get_chunk_result(chunk_scene_id)
                    if cached_chunk and cached_chunk.get('path'):
                        chunk_path = cached_chunk['path']
                        if os.path.exists(chunk_path):
                            print(f"✅ Using cached chunk: {chunk_scene_id}")
                            generated_videos.append(chunk_path)
                            # Update current input image for next chunk continuity
                            cached_frame = cached_chunk.get('last_frame_filename', image_filename)
                            # Verify the cached frame file exists in ComfyUI input folder
                            cached_frame_path = os.path.join(self.comfyui_input_folder, cached_frame)
                            if os.path.exists(cached_frame_path):
                                current_input_image = cached_frame
                                print_flush(f"🔄 Cached chunk continuity: will use {cached_frame} for next chunk")
                            else:
                                # Frame file missing, try to find it from saved frames
                                print_flush(f"⚠️ Cached frame file missing: {cached_frame}, will search for last saved frame")
                                last_saved_frame = self._find_last_saved_frame(scene_id, chunk_suffix)
                                if last_saved_frame:
                                    # Copy to input folder
                                    last_frame_filename = f"{chunk_scene_id}_last_frame.png"
                                    comfyui_frame_path = os.path.join(self.comfyui_input_folder, last_frame_filename)
                                    shutil.copy2(last_saved_frame, comfyui_frame_path)
                                    current_input_image = last_frame_filename
                                    print_flush(f"🔄 Recovered cached chunk continuity: using {last_frame_filename}")
                                else:
                                    current_input_image = image_filename
                                    print_flush(f"⚠️ Could not recover continuity, will use original image")
                            continue
                        else:
                            print(f"⚠️ Cached chunk file missing, regenerating: {chunk_scene_id}")
                
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
                
                # Build workflow for this chunk (single image only, no guide images)
                workflow = self._build_av_workflow(chunk_scene_id, scene_description, current_input_image, duration, characters_data, motion_data, locations_data, chunk_frames)
                if not workflow:
                    print(f"ERROR: Failed to build workflow for {chunk_scene_id}")
                    continue

                # Print workflow summary
                if WORKFLOW_SUMMARY_ENABLED:
                    self._print_workflow_summary(workflow, f"AV: {chunk_scene_id}")

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
                generated_video = self._find_newest_output_with_prefix(f"AV-{chunk_scene_id}")
                if not generated_video:
                    print(f"ERROR: Could not find generated video for {chunk_scene_id}")
                    continue

                final_path = os.path.join(self.final_output_dir, f"{chunk_scene_id}.mp4")
                shutil.copy2(generated_video, final_path)
                print(f"Saved video: {final_path}")
                generated_videos.append(final_path)
                
                # Save chunk to checkpoint
                if resumable_state:
                    chunk_result = {
                        'path': final_path,
                        'chunk_id': chunk_scene_id,
                        'scene_id': scene_id,
                        'chunk_idx': chunk_idx,
                        'chunk_suffix': chunk_suffix,
                        'frame_count': chunk_frames,
                        'last_frame_filename': current_input_image
                    }
                    resumable_state.set_chunk_result(chunk_scene_id, chunk_result)
                    print(f"💾 Checkpoint saved for chunk: {chunk_scene_id}")
            
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
        
        # Search in video subdirectory and root
        search_dirs = [
            os.path.join(self.comfyui_output_folder, "video"),
            self.comfyui_output_folder
        ]
        
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            
            for root, _, files in os.walk(search_dir):
                for name in files:
                    # Check if filename contains prefix and is a video file
                    if prefix in name and any(name.lower().endswith(ext) for ext in video_exts):
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
            frame_duration = max(0.1, 1.0 / FRAMES_PER_SECOND)
            
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

    def generate_all_videos(self, force_regenerate: bool = False, resumable_state=None) -> Dict[str, str]:
        """Convert unique scenes to videos using story file as source of truth."""
        print_flush("🎬 Starting AV video generation process...")
        print_flush(f"📁 Story:     {self.story_file}")
        print_flush(f"🖼️  Images:    {self.scene_images_dir}")
        print_flush(f"🎥 Output:    {self.final_output_dir}")
        
        # Read story data
        durations, scenes = self.read_story_data()
        if not durations or not scenes:
            return {}
        
        # Read character data
        characters_data = self._read_character_data()
        if characters_data:
            print_flush(f"📖 Loaded {len(characters_data)} character descriptions")
        else:
            print_flush("⚠️ No character data found - videos will use scene descriptions only")
        
        # Read motion data
        motion_data = self._read_motion_data()
        if motion_data:
            print_flush(f"🎬 Loaded {len(motion_data)} motion prompts")
        else:
            print_flush("⚠️ No motion data found - videos will use basic motion prompts only")
        
        # Read location data
        locations_data = self._read_location_data()
        if locations_data:
            print_flush(f"📍 Loaded {len(locations_data)} location descriptions")
        else:
            print_flush("⚠️ No location data found")
        
        if len(durations) != len(scenes):
            print_flush(f"⚠️ Count mismatch: story entries={len(durations)} vs scenes={len(scenes)}. Proceeding with min count.")
        
        # Combine adjacent same scenes into single large durations
        combined_scenes = self.combine_adjacent_scenes(durations, scenes)
        if not combined_scenes:
            print_flush("❌ Failed to combine scenes or scenes appear non-contiguously.")
            return {}
        
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
        
        # Calculate total words and duration for ETA calculations
        self.total_video_words = sum(len(scene_info["description"].split()) for scene_info in scenes_to_process.values())
        self.total_video_duration = sum(scene_info["total_duration"] for scene_info in scenes_to_process.values())
        print_flush(f"📊 Total video words: {self.total_video_words:,}")
        print_flush(f"📊 Total video duration: {self.total_video_duration:.1f} seconds ({self.total_video_duration/60:.1f} minutes)")
        
        print_flush("=" * 60)

        results = {}
        # Initialize start time for time estimation
        self.start_time = time.time()
        
        print_flush(f"\n📊 AV VIDEO GENERATION PROGRESS")
        print_flush("=" * 120)
        
        for i, (scene_name, scene_info) in enumerate(scenes_to_process.items(), 1):
            scene_start_time = time.time()
            duration = scene_info["total_duration"]
            description = scene_info["description"]
            image_path = scene_info["image_path"]
            frame_count = self._calculate_frame_count(duration)
            
            eta = self.estimate_remaining_time(i, len(scenes_to_process), scene_description=description, duration=duration)
            print_flush(f"🔄 Scene {i}/{len(scenes_to_process)} - {description[:50]} ({duration:.2f}s) - Processing...")
            print_flush(f"📊 Estimated time remaining: {eta}")
            
            output_paths = self._generate_video(scene_name, description, image_path, duration, characters_data, motion_data, locations_data, resumable_state)
            
            scene_processing_time = time.time() - scene_start_time
            self.processing_times.append(scene_processing_time)
            
            if output_paths:
                eta = self.estimate_remaining_time(i, len(scenes_to_process), scene_processing_time, description, duration)
                if len(output_paths) == 1:
                    results[scene_name] = output_paths[0]
                    print_flush(f"✅ Scene {i}/{len(scenes_to_process)} - {description[:50]} ({duration:.2f}s) - Completed in {self.format_processing_time(scene_processing_time)}")
                    print_flush(f"📊 Estimated time remaining: {eta}")
                    print_flush(f"✅ Generated video: {scene_name}")
                else:
                    # Multiple chunks generated
                    results[scene_name] = output_paths
                    print_flush(f"✅ Scene {i}/{len(scenes_to_process)} - {description[:50]} ({duration:.2f}s) - Completed in {self.format_processing_time(scene_processing_time)}")
                    print_flush(f"📊 Estimated time remaining: {eta}")
                    print_flush(f"✅ Generated {len(output_paths)} video chunks: {scene_name}")
                    for j, path in enumerate(output_paths, 1):
                        print_flush(f"   Chunk {j}: {os.path.basename(path)}")
            else:
                eta = self.estimate_remaining_time(i, len(scenes_to_process), scene_processing_time, description, duration)
                print_flush(f"❌ Scene {i}/{len(scenes_to_process)} - {description[:50]} ({duration:.2f}s) - Failed after {self.format_processing_time(scene_processing_time)}")
                print_flush(f"📊 Estimated time remaining: {eta}")
                print_flush(f"❌ Failed: {scene_name}")
        
        elapsed = time.time() - self.start_time
        
        print_flush("\n📊 AV GENERATION SUMMARY:")
        print_flush(f"   • Total story entries: {len(durations)}")
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
        print("   " + "="*50)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AV videos from story and scene images")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration of all videos")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")
    args = parser.parse_args()
    
    generator = AVVideoGenerator()
    
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
    
    # Process all unique scenes from story
    results = generator.generate_all_videos(force_regenerate=args.force, resumable_state=resumable_state)
    
    if results:
        total_videos = sum(len(v) if isinstance(v, list) else 1 for v in results.values())
        print_flush(f"\n✅ Successfully generated {total_videos} AV videos from {len(results)} scenes:")
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
    exit(main())
