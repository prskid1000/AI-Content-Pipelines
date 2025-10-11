import os
import json
import time
import shutil
import re
import argparse
import requests
import random
from PIL import Image
from pathlib import Path

# Feature flags
ENABLE_RESUMABLE_MODE = True
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them
WORKFLOW_SUMMARY_ENABLED = False # Set to True to enable workflow summary printing

# Image resizing configuration (characters only)
# Character image resize factor: 0.125 (12.5% of original size) - Better aspect ratio for stitching
CHARACTER_RESIZE_FACTOR = 0.5

# Image compression configuration
# JPEG quality: 1-100 (100 = best quality, larger file; 1 = worst quality, smaller file)
IMAGE_COMPRESSION_QUALITY = 90

# Character prompt handling modes
# "IMAGE_TEXT" Send character images + character details appended from characters.txt
# "TEXT" Only character details from characters.txt
# "IMAGE" Only images
# "NONE" Skip character processing entirely

# Location prompt handling modes
# "IMAGE_TEXT" Use location images as reference images (stitched with characters) + location details appended from locations.txt
# "TEXT" Only location details from locations.txt (replace {{loc_X}} with descriptions)
# "IMAGE" Only location images as reference images (stitched with characters, no text replacement)
# "NONE" Skip location processing entirely
# Note: LATENT_MODE controls whether location images are ALSO used as latent input (separate from grouping)

# Location prompt handling modes
# "IMAGE_TEXT" Use location images as reference images (stitched with characters) + location details appended from locations.txt
# "TEXT" Only location details from locations.txt (replace {{loc_X}} with descriptions)
# "IMAGE" Only location images as reference images (stitched with characters, no text replacement)
# Note: LATENT_MODE controls whether location images are ALSO used as latent input (separate from grouping)

# HARDCODED CHARACTER MODE - Change this to switch modes
ACTIVE_CHARACTER_MODE = "IMAGE"

# HARDCODED LOCATION MODE - Change this to switch modes
ACTIVE_LOCATION_MODE = "TEXT"

WORD_FACTOR = 6
LOCATION_WORD_LIMIT = 120
CHARACTER_WORD_LIMIT = 120

# Image Resolution Constants
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Latent Input Mode Configuration
LATENT_MODE = "LATENT"  # "LATENT" for normal noise generation, "IMAGE" for load image input
LATENT_DENOISING_STRENGTH = 0.90  # Denoising strength when using IMAGE mode (0.0-1.0, higher = more change)

# Image Stitching Configuration (1-5)
IMAGE_STITCH_COUNT = 1  # Number of images to stitch together in each group

# LoRA Configuration
USE_LORA = True  # Set to False to disable LoRA usage in workflow
LORA_MODE = "serial"  # "serial" for independent LoRA application, "chained" for traditional chaining

# LoRA Configuration
# Each LoRA can be configured for both serial and chained modes
# For serial mode: each LoRA runs independently with its own steps and denoising
# For chained mode: LoRAs are applied in sequence to the same generation
LORAS = [
    {
        "name": "FLUX.1-Turbo-Alpha.safetensors",
        "strength_model": 2.0,    # Model strength (0.0 - 2.0)
        "strength_clip": 2.0,     # CLIP strength (0.0 - 2.0)
        "bypass_model": False,    # Set to True to bypass model part of this LoRA
        "bypass_clip": False,     # Set to True to bypass CLIP part of this LoRA
        "enabled": True,          # Set to False to disable this LoRA entirely
        
        # Serial mode specific settings (only used when LORA_MODE = "serial")
        "steps": 9,               # Sampling steps for this LoRA (serial mode only)
        "denoising_strength": 1, # Denoising strength (0.0 - 1.0) (serial mode only)
        "save_intermediate": True, # Save intermediate results for debugging (serial mode only)
        "use_only_intermediate": False # Set to True to disable character images and use only intermediate result
    },
    {
        "name": "FLUX.1-Turbo-Alpha.safetensors",
        "strength_model": 2.0,    # Model strength (0.0 - 2.0)
        "strength_clip": 2.0,     # CLIP strength (0.0 - 2.0)
        "bypass_model": False,    # Set to True to bypass model part of this LoRA
        "bypass_clip": False,     # Set to True to bypass CLIP part of this LoRA
        "enabled": False,          # Set to False to disable this LoRA entirely
        
        # Serial mode specific settings (only used when LORA_MODE = "serial")
        "steps": 9,               # Sampling steps for this LoRA (serial mode only)
        "denoising_strength": 0.1, # Denoising strength (0.0 - 1.0) (serial mode only)
        "save_intermediate": True, # Save intermediate results for debugging (serial mode only)
        "use_only_intermediate": True # Set to True to disable character images and use only intermediate result
    }
]

# Sampling Configuration
SAMPLING_STEPS = 25 # Number of sampling steps (higher = better quality, slower)

# Negative Prompt Configuration
USE_NEGATIVE_PROMPT = True  # Set to True to enable negative prompts, False to disable
NEGATIVE_PROMPT = "blur, distorted, text, watermark, extra limbs, bad anatomy, poorly drawn, asymmetrical, malformed, disfigured, ugly, bad proportions, plastic texture, artificial looking, cross-eyed, missing fingers, extra fingers, bad teeth, missing teeth, unrealistic"

# Random Seed Configuration
USE_RANDOM_SEED = True  # Set to True to use random seed, False to use fixed seed - > Use when correcting images by regenerating
FIXED_SEED = 333555666  # Fixed seed value when USE_RANDOM_SEED is False

# Location Information Configuration (now handled by ACTIVE_LOCATION_MODE above)

ART_STYLE = "Realistic Anime"

USE_SUMMARY_TEXT = True  # Set to True to use summary text

class ResumableState:
    """Manages resumable state for expensive scene generation operations."""
    
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
            "scenes": {
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
    
    def is_scene_complete(self, scene_id: str) -> bool:
        """Check if scene generation is complete."""
        return scene_id in self.state["scenes"]["completed"]
    
    def get_scene_result(self, scene_id: str) -> dict:
        """Get scene generation result."""
        return self.state["scenes"]["results"].get(scene_id, {})
    
    def set_scene_result(self, scene_id: str, result: dict):
        """Set scene generation result and mark as complete."""
        self.state["scenes"]["results"][scene_id] = result
        if scene_id not in self.state["scenes"]["completed"]:
            self.state["scenes"]["completed"].append(scene_id)
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
    
    def validate_and_cleanup_results(self, output_scene_dir: str = None) -> int:
        """Validate that all completed scene files actually exist and clean up missing entries.
        
        Args:
            output_scene_dir: Path to the output/scene directory to check for actual files
        
        Returns:
            int: Number of entries cleaned up (removed from completed list)
        """
        cleaned_count = 0
        scenes_to_remove = []
        
        print(f"Validating {len(self.state['scenes']['completed'])} completed scenes against output/scene directory...")
        
        # Check each completed scene
        for scene_id in self.state["scenes"]["completed"]:
            result = self.state["scenes"]["results"].get(scene_id, {})
            file_path = result.get('path', '')
            
            # Check if file actually exists
            if not file_path or not os.path.exists(file_path):
                print(f"Precheck: File missing for {scene_id} - marking as not completed")
                scenes_to_remove.append(scene_id)
                cleaned_count += 1
            elif output_scene_dir:
                # Additional check: verify file exists in output/scene directory
                expected_scene_file = os.path.join(output_scene_dir, f"{scene_id}.png")
                if not os.path.exists(expected_scene_file):
                    print(f"Precheck: Scene file missing in output/scene directory for {scene_id} - marking as not completed")
                    print(f"  Expected: {expected_scene_file}")
                    scenes_to_remove.append(scene_id)
                    cleaned_count += 1
                else:
                    print(f"Precheck: ✓ {scene_id} validated in output/scene directory")
        
        # Remove invalid entries
        for scene_id in scenes_to_remove:
            if scene_id in self.state["scenes"]["completed"]:
                self.state["scenes"]["completed"].remove(scene_id)
            if scene_id in self.state["scenes"]["results"]:
                del self.state["scenes"]["results"][scene_id]
            
            # Also clear any LoRA progress for this scene
            lora_progress_key = f"{scene_id}_lora_progress"
            if "lora_progress" in self.state and lora_progress_key in self.state["lora_progress"]:
                del self.state["lora_progress"][lora_progress_key]
                print(f"Precheck: Cleared LoRA progress for {scene_id}")
        
        # Save cleaned state if any changes were made
        if cleaned_count > 0:
            self._save_state()
            print(f"Precheck: Cleaned up {cleaned_count} invalid entries from checkpoint")
        
        return cleaned_count
    
    def sync_with_output_directory(self, output_scene_dir: str) -> int:
        """Sync resumable state with actual files in output directory.
        
        This method finds files that exist in the output directory but aren't tracked
        in the resumable state, and adds them to the completed list.
        
        Args:
            output_scene_dir: Path to the output/scene directory to check for actual files
        
        Returns:
            int: Number of files found and added to completed list
        """
        if not os.path.exists(output_scene_dir):
            print(f"Output/scene directory does not exist: {output_scene_dir}")
            return 0
            
        added_count = 0
        tracked_scenes = set(self.state["scenes"]["completed"])
        
        print(f"Scanning output/scene directory for untracked files: {output_scene_dir}")
        
        # Find all .png files in the output directory
        for filename in os.listdir(output_scene_dir):
            if filename.endswith('.png'):
                scene_id = filename[:-4]  # Remove .png extension
                
                # If this scene isn't tracked, add it to completed
                if scene_id not in tracked_scenes:
                    file_path = os.path.join(output_scene_dir, filename)
                    result = {
                        'path': file_path,
                        'scene_id': scene_id,
                        'scene_description': f"Auto-detected from output/scene directory",
                        'character_names': [],
                        'auto_detected': True
                    }
                    self.state["scenes"]["results"][scene_id] = result
                    self.state["scenes"]["completed"].append(scene_id)
                    added_count += 1
                    print(f"Auto-detected completed scene: {scene_id} -> {file_path}")
                else:
                    print(f"Scene already tracked: {scene_id}")
        
        # Save state if any files were added
        if added_count > 0:
            self._save_state()
            print(f"Auto-detection: Added {added_count} scenes from output/scene directory")
        else:
            print("No untracked scenes found in output/scene directory")
        
        return added_count
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        completed = len(self.state["scenes"]["completed"])
        total = len(self.state["scenes"]["results"]) + len([k for k in self.state["scenes"]["results"].keys() if k not in self.state["scenes"]["completed"]])
        
        return f"Progress: Scenes({completed}/{total})"


class SceneGenerator:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188/"):
        self.comfyui_url = comfyui_url
        self.character_mode = ACTIVE_CHARACTER_MODE
        self.location_mode = ACTIVE_LOCATION_MODE
        # ComfyUI saves images under this folder
        self.comfyui_output_folder = "../../ComfyUI/output"
        # ComfyUI input folder where we need to copy character images
        self.comfyui_input_folder = "../../ComfyUI/input"
        # Final destination inside this repo
        self.final_output_dir = "../output/scene"
        self.intermediate_output_dir = "../output/lora"
        self.scene_file = "../input/3.scene.txt"
        self.character_file = "../input/3.character.txt" if USE_SUMMARY_TEXT else "../input/2.character.txt"
        self.location_file = "../input/3.location.txt" if USE_SUMMARY_TEXT else "../input/2.location.txt"
        self.workflow_file = "../workflow/scene.json"
        self.character_images_dir = "../output/characters"
        # Latent image input file path
        self.latent_image_path = "../input/3.latent.small.png"

        # Time estimation tracking
        self.processing_times = []
        self.start_time = None

        # Create output directories
        os.makedirs(self.final_output_dir, exist_ok=True)
        os.makedirs(self.intermediate_output_dir, exist_ok=True)
        # Ensure ComfyUI input directory exists
        os.makedirs(self.comfyui_input_folder, exist_ok=True)

    def estimate_remaining_time(self, current_scene: int, total_scenes: int, scene_processing_time: float = None, scene_description: str = None) -> str:
        """Estimate remaining time based on words per minute - simple and accurate approach"""
        
        # Calculate total words in all scene descriptions
        total_scene_words = getattr(self, 'total_scene_words', 0)
        
        # For first scene with no data, provide a reasonable initial estimate
        if not self.processing_times and scene_processing_time is None:
            if total_scene_words > 0:
                # Initial estimate: assume 200 words per minute for scene generation (image generation)
                words_per_minute = 200
                remaining_words = total_scene_words - (current_scene - 1) * (total_scene_words // total_scenes)
                estimated_remaining_minutes = remaining_words / words_per_minute
                estimated_remaining_seconds = estimated_remaining_minutes * 60
            else:
                # Fallback: assume 120 seconds per scene (2 minutes for image generation)
                remaining_scenes = total_scenes - current_scene
                estimated_remaining_seconds = remaining_scenes * 120.0
            
            return self._format_time_with_confidence(estimated_remaining_seconds, confidence="low")
        
        # Calculate actual words per minute from completed scenes
        if scene_processing_time and scene_description:
            scene_words = len(scene_description.split())
            scene_minutes = scene_processing_time / 60
            if scene_minutes > 0:
                current_wpm = scene_words / scene_minutes
                # Store word processing data for better estimation
                if not hasattr(self, 'word_processing_data'):
                    self.word_processing_data = []
                self.word_processing_data.append({'words': scene_words, 'time': scene_processing_time, 'wpm': current_wpm})
        
        # Use word-based estimation if we have word processing data
        if hasattr(self, 'word_processing_data') and self.word_processing_data:
            # Calculate average words per minute from actual data
            total_words_processed = sum(data['words'] for data in self.word_processing_data)
            total_time_processed = sum(data['time'] for data in self.word_processing_data)
            
            if total_time_processed > 0:
                actual_wpm = total_words_processed / (total_time_processed / 60)
                
                # Estimate remaining words
                words_processed_so_far = sum(data['words'] for data in self.word_processing_data)
                remaining_words = total_scene_words - words_processed_so_far
                
                # Calculate remaining time based on actual WPM
                estimated_remaining_minutes = remaining_words / actual_wpm
                estimated_remaining_seconds = estimated_remaining_minutes * 60
                
                # Determine confidence based on data points
                confidence = "low"
                if len(self.word_processing_data) >= 5:
                    confidence = "high"
                elif len(self.word_processing_data) >= 3:
                    confidence = "medium"
                
                return self._format_time_with_confidence(estimated_remaining_seconds, confidence)
        
        # Fallback to scene-based estimation if no word data available
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

    def _read_scene_data(self) -> dict[str, str]:
        """Parse scene data from input file."""
        scenes = {}
        try:
            with open(self.scene_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            # Split by double newlines and process each entry
            entries = [entry.strip() for entry in content.split('\n\n') if entry.strip()]
            for entry in entries:
                # Handle both formats: (scene_id) and [scene_id]:
                # First try the new format with parentheses
                match = re.match(r'\(([^)]+)\)\s*(.+)', entry, re.DOTALL)
                if match:
                    scenes[match.group(1).strip()] = match.group(2).strip()
                else:
                    # Fallback to old format with brackets
                    match = re.match(r'\[([^\]]+)\]\s*:?\s*(.+)', entry, re.DOTALL)
                    if match:
                        scenes[match.group(1).strip()] = match.group(2).strip()
        except Exception as e:
            print(f"ERROR: Failed to read scene data: {e}")
        return scenes

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
            print(f"ERROR: Failed to read character data: {e}")
        return characters

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
            print(f"ERROR: Failed to read location data: {e}")
        return locations
    
    def _get_ordinal_suffix(self, num: int) -> str:
        """Get ordinal suffix for numbers (1st, 2nd, 3rd, etc.)."""
        if 10 <= num % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
        return suffix

    def _get_position_description(self, position_in_group: int, group_number: int) -> str:
        """Get position description for character placement."""
        if IMAGE_STITCH_COUNT == 1:
            # When only one character per image, use simpler description
            return f"Character in Image {group_number}"
        elif IMAGE_STITCH_COUNT == 2:
            if position_in_group == 1:
                # When two characters per image, use simple description
                return f"Leftmost Character in Image {group_number}"
            else:
                return f"Rightmost Character in Image {group_number}"
        elif IMAGE_STITCH_COUNT == 3:
            if position_in_group == 1:
                return f"Leftmost Character in Image {group_number}"
            elif position_in_group == 2:
                return f"Middle Character in Image {group_number}"
            else:
                return f"Rightmost Character in Image {group_number}"
        elif IMAGE_STITCH_COUNT == 4:
            if position_in_group == 1:
                return f"Leftmost Character in Image {group_number}"
            elif position_in_group == 2:
                return f"Middle Left Character in Image {group_number}"
            elif position_in_group == 3:
                return f"Middle Right Character in Image {group_number}"
            else:
                return f"Rightmost Character in Image {group_number}"
        elif IMAGE_STITCH_COUNT == 5:
            if position_in_group == 1:
                return f"Leftmost Character in Image {group_number}"
            elif position_in_group == 2:
                return f"Middle Left Character in Image {group_number}"
            elif position_in_group == 3:
                return f"Middle Right Character in Image {group_number}"
            elif position_in_group == 4:
                return f"Rightmost Character in Image {group_number}"
            else:
                return f"Center Character in Image {group_number}"
        else:
            # When multiple characters per image, use ordinal position
            ordinal_suffix = self._get_ordinal_suffix(position_in_group)
            return f"{position_in_group}{ordinal_suffix} Character from Left in Image {group_number}"

    def _get_location_position_description(self, position_in_group: int, group_number: int) -> str:
        """Get position description for location placement."""
        if IMAGE_STITCH_COUNT == 1:
            # When only one location per image, use simpler description
            return f"Location in Image {group_number}"
        elif IMAGE_STITCH_COUNT == 2:
            if position_in_group == 1:
                # When two locations per image, use simple description
                return f"Leftmost Location in Image {group_number}"
            else:
                return f"Rightmost Location in Image {group_number}"
        elif IMAGE_STITCH_COUNT == 3:
            if position_in_group == 1:
                return f"Leftmost Location in Image {group_number}"
            elif position_in_group == 2:
                return f"Middle Location in Image {group_number}"
            else:
                return f"Rightmost Location in Image {group_number}"
        elif IMAGE_STITCH_COUNT == 4:
            if position_in_group == 1:
                return f"Leftmost Location in Image {group_number}"
            elif position_in_group == 2:
                return f"Middle Left Location in Image {group_number}"
            elif position_in_group == 3:
                return f"Middle Right Location in Image {group_number}"
            else:
                return f"Rightmost Location in Image {group_number}"
        elif IMAGE_STITCH_COUNT == 5:
            if position_in_group == 1:
                return f"Leftmost Location in Image {group_number}"
            elif position_in_group == 2:
                return f"Middle Left Location in Image {group_number}"
            elif position_in_group == 3:
                return f"Middle Right Location in Image {group_number}"
            elif position_in_group == 4:
                return f"Rightmost Location in Image {group_number}"
            else:
                return f"Center Location in Image {group_number}"
        else:
            # When multiple locations per image, use ordinal position
            ordinal_suffix = self._get_ordinal_suffix(position_in_group)
            return f"{position_in_group}{ordinal_suffix} Location from Left in Image {group_number}"

    def _replace_location_references(self, scene_description: str, location_ids: list[str], locations_data: dict[str, str]) -> str:
        """Replace {{loc_id}} references with position and location descriptions."""
        if not location_ids:
            return scene_description
        
        def replace_func(match):
            full_match = match.group(0)
            # Extract loc_id (handle both {{loc_id}} and {{loc_id, description}} formats)
            content = match.group(1).strip()
            loc_id = content.split(',')[0].strip() if ',' in content else content
            
            # Find the index of this location in the list
            if loc_id in location_ids:
                loc_index = location_ids.index(loc_id)
                # Calculate position information
                position_in_group = (loc_index % IMAGE_STITCH_COUNT) + 1
                group_number = (loc_index // IMAGE_STITCH_COUNT) + 1
                
                # Create position description using helper method
                position_desc = self._get_location_position_description(position_in_group, group_number)
                
                # Only add location details in TEXT and IMAGE_TEXT modes, skip in NONE and IMAGE mode
                # Only add position description in IMAGE and IMAGE_TEXT modes, skip in NONE and TEXT mode
                return f"USE{ "," +position_desc if self.location_mode in ['IMAGE', 'IMAGE_TEXT'] else ''}, {locations_data.get(loc_id, '')[:(LOCATION_WORD_LIMIT * WORD_FACTOR)] if self.location_mode in ['TEXT', 'IMAGE_TEXT'] else ''} to illustrate the scene - "
            else:
                # Location not found in the list, keep original
                return full_match
        
        # Replace {{loc_id}} patterns with position format
        result = re.sub(r'\{\{([^}]+)\}\}', replace_func, scene_description)
        return result

    def _replace_character_references(self, scene_description: str, character_names: list[str], characters_data) -> str:
        """Replace ((character_name)) references with position format in scene description."""
        if not character_names:
            return scene_description
        
        def replace_character_func(match):
            full_match = match.group(0)
            char_name = match.group(1).strip()
            
            # Find the index of this character in the list
            if char_name in character_names:
                char_index = character_names.index(char_name)
                # Calculate position information
                position_in_group = (char_index % IMAGE_STITCH_COUNT) + 1
                group_number = (char_index // IMAGE_STITCH_COUNT) + 1
                
                # Create position description using helper method
                position_desc = self._get_position_description(position_in_group, group_number)

                # Only add character details in TEXT and IMAGE_TEXT modes, skip in NONE  and IMAGE mode.Only add position description in IMAGE and IMAGE_TEXT modes, skip in NONE and TEXT mode.
                return f"\nUSE{ "," +position_desc if self.character_mode in ['IMAGE', 'IMAGE_TEXT'] else ''}, {characters_data[char_name][:(CHARACTER_WORD_LIMIT * WORD_FACTOR)] if self.character_mode in ['TEXT', 'IMAGE_TEXT'] else ''} to illustrate the character - "
            else:
                # Character not found in the list, keep original
                return full_match
        
        # Replace ((character_name)) patterns with position format
        result = re.sub(r'\(\(([^)]+)\)\)', replace_character_func, scene_description)
        return result

    def _get_master_prompt(self) -> str:
        """Get the master prompt content."""
        return """Create a 16K ultra-high-resolution, illustration in the style of {ART_STYLE}.
        """.format(ART_STYLE=ART_STYLE)

    def _get_seed(self) -> int:
        """Get seed value based on configuration."""
        if USE_RANDOM_SEED:
            return random.randint(0, 2**32 - 1)
        else:
            return FIXED_SEED

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

    def _extract_location_ids_from_scene(self, scene_description: str) -> list[str]:
        """Extract location IDs from scene description."""
        location_matches = re.findall(r'\{\{([^}]+)\}\}', scene_description)
        seen = set()
        unique_locations = []
        for loc in location_matches:
            loc = loc.strip()
            if loc and loc not in seen:
                seen.add(loc)
                unique_locations.append(loc)
        return unique_locations

    def _get_location_image_path(self, location_id: str) -> str | None:
        """Get the path to a location's generated image."""
        # Location images are stored as loc_X.png in the locations output directory
        location_images_dir = "../output/locations"
        image_path = os.path.join(location_images_dir, f"{location_id}.png")
        return image_path if os.path.exists(image_path) else None

    def _get_location_latent_image_path(self, scene_description: str) -> str | None:
        """Get the appropriate location image path for latent input based on scene description.
        
        This is controlled by LATENT_MODE, not location mode. Location mode only controls 
        whether location images are added to groups/stitching.
        """
        # LATENT_MODE controls whether to use location images as latent input
        # Location mode only controls grouping/stitching
        location_ids = self._extract_location_ids_from_scene(scene_description)
        
        if not location_ids:
            # No location references found, use default latent image
            return self.latent_image_path
        
        # Use the first location found (in case of multiple locations)
        primary_location_id = location_ids[0]
        location_image_path = self._get_location_image_path(primary_location_id)
        
        if location_image_path:
            print(f"Using location image for latent input: {primary_location_id} -> {location_image_path}")
            return location_image_path
        else:
            print(f"Location image not found for {primary_location_id}, using default latent image")
            return self.latent_image_path

    def _get_character_image_path(self, character_name: str) -> str | None:
        """Get the path to a character's generated image."""
        clean_name = re.sub(r'[^\w\s.-]', '', character_name).strip()
        clean_name = re.sub(r'[-\s]+', '_', clean_name)
        image_path = os.path.join(self.character_images_dir, f"{clean_name}.png")
        return image_path if os.path.exists(image_path) else None

    def _copy_character_images_to_comfyui(self, character_names: list[str]) -> dict[str, str]:
        """Copy, resize, and compress character images to ComfyUI input directory."""
        copied_images = {}
        for char_name in character_names:
            source_path = self._get_character_image_path(char_name)
            if source_path:
                clean_name = re.sub(r'[^\w\s.-]', '', char_name).strip()
                clean_name = re.sub(r'[-\s]+', '_', clean_name)
                dest_path = os.path.join(self.comfyui_input_folder, f"{clean_name}.jpg")
                
                try:
                    # Open, resize, compress and save the image
                    with Image.open(source_path) as img:
                        # Convert RGBA to RGB if necessary (for JPEG compatibility)
                        if img.mode in ('RGBA', 'LA'):
                            # Create white background
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'RGBA':
                                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                            else:
                                background.paste(img)
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Resize the image using the resize factor
                        new_width = int(img.width * CHARACTER_RESIZE_FACTOR)
                        new_height = int(img.height * CHARACTER_RESIZE_FACTOR)
                        
                        # Resize with aspect ratio preserved
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Save with compression
                        img.save(dest_path, 'JPEG', quality=IMAGE_COMPRESSION_QUALITY, optimize=True)
                    
                    copied_images[char_name] = dest_path
                    
                    # Get file sizes for logging
                    original_size = os.path.getsize(source_path) / 1024  # KB
                    compressed_size = os.path.getsize(dest_path) / 1024  # KB
                    compression_ratio = (1 - compressed_size / original_size) * 100
                    
                    print(f"Compressed {char_name} image: {original_size:.1f}KB → {compressed_size:.1f}KB ({compression_ratio:.1f}% reduction)")
                    
                except Exception as e:
                    print(f"ERROR: Failed to compress image for {char_name}: {e}")
                    # Fallback to simple copy
                    shutil.copy2(source_path, dest_path.replace('.jpg', '.png'))
                    copied_images[char_name] = dest_path.replace('.jpg', '.png')
            else:
                print(f"WARNING: Character image not found for: {char_name}")
        return copied_images

    def _copy_location_images_to_comfyui(self, location_ids: list[str]) -> dict[str, str]:
        """Copy, resize, and compress location images to ComfyUI input directory."""
        copied_images = {}
        for location_id in location_ids:
            source_path = self._get_location_image_path(location_id)
            if source_path:
                clean_name = re.sub(r'[^\w\s.-]', '', location_id).strip()
                clean_name = re.sub(r'[-\s]+', '_', clean_name)
                dest_path = os.path.join(self.comfyui_input_folder, f"loc_{clean_name}.jpg")
                
                try:
                    # Open, resize, compress and save the image
                    with Image.open(source_path) as img:
                        # Convert RGBA to RGB if necessary (for JPEG compatibility)
                        if img.mode in ('RGBA', 'LA'):
                            # Create white background
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'RGBA':
                                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                            else:
                                background.paste(img)
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        # Resize the image using the resize factor
                        new_width = int(img.width * CHARACTER_RESIZE_FACTOR)
                        new_height = int(img.height * CHARACTER_RESIZE_FACTOR)
                        
                        # Resize with aspect ratio preserved
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Save with compression
                        img.save(dest_path, 'JPEG', quality=IMAGE_COMPRESSION_QUALITY, optimize=True)
                    
                    copied_images[location_id] = dest_path
                    
                    # Get file sizes for logging
                    original_size = os.path.getsize(source_path) / 1024  # KB
                    compressed_size = os.path.getsize(dest_path) / 1024  # KB
                    compression_ratio = (1 - compressed_size / original_size) * 100
                    
                    print(f"Compressed {location_id} image: {original_size:.1f}KB → {compressed_size:.1f}KB ({compression_ratio:.1f}% reduction)")
                    
                except Exception as e:
                    print(f"ERROR: Failed to compress image for {location_id}: {e}")
                    # Fallback to simple copy
                    shutil.copy2(source_path, dest_path.replace('.jpg', '.png'))
                    copied_images[location_id] = dest_path.replace('.jpg', '.png')
            else:
                print(f"WARNING: Location image not found for: {location_id}")
        return copied_images

    def _load_base_workflow(self) -> dict:
        """Load the base scene workflow and modify based on LoRA settings."""
        try:
            with open(self.workflow_file, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            
            # Modify workflow based on USE_LORA setting
            if USE_LORA:
                if LORA_MODE == "serial":
                    print("Serial LoRA mode enabled - will create separate workflow")
                    # For serial mode, we don't modify the base workflow here
                    # The serial workflow will be created separately
                else:
                    # Handle multiple LoRAs in series (chained mode)
                    self._apply_loras(workflow)
                    print("Chained LoRA mode enabled in workflow")
            else:
                # Remove all LoRA nodes if they exist
                self._remove_all_lora_nodes(workflow)
                print("LoRA disabled in workflow")
            
            # Set sampling steps and seed for all modes
            if USE_LORA and LORA_MODE == "chained":
                workflow["16"]["inputs"]["steps"] = SAMPLING_STEPS
                seed = self._get_seed()
                workflow["16"]["inputs"]["seed"] = seed
                print(f"Sampling steps set to: {SAMPLING_STEPS}")
                print(f"Seed set to: {seed}")
            elif not USE_LORA:
                # Set seed for non-LoRA mode
                workflow["16"]["inputs"]["steps"] = SAMPLING_STEPS
                seed = self._get_seed()
                workflow["16"]["inputs"]["seed"] = seed
                print(f"Sampling steps set to: {SAMPLING_STEPS}")
                print(f"Seed set to: {seed}")
            
            return workflow
        except Exception as e:
            print(f"ERROR: Failed to load workflow: {e}")
            return {}

    def _apply_loras(self, workflow: dict) -> None:
        """Apply LoRAs based on mode (serial or chained)."""
        if LORA_MODE == "serial":
            self._apply_loras_serial(workflow)
        else:
            self._apply_loras_chained(workflow)
    
    def _apply_loras_serial(self, workflow: dict) -> None:
        """Apply LoRAs in serial mode - each LoRA runs independently.
        
        Serial mode logic:
        - Each LoRA runs in a separate workflow execution
        - LoRA 0: Uses EmptySD3LatentImage (latent mode) or image input (image mode)
        - LoRA 1+: Uses previous LoRA output as input
        """
        enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
        
        if not enabled_loras:
            print("No enabled LoRAs found in LORAS configuration")
            return
        
        print(f"Serial LoRA mode: {len(enabled_loras)} LoRAs will run independently")
        print("Note: Serial mode requires separate workflow execution for each LoRA")
    
    def _apply_loras_chained(self, workflow: dict) -> None:
        """Apply LoRAs in series with individual bypass options (chained mode)."""
        enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
        
        if not enabled_loras:
            print("No enabled LoRAs found in LORAS configuration")
            return
        
        print(f"Applying {len(enabled_loras)} LoRAs in chained mode...")
        
        # Get initial model and clip connections
        model_input = self._find_node_by_class(workflow, "UnetLoaderGGUF") or ["41", 0]
        clip_input = self._find_node_by_class(workflow, ["DualCLIPLoader", "TripleCLIPLoader"]) or ["10", 0]
        
        last_model_output = model_input
        last_clip_output = clip_input
        
        # Apply each LoRA in series
        for i, lora_config in enumerate(enabled_loras):
            lora_node_id = f"lora_{i + 1}"
            
            # Create LoRA node inputs
            lora_inputs = {
                "lora_name": lora_config["name"],
                "model": last_model_output,
                "clip": last_clip_output
            }
            
            # Apply strength settings with bypass options
            if lora_config.get("bypass_model", False):
                lora_inputs["strength_model"] = 0.0
                print(f"  LoRA {i + 1} ({lora_config['name']}): Model bypassed")
            else:
                lora_inputs["strength_model"] = lora_config.get("strength_model", 1.0)
                print(f"  LoRA {i + 1} ({lora_config['name']}): Model strength {lora_inputs['strength_model']}")
            
            if lora_config.get("bypass_clip", False):
                lora_inputs["strength_clip"] = 0.0
                print(f"  LoRA {i + 1} ({lora_config['name']}): CLIP bypassed")
            else:
                lora_inputs["strength_clip"] = lora_config.get("strength_clip", 1.0)
                print(f"  LoRA {i + 1} ({lora_config['name']}): CLIP strength {lora_inputs['strength_clip']}")
            
            # Create LoRA node
            workflow[lora_node_id] = {
                "inputs": lora_inputs,
                "class_type": "LoraLoader",
                "_meta": {"title": f"Load LoRA {i + 1}: {lora_config['name']}"}
            }
            
            # Update connections for next LoRA in chain
            last_model_output = [lora_node_id, 0]
            last_clip_output = [lora_node_id, 1]
        
        # Connect final LoRA outputs to workflow nodes
        self._update_node_connections(workflow, "KSampler", "model", last_model_output)
        self._update_node_connections(workflow, ["CLIPTextEncode", "CLIP Text Encode (Prompt)"], "clip", last_clip_output)
        
        print(f"LoRAs chain completed with {len(enabled_loras)} LoRAs")
    
    
    def _remove_all_lora_nodes(self, workflow: dict) -> None:
        """Remove all LoRA nodes from workflow."""
        # Ensure KSampler uses base model directly
        workflow["16"]["inputs"]["model"] = ["41", 0]
        # Ensure CLIPTextEncode uses base clip directly
        workflow["33"]["inputs"]["clip"] = ["10", 0]
        
        # Remove LoRA nodes (lora_1, lora_2, etc.)
        nodes_to_remove = []
        for node_id in workflow.keys():
            if node_id.startswith("lora_"):
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            del workflow[node_id]
    
    def _print_workflow_summary(self, workflow: dict, title: str) -> None:
        """Print a simplified workflow summary showing flow to sampler inputs."""
        if not WORKFLOW_SUMMARY_ENABLED:
            return
        print(f"\n{'='*60}")
        print(f"WORKFLOW SUMMARY: {title}")
        print(f"{'='*60}")
        
        # Find KSampler
        ksampler = self._find_node_by_class(workflow, "KSampler")
        if not ksampler:
            print("❌ No KSampler found in workflow")
            return
            
        sampler_id = ksampler[0]
        sampler_inputs = workflow[sampler_id].get("inputs", {})
        
        print(f"\n🎲 KSampler ({sampler_id}) Configuration:")
        print(f"   Steps: {sampler_inputs.get('steps', 'N/A')}")
        print(f"   Denoising: {sampler_inputs.get('denoise', 'N/A')}")
        print(f"   Seed: {sampler_inputs.get('seed', 'N/A')}")
        print(f"   CFG: {sampler_inputs.get('cfg', 'N/A')}")
        print(f"   Sampler: {sampler_inputs.get('sampler_name', 'N/A')}")
        print(f"   Scheduler: {sampler_inputs.get('scheduler', 'N/A')}")
        
        print(f"\n📋 FLOW TO SAMPLER INPUTS:")
        
        # Trace each sampler input back to its source
        for input_name, input_value in sampler_inputs.items():
            if isinstance(input_value, list) and len(input_value) >= 2:
                source_node_id = input_value[0]
                source_output = input_value[1]
                self._trace_input_flow(workflow, input_name, source_node_id, source_output, sampler_id)
        
        print(f"{'='*60}\n")

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
            
        elif node_type == "EmptySD3LatentImage":
            print(f"{indent}🖼️ Width: {node_inputs.get('width', 'N/A')}")
            print(f"{indent}🖼️ Height: {node_inputs.get('height', 'N/A')}")
            print(f"{indent}📦 Batch: {node_inputs.get('batch_size', 'N/A')}")
            
        elif node_type == "LoadImage":
            print(f"{indent}🖼️ Image: {node_inputs.get('image', 'N/A')}")
            
        elif node_type == "VAEEncode":
            print(f"{indent}🔄 VAE: {node_inputs.get('vae', 'N/A')}")
            print(f"{indent}🖼️ Pixels: {node_inputs.get('pixels', 'N/A')}")
            
        elif node_type == "VAEDecode":
            print(f"{indent}🔄 VAE: {node_inputs.get('vae', 'N/A')}")
            print(f"{indent}📦 Samples: {node_inputs.get('samples', 'N/A')}")
            
        elif node_type == "DualCLIPLoader" or node_type == "TripleCLIPLoader":
            print(f"{indent}📖 Type: {node_type}")
            print(f"{indent}📁 Clip: {node_inputs.get('clip_name1', node_inputs.get('clip_name', 'N/A'))}")
            
        elif node_type == "VAELoader":
            print(f"{indent}🔄 VAE: {node_inputs.get('vae_name', 'N/A')}")
            print(f"{indent}📱 Device: {node_inputs.get('device', 'N/A')}")
            
        elif node_type == "SaveImage":
            print(f"{indent}💾 Filename: {node_inputs.get('filename_prefix', 'N/A')}")
            print(f"{indent}📁 Format: {node_inputs.get('format', 'N/A')}")
            print(f"{indent}⭐ Quality: {node_inputs.get('quality', 'N/A')}")
            
        elif node_type == "SaveAudio":
            print(f"{indent}💾 Prefix: {node_inputs.get('filename_prefix', 'N/A')}")
            
        elif node_type == "SaveAudioMP3":
            print(f"{indent}💾 Prefix: {node_inputs.get('filename_prefix', 'N/A')}")
            print(f"{indent}🎵 Quality: {node_inputs.get('quality', 'N/A')}")
            
        elif node_type == "ConditioningZeroOut":
            print(f"{indent}🔄 Conditioning: Zero Out")
            
        elif node_type == "ModelSamplingSD3":
            print(f"{indent}⚙️ Shift: {node_inputs.get('shift', 'N/A')}")
            
        elif node_type == "FluxGuidance":
            print(f"{indent}🎯 Guidance: {node_inputs.get('guidance', 'N/A')}")
            
        elif node_type == "CheckpointLoaderSimple":
            print(f"{indent}📦 Checkpoint: {node_inputs.get('ckpt_name', 'N/A')}")
            
        elif node_type == "EmptyLatentAudio":
            print(f"{indent}🎵 Seconds: {node_inputs.get('seconds', 'N/A')}")
            print(f"{indent}📦 Batch: {node_inputs.get('batch_size', 'N/A')}")
            
        elif node_type == "VAEDecodeAudio":
            print(f"{indent}🔄 VAE: {node_inputs.get('vae', 'N/A')}")
            print(f"{indent}📦 Samples: {node_inputs.get('samples', 'N/A')}")
            
        elif node_type == "UnifiedTTSTextNode":
            print(f"{indent}🎤 Voice: {node_inputs.get('narrator_voice', 'N/A')}")
            print(f"{indent}🌱 Seed: {node_inputs.get('seed', 'N/A')}")
            print(f"{indent}📝 Chunking: {node_inputs.get('enable_chunking', 'N/A')}")
            print(f"{indent}📏 Max Chars: {node_inputs.get('max_chars_per_chunk', 'N/A')}")
            
        elif node_type == "ChatterBoxEngineNode":
            print(f"{indent}🌍 Language: {node_inputs.get('language', 'N/A')}")
            print(f"{indent}📱 Device: {node_inputs.get('device', 'N/A')}")
            print(f"{indent}🎭 Exaggeration: {node_inputs.get('exaggeration', 'N/A')}")
            print(f"{indent}🌡️ Temperature: {node_inputs.get('temperature', 'N/A')}")
            
        elif node_type == "LTXVBaseSampler":
            print(f"{indent}📐 Dimensions: {node_inputs.get('width', 'N/A')}x{node_inputs.get('height', 'N/A')}")
            print(f"{indent}🎬 Frames: {node_inputs.get('num_frames', 'N/A')}")
            print(f"{indent}💪 Strength: {node_inputs.get('strength', 'N/A')}")
            print(f"{indent}🎯 Crop: {node_inputs.get('crop', 'N/A')}")
            
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
            
        elif node_type == "PrimitiveStringMultiline":
            print(f"{indent}📝 Value: {node_inputs.get('value', 'N/A')}")
            
        # Show any other relevant parameters
        for key, value in node_inputs.items():
            if key not in ['model', 'clip', 'vae', 'pixels', 'samples', 'image', 'text', 'lora_name', 'strength_model', 'strength_clip', 'model_name', 'device', 'width', 'height', 'batch_size', 'filename_prefix', 'format', 'quality', 'clip_name1', 'clip_name', 'vae_name', 'narrator_voice', 'seed', 'enable_chunking', 'max_chars_per_chunk', 'language', 'exaggeration', 'temperature', 'num_frames', 'strength', 'crop', 'frame_rate', 'skip_steps_sigma_threshold', 'cfg_star_rescale', 'noise_seed', 'string', 'timestep', 'scale', 'sampler_name', 'loop_count', 'pix_fmt', 'crf', 'value', 'guidance', 'shift', 'ckpt_name', 'seconds']:
                if isinstance(value, (str, int, float, bool)) and len(str(value)) < 50:
                    print(f"{indent}⚙️ {key}: {value}")

    def _trace_model_flow(self, workflow: dict) -> list:
        """Trace the model flow through the workflow."""
        flow = []
        
        # Start from UnetLoaderGGUF
        unet_loader = self._find_node_by_class(workflow, "UnetLoaderGGUF")
        if unet_loader:
            flow.append(f"UnetLoaderGGUF({unet_loader[0]})")
            
            # Follow model connections
            current = unet_loader[0]
            visited = set()
            
            while current and current not in visited:
                visited.add(current)
                node_data = workflow.get(current, {})
                
                if node_data.get("class_type") == "LoraLoader":
                    flow.append(f"LoRA({current})")
                elif node_data.get("class_type") == "KSampler":
                    flow.append(f"KSampler({current})")
                    break
                    
                # Find next node connected to model output
                next_node = None
                for node_id, node_data in workflow.items():
                    if isinstance(node_data, dict) and "inputs" in node_data:
                        for input_name, input_value in node_data["inputs"].items():
                            if isinstance(input_value, list) and len(input_value) >= 2:
                                if input_value[0] == current and input_name == "model":
                                    next_node = node_id
                                    break
                    if next_node:
                        break
                        
                current = next_node
        
        return flow

    def _trace_clip_flow(self, workflow: dict) -> list:
        """Trace the CLIP flow through the workflow."""
        flow = []
        
        # Start from CLIPLoader
        clip_loader = self._find_node_by_class(workflow, ["DualCLIPLoader", "TripleCLIPLoader"])
        if clip_loader:
            flow.append(f"CLIPLoader({clip_loader[0]})")
            
            # Find CLIPTextEncode nodes
            clip_text_nodes = [node_id for node_id, node_data in workflow.items() 
                              if isinstance(node_data, dict) and node_data.get("class_type") == "CLIPTextEncode"]
            for node_id in clip_text_nodes:
                flow.append(f"CLIPTextEncode({node_id})")
        
        return flow

    def _trace_latent_flow(self, workflow: dict) -> list:
        """Trace the latent flow through the workflow."""
        flow = []
        
        # Start from EmptySD3LatentImage or LoadImage
        latent_start = self._find_node_by_class(workflow, "EmptySD3LatentImage")
        if not latent_start:
            latent_start = self._find_node_by_class(workflow, "LoadImage")
        
        if latent_start:
            node_type = workflow[latent_start[0]].get("class_type", "")
            flow.append(f"{node_type}({latent_start[0]})")
            
            # Follow latent connections
            current = latent_start[0]
            visited = set()
            
            while current and current not in visited:
                visited.add(current)
                node_data = workflow.get(current, {})
                
                if node_data.get("class_type") == "VAEEncode":
                    flow.append(f"VAEEncode({current})")
                elif node_data.get("class_type") == "KSampler":
                    flow.append(f"KSampler({current})")
                    break
                elif node_data.get("class_type") == "VAEDecode":
                    flow.append(f"VAEDecode({current})")
                    break
                    
                # Find next node connected to latent output
                next_node = None
                for node_id, node_data in workflow.items():
                    if isinstance(node_data, dict) and "inputs" in node_data:
                        for input_name, input_value in node_data["inputs"].items():
                            if isinstance(input_value, list) and len(input_value) >= 2:
                                if input_value[0] == current and input_name in ["latent", "samples"]:
                                    next_node = node_id
                                    break
                    if next_node:
                        break
                        
                current = next_node
        
        return flow

    def _trace_image_flow(self, workflow: dict) -> list:
        """Trace the image flow through the workflow."""
        flow = []
        
        # Start from VAEDecode
        vae_decode = self._find_node_by_class(workflow, "VAEDecode")
        if vae_decode:
            flow.append(f"VAEDecode({vae_decode[0]})")
            
            # Find SaveImage
            save_image = self._find_node_by_class(workflow, "SaveImage")
            if save_image:
                flow.append(f"SaveImage({save_image[0]})")
        
        return flow

    def _find_node_by_class(self, workflow: dict, class_types: str | list[str]) -> list | None:
        """Find a node by its class type and return its connection."""
        if isinstance(class_types, str):
            class_types = [class_types]
        
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") in class_types:
                return [node_id, 0]
        return None
    
    def _update_node_connections(self, workflow: dict, class_types: str | list[str], input_key: str, value) -> None:
        """Update specific input connections for nodes matching class types."""
        if isinstance(class_types, str):
            class_types = [class_types]
        
        for node_id, node in workflow.items():
            if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
                if node.get("class_type") in class_types and input_key in node["inputs"]:
                    node["inputs"][input_key] = value

    def _create_image_processing_nodes(self, workflow: dict, all_images: dict, start_node_id: int) -> list[str]:
        """Create all image processing nodes with stitching and return reference latent node IDs."""
        next_node_id = start_node_id
        item_names = list(all_images.keys())
        
        # Create LoadImage nodes
        load_nodes = []
        for item_name, img_path in all_images.items():
            node_id = str(next_node_id)
            workflow[node_id] = {
                "inputs": {"image": os.path.basename(img_path)},
                "class_type": "LoadImage",
                "_meta": {"title": f"Load {item_name}"}
            }
            load_nodes.append(node_id)
            next_node_id += 1

        # Create Scale nodes (connect directly to LoadImage nodes since resizing is done during compression)
        scale_nodes = []
        for i, load_node in enumerate(load_nodes):
            node_id = str(next_node_id)
            workflow[node_id] = {
                "inputs": {"image": [load_node, 0]},
                "class_type": "FluxKontextImageScale",
                "_meta": {"title": f"Scale {item_names[i]}"}
            }
            scale_nodes.append(node_id)
            next_node_id += 1

        # Group images for stitching
        stitch_groups = []
        for i in range(0, len(scale_nodes), IMAGE_STITCH_COUNT):
            group = scale_nodes[i:i + IMAGE_STITCH_COUNT]
            stitch_groups.append(group)

        # Create ImageStitch nodes for each group
        stitched_nodes = []
        for group_idx, group in enumerate(stitch_groups):
            if len(group) == 1:
                # Single image, no stitching needed
                stitched_nodes.append(group[0])
            elif len(group) == 2:
                # Two images, single stitch
                node_id = str(next_node_id)
                workflow[node_id] = {
                    "inputs": {
                        "image1": [group[0], 0],
                        "image2": [group[1], 0],
                        "direction": "right",
                        "match_image_size": True,
                        "spacing_width": 0,
                        "spacing_color": "white"
                    },
                    "class_type": "ImageStitch",
                    "_meta": {"title": f"Stitch Group {group_idx + 1}"}
                }
                stitched_nodes.append(node_id)
                next_node_id += 1
            else:
                # Three or more images, chain stitches
                current_stitch = group[0]
                for img_idx in range(1, len(group)):
                    node_id = str(next_node_id)
                    workflow[node_id] = {
                        "inputs": {
                            "image1": [current_stitch, 0],
                            "image2": [group[img_idx], 0],
                            "direction": "right",
                            "match_image_size": True,
                            "spacing_width": 0,
                            "spacing_color": "white"
                        },
                        "class_type": "ImageStitch",
                        "_meta": {"title": f"Stitch Group {group_idx + 1} Step {img_idx}"}
                    }
                    current_stitch = node_id
                    next_node_id += 1
                stitched_nodes.append(current_stitch)

        # Create SaveImage nodes for stitched images (for verification)
        save_nodes = []
        for i, stitched_node in enumerate(stitched_nodes):
            node_id = str(next_node_id)
            workflow[node_id] = {
                "inputs": {
                    "filename_prefix": f"stitched_group_{i + 1}",
                    "images": [stitched_node, 0]
                },
                "class_type": "SaveImage",
                "_meta": {"title": f"Save Stitched Group {i + 1}"}
            }
            save_nodes.append(node_id)
            next_node_id += 1

        # Create Encode nodes for stitched images
        encode_nodes = []
        for i, stitched_node in enumerate(stitched_nodes):
            node_id = str(next_node_id)
            workflow[node_id] = {
                "inputs": {"pixels": [stitched_node, 0], "vae": ["11", 0]},
                "class_type": "VAEEncode",
                "_meta": {"title": f"Encode Stitched Group {i + 1}"}
            }
            encode_nodes.append(node_id)
            next_node_id += 1

        # Create cascading ReferenceLatent nodes
        ref_latent_nodes = []
        previous_conditioning = ["33", 0]
        for i, encode_node in enumerate(encode_nodes):
            node_id = str(next_node_id)
            workflow[node_id] = {
                "inputs": {"conditioning": previous_conditioning, "latent": [encode_node, 0]},
                "class_type": "ReferenceLatent",
                "_meta": {"title": f"Reference Stitched Group {i + 1}"}
            }
            ref_latent_nodes.append(node_id)
            previous_conditioning = [node_id, 0]
            next_node_id += 1

        return ref_latent_nodes

    def _build_dynamic_workflow(self, scene_id: str, scene_description: str, character_names: list[str], location_ids: list[str], master_prompt: str, characters_data: dict[str, str], locations_data: dict[str, str] = None) -> dict:
        """Build a dynamic workflow with N character images."""
        # Start with base workflow
        workflow = self._load_base_workflow()
        
        if not workflow:
            return {}

        # For serial mode, don't apply LoRAs here - they'll be applied later
        if USE_LORA and LORA_MODE == "serial":
            print(f"Building workflow for serial LoRA mode: {scene_id}")
        else:
            # Apply LoRAs for chained mode
            if USE_LORA:
                self._apply_loras(workflow)
                print("Chained LoRA mode enabled in workflow")
            else:
                # Remove all LoRA nodes if they exist
                self._remove_all_lora_nodes(workflow)
                print("LoRA disabled in workflow")
        
        # Handle chained mode workflow setup
        # Handle different character and location modes
        all_images = {}
        
        # Copy character images if in IMAGE or IMAGE_TEXT mode (skip in NONE mode)
        if self.character_mode in ["IMAGE", "IMAGE_TEXT"]:
            character_images = self._copy_character_images_to_comfyui(character_names)
            all_images.update(character_images)
            print(f"Added {len(character_images)} character images to stitching process")
            if not character_images and self.character_mode == "IMAGE":
                print("ERROR: No character images copied to ComfyUI!")
                return {}
        
        # Copy location images if in IMAGE or IMAGE_TEXT mode (skip in NONE mode)
        if self.location_mode in ["IMAGE", "IMAGE_TEXT"]:
            location_images = self._copy_location_images_to_comfyui(location_ids)
            all_images.update(location_images)
            print(f"Added {len(location_images)} location images to stitching process")
            if not location_images and self.location_mode == "IMAGE":
                print("ERROR: No location images copied to ComfyUI!")
                return {}
        
        print(f"Character mode: {self.character_mode}, Location mode: {self.location_mode}, Total images: {len(all_images)}")
        
        # Calculate stitching groups for logging
        if all_images:
            total_images = len(all_images)
            groups_needed = (total_images + IMAGE_STITCH_COUNT - 1) // IMAGE_STITCH_COUNT
            group_sizes = [IMAGE_STITCH_COUNT] * (groups_needed - 1)
            if total_images % IMAGE_STITCH_COUNT != 0:
                group_sizes.append(total_images % IMAGE_STITCH_COUNT)
            print(f"Image stitching: {total_images} images → {groups_needed} groups {group_sizes}")
        
        next_node_id = 100

        # Process images if available
        ref_latent_nodes = []
        if all_images:
            ref_latent_nodes = self._create_image_processing_nodes(workflow, all_images, next_node_id)

        # Set conditioning based on available reference latents
        if ref_latent_nodes:
            workflow["32"]["inputs"]["conditioning"] = [ref_latent_nodes[-1], 0]
            print(f"Using {'single' if len(ref_latent_nodes) == 1 else 'cascaded'} reference conditioning")
        else:
            workflow["32"]["inputs"]["conditioning"] = ["33", 0]
            print("Using text conditioning only")

        text_prompt = f"{master_prompt}"

        # Replace location references if location data is available and in appropriate mode (skip in NONE mode)
        processed_scene_description = scene_description

        processed_scene_description = self._replace_location_references(scene_description, location_ids, locations_data)
        
        processed_scene_description = self._replace_character_references(processed_scene_description, character_names, characters_data)

        text_prompt += f"\n{processed_scene_description}\n"
        
        workflow["33"]["inputs"]["text"] = text_prompt
        workflow["21"]["inputs"]["filename_prefix"] = scene_id
        
        # Set resolution parameters and handle latent input mode
        # For chained mode: Apply latent/image logic based on LATENT_MODE
        # For serial mode: This method is called for each LoRA individually
        if LATENT_MODE == "IMAGE":
            # Get location-specific latent image path based on scene description
            latent_image_path = self._get_location_latent_image_path(scene_description)
            
            # For chained mode: Replace EmptySD3LatentImage with LoadImage + VAEEncode
            # For serial mode: This will be handled individually in _generate_character_image_serial
            if LORA_MODE == "chained" or not USE_LORA:
                self._replace_latent_with_image_input(workflow, "19", latent_image_path, LATENT_DENOISING_STRENGTH)
                print(f"Using image input mode with file: {latent_image_path}")
            else:
                # Serial mode: Just set dimensions, individual LoRA handling will replace this
                workflow["19"]["inputs"]["width"] = IMAGE_WIDTH
                workflow["19"]["inputs"]["height"] = IMAGE_HEIGHT
                print(f"Serial mode: Set latent dimensions, individual LoRA handling will replace with image input")
        else:
            # Normal latent mode - set dimensions
            workflow["19"]["inputs"]["width"] = IMAGE_WIDTH
            workflow["19"]["inputs"]["height"] = IMAGE_HEIGHT
            print(f"Using latent mode with dimensions: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        
        # Handle negative prompt
        if USE_NEGATIVE_PROMPT:
            # Create a new CLIPTextEncode node for negative prompt
            negative_node_id = "35"
            
            # Find the final CLIP output from LoRAs or use base CLIP
            final_clip_output = ["10", 0]  # Base clip
            if USE_LORA and LORA_MODE == "chained":
                enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
                if enabled_loras:
                    final_clip_output = [f"lora_{len(enabled_loras)}", 1]
            
            workflow[negative_node_id] = {
                "inputs": {
                    "text": NEGATIVE_PROMPT,
                    "clip": final_clip_output
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Negative Prompt)"
                }
            }
            # Connect negative prompt directly to sampler's negative conditioning
            workflow["16"]["inputs"]["negative"] = [negative_node_id, 0]
            # Remove ConditioningZeroOut from the workflow when using negative prompt
            if "34" in workflow:
                del workflow["34"]
            print(f"Negative prompt enabled: {NEGATIVE_PROMPT}")
        else:
            # Keep ConditioningZeroOut for empty negative (it's already connected in base workflow)
            print("Negative prompt disabled - using ConditioningZeroOut")
        
        # Set seed for all modes (serial, chained, and no-LoRA)
        seed = self._get_seed()
        self._update_node_connections(workflow, "KSampler", "seed", seed)
        print(f"Seed set to: {seed}")
        
        print("\n\n\n")
        print(f"Text prompt: {text_prompt}")
        return workflow

    def _replace_latent_with_image_input(self, workflow: dict, latent_node_id: str, image_path: str, denoising_strength: float = None) -> None:
        """Replace EmptySD3LatentImage with LoadImage + VAEEncode for image input."""
        try:
            # Copy the image to ComfyUI input folder
            image_filename = os.path.basename(image_path)
            comfyui_input_path = os.path.join("../../ComfyUI/input", image_filename)
            if os.path.exists(image_path):
                shutil.copy2(image_path, comfyui_input_path)
                print(f"  Copied image to ComfyUI input: {image_filename}")
            else:
                print(f"WARNING: Image file not found: {image_path}")
            
            # Find the next available node ID
            max_id = max(int(k) for k in workflow.keys() if k.isdigit())
            load_image_node_id = str(max_id + 1)
            encode_node_id = str(max_id + 2)
            
            # Create LoadImage node
            workflow[load_image_node_id] = {
                "inputs": {"image": image_filename},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Latent Image"}
            }
            
            # Create VAEEncode node to convert image to latent
            workflow[encode_node_id] = {
                "inputs": {
                    "pixels": [load_image_node_id, 0],
                    "vae": ["11", 0]  # Use existing VAE
                },
                "class_type": "VAEEncode",
                "_meta": {"title": "VAE Encode (Latent Image)"}
            }
            
            # Find KSampler and update its latent_image input and denoising strength
            for sampler_id, sampler_node in workflow.items():
                if isinstance(sampler_node, dict) and sampler_node.get("class_type") == "KSampler":
                    if "latent_image" in sampler_node.get("inputs", {}):
                        sampler_node["inputs"]["latent_image"] = [encode_node_id, 0]
                        # Use provided denoising strength or fall back to LATENT_DENOISING_STRENGTH
                        if denoising_strength is not None:
                            sampler_node["inputs"]["denoise"] = denoising_strength
                        else:
                            sampler_node["inputs"]["denoise"] = LATENT_DENOISING_STRENGTH
                        break
            
            # Remove the original EmptySD3LatentImage node
            del workflow[latent_node_id]
            
            print(f"  Replaced EmptySD3LatentImage with LoadImage + VAEEncode")
            print(f"  LoadImage node: {load_image_node_id}, VAEEncode node: {encode_node_id}")
            print(f"  Denoising strength set to: {denoising_strength if denoising_strength is not None else LATENT_DENOISING_STRENGTH}")
            
        except Exception as e:
            print(f"WARNING: Failed to replace latent with image input: {e}")

    def _replace_latent_with_previous_output(self, workflow: dict, image_path: str, denoising_strength: float = None) -> None:
        """Replace EmptySD3LatentImage with LoadImage + VAEEncode for previous LoRA output."""
        try:
            # Copy the previous LoRA output to ComfyUI input folder
            image_filename = os.path.basename(image_path)
            comfyui_input_path = os.path.join("../../ComfyUI/input", image_filename)
            if os.path.exists(image_path):
                shutil.copy2(image_path, comfyui_input_path)
                print(f"  Copied previous LoRA output to ComfyUI input: {image_filename}")
            else:
                print(f"WARNING: Previous LoRA output not found: {image_path}")
            
            # Find the next available node ID
            max_id = max(int(k) for k in workflow.keys() if k.isdigit())
            load_image_node_id = str(max_id + 1)
            encode_node_id = str(max_id + 2)
            
            # Create LoadImage node
            workflow[load_image_node_id] = {
                "inputs": {"image": image_filename},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Previous LoRA Output"}
            }
            
            # Create VAEEncode node to convert image to latent
            workflow[encode_node_id] = {
                "inputs": {
                    "pixels": [load_image_node_id, 0],
                    "vae": ["11", 0]  # Use existing VAE
                },
                "class_type": "VAEEncode",
                "_meta": {"title": "VAE Encode (Previous LoRA Output)"}
            }
            
            # Find KSampler and update its latent_image input and denoising strength
            for sampler_id, sampler_node in workflow.items():
                if isinstance(sampler_node, dict) and sampler_node.get("class_type") == "KSampler":
                    if "latent_image" in sampler_node.get("inputs", {}):
                        sampler_node["inputs"]["latent_image"] = [encode_node_id, 0]
                        # Use provided denoising strength or keep existing setting
                        if denoising_strength is not None:
                            sampler_node["inputs"]["denoise"] = denoising_strength
                        break
            
            # Remove the original EmptySD3LatentImage node
            if "19" in workflow:
                del workflow["19"]
            
            print(f"  Replaced EmptySD3LatentImage with LoadImage + VAEEncode")
            print(f"  LoadImage node: {load_image_node_id}, VAEEncode node: {encode_node_id}")
            
        except Exception as e:
            print(f"WARNING: Failed to replace latent with previous output: {e}")

    def _generate_scene_image_serial(self, scene_id: str, scene_description: str, character_names: list[str], location_ids: list[str], master_prompt: str, characters_data: dict[str, str], locations_data: dict[str, str] = None, resumable_state=None) -> str | None:
        """Generate scene image using serial LoRA mode with intermediate storage."""
        try:
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_scene_complete(scene_id):
                cached_result = resumable_state.get_scene_result(scene_id)
                if cached_result and os.path.exists(cached_result.get('path', '')):
                    print(f"Using cached scene image: {scene_id}")
                    return cached_result['path']
                elif cached_result:
                    print(f"Cached file missing, regenerating: {scene_id}")
            
            print(f"Generating scene: {scene_id} with characters: {', '.join(character_names)} (Serial LoRA mode)")
            
            enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
            if not enabled_loras:
                print("ERROR: No enabled LoRAs found for serial mode")
                return None
            
            # Clean scene ID for filenames (preserve dots for version numbers like 1.1)
            clean_scene_id = re.sub(r'[^\w\s.-]', '', scene_id).strip()
            clean_scene_id = re.sub(r'[-\s]+', '_', clean_scene_id)
            
            # Check for existing LoRA progress
            lora_progress_key = f"{scene_id}_lora_progress"
            completed_loras = []
            current_image_path = None
            intermediate_paths = []
            
            if resumable_state:
                lora_progress = resumable_state.state.get("lora_progress", {}).get(lora_progress_key, {})
                completed_loras = lora_progress.get("completed_loras", [])
                current_image_path = lora_progress.get("current_image_path")
                intermediate_paths = lora_progress.get("intermediate_paths", [])
                saved_lora_configs = lora_progress.get("lora_configs", {})
                
                if completed_loras:
                    print(f"Resuming from LoRA {len(completed_loras) + 1}/{len(enabled_loras)}")
                    if current_image_path and os.path.exists(current_image_path):
                        print(f"Using previous LoRA output: {current_image_path}")
                    else:
                        print("Previous LoRA output missing, restarting from LoRA 1")
                        completed_loras = []
                        current_image_path = None
                        intermediate_paths = []
                        saved_lora_configs = {}  # Clear saved configs when restarting
                        
                        # Update resumable state to reflect LoRA progress invalidation
                        lora_progress = {
                            "completed_loras": [],
                            "current_image_path": None,
                            "intermediate_paths": [],
                            "lora_configs": {}
                        }
                        resumable_state.state.setdefault("lora_progress", {})[lora_progress_key] = lora_progress
                        resumable_state._save_state()
                        print("  Updated resumable state: LoRA progress invalidated")
            
            # Process each LoRA in sequence, using previous output as input
            for i, lora_config in enumerate(enabled_loras):
                lora_name = lora_config['name']
                lora_clean_name = re.sub(r'[^\w\s-]', '', lora_name).strip()
                lora_clean_name = re.sub(r'[-\s]+', '_', lora_clean_name)
                
                # Create unique identifier for this LoRA (index + name)
                lora_unique_id = f"{i}_{lora_name}"
                
                # Skip if this LoRA was already completed
                if lora_unique_id in completed_loras:
                    print(f"Skipping completed LoRA {i + 1}/{len(enabled_loras)}: {lora_name}")
                    continue
                
                print(f"\nProcessing LoRA {i + 1}/{len(enabled_loras)}: {lora_name}")
                
                # Check if this LoRA should use only intermediate result (no character images)
                # Use saved config if resuming, otherwise use current config
                if saved_lora_configs and lora_unique_id in saved_lora_configs:
                    saved_config = saved_lora_configs[lora_unique_id]
                    use_only_intermediate = saved_config.get("use_only_intermediate", False)
                    print(f"  Using saved LoRA configuration for {lora_name}")
                else:
                    use_only_intermediate = lora_config.get("use_only_intermediate", False)
                    print(f"  Using current LoRA configuration for {lora_name}")
                
                # Determine character and location lists for this LoRA
                if use_only_intermediate and i > 0:
                    # This LoRA should only use intermediate result, disable character and location images
                    print(f"  LoRA {i + 1} configured to use only intermediate result (no character or location images)")
                    lora_character_names = []
                    lora_location_ids = []
                else:
                    # Normal workflow with character and location images based on modes (skip in NONE mode)
                    lora_character_names = character_names
                    lora_location_ids = location_ids
                
                # Build the complete workflow with all prompting logic
                workflow = self._build_dynamic_workflow(scene_id, scene_description, lora_character_names, lora_location_ids, master_prompt, characters_data, locations_data)
                
                if not workflow:
                    print(f"ERROR: Failed to build workflow for LoRA {i + 1}")
                    continue
                
                # Append dots to prompt based on LoRA position (LoRA 0 = no dots, LoRA 1 = 1 dot, etc.)
                if i > 0:  # Skip first LoRA (index 0)
                    dots = "." * i
                    current_prompt = workflow["33"]["inputs"]["text"]
                    workflow["33"]["inputs"]["text"] = current_prompt + dots
                    print(f"  Added {i} dots to prompt: {dots}")
                
                # Apply only this LoRA to the workflow (after building the complete workflow)
                self._apply_single_lora(workflow, lora_config, i + 1)
                
                # Set LoRA-specific sampling steps, seed, and denoising
                steps = lora_config.get("steps", SAMPLING_STEPS)
                denoising_strength = lora_config.get("denoising_strength", 1.0)
                seed = self._get_seed()
                self._update_node_connections(workflow, "KSampler", "steps", steps)
                self._update_node_connections(workflow, "KSampler", "seed", seed)
                print(f"  Seed set to: {seed}")
                
                # Handle input for this LoRA based on serial mode logic
                if i == 0:
                    # First LoRA: Use latent/image mode based on LATENT_MODE setting
                    if LATENT_MODE == "IMAGE":
                        # Get location-specific latent image path based on scene description
                        latent_image_path = self._get_location_latent_image_path(scene_description)
                        
                        # Replace EmptySD3LatentImage with image input + LATENT_DENOISING_STRENGTH
                        self._replace_latent_with_image_input(workflow, "19", latent_image_path, LATENT_DENOISING_STRENGTH)
                        # Apply LATENT_DENOISING_STRENGTH to KSampler for first LoRA in IMAGE mode
                        self._update_node_connections(workflow, "KSampler", "denoise", LATENT_DENOISING_STRENGTH)
                        self._update_node_connections(workflow, "KSampler", "seed", seed)
                        denoising_strength = LATENT_DENOISING_STRENGTH
                        print(f"  Using image input mode for first LoRA with file: {latent_image_path}")
                        print(f"  Using LATENT_DENOISING_STRENGTH: {LATENT_DENOISING_STRENGTH}")
                    else:
                        # Normal latent mode - set dimensions and use LoRA's denoising strength
                        workflow["19"]["inputs"]["width"] = IMAGE_WIDTH
                        workflow["19"]["inputs"]["height"] = IMAGE_HEIGHT
                        # Apply LoRA's denoising_strength to KSampler for first LoRA in LATENT mode
                        self._update_node_connections(workflow, "KSampler", "denoise", denoising_strength)
                        print(f"  Using latent mode with dimensions: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
                        print(f"  Using LoRA denoising_strength: {denoising_strength}")
                else:
                    # Subsequent LoRAs: Use previous LoRA output as input
                    if current_image_path:
                        self._replace_latent_with_previous_output(workflow, current_image_path, denoising_strength)
                        # Apply LoRA's denoising_strength to KSampler for subsequent LoRAs
                        self._update_node_connections(workflow, "KSampler", "denoise", denoising_strength)
                        print(f"  Using previous LoRA output as latent input")
                        print(f"  Using LoRA denoising_strength: {denoising_strength}")
                    else:
                        print(f"  ERROR: No previous LoRA output available for LoRA {i + 1}")
                        continue
                
                # If this LoRA uses only intermediate result, ensure no character images are processed
                if use_only_intermediate:
                    print(f"  Disabled character image processing for LoRA {i + 1}")
                    # The workflow was already built without character images above
                
                # Generate filename for this LoRA step
                lora_clean_name = re.sub(r'[^\w\s-]', '', lora_config['name']).strip()
                lora_clean_name = re.sub(r'[-\s]+', '_', lora_clean_name)
                lora_filename = f"{clean_scene_id}.{lora_clean_name}"
                workflow["21"]["inputs"]["filename_prefix"] = lora_filename
                
                print(f"  Steps: {steps}, Denoising: {denoising_strength}")
                
                # Print workflow summary before sending
                self._print_workflow_summary(workflow, f"LoRA {i + 1}: {lora_name}")
                
                # Submit workflow to ComfyUI
                resp = requests.post(f"{self.comfyui_url}prompt", json={"prompt": workflow}, timeout=120)
                if resp.status_code != 200:
                    print(f"ERROR: ComfyUI API error for LoRA {i + 1}: {resp.status_code} {resp.text}")
                    continue
                    
                prompt_id = resp.json().get("prompt_id")
                if not prompt_id:
                    print(f"ERROR: No prompt ID returned for LoRA {i + 1}")
                    continue

                # Wait for completion
                print(f"  Waiting for LoRA {i + 1} generation to complete...")
                while True:
                    h = requests.get(f"{self.comfyui_url}history/{prompt_id}")
                    if h.status_code == 200:
                        data = h.json()
                        if prompt_id in data:
                            status = data[prompt_id].get("status", {})
                            if status.get("exec_info", {}).get("queue_remaining", 0) == 0:
                                time.sleep(2)  # Give it a moment to finish
                                break
                    time.sleep(2)

                # Find the generated image
                generated_image = self._find_newest_output_with_prefix(lora_filename)
                if not generated_image:
                    print(f"ERROR: Could not find generated image for LoRA {i + 1}")
                    print(f"  Looking for files with prefix: {lora_filename}")
                    # List files in output directory for debugging
                    if os.path.exists(self.comfyui_output_folder):
                        print(f"  Available files in {self.comfyui_output_folder}:")
                        for root, dirs, files in os.walk(self.comfyui_output_folder):
                            for file in files:
                                if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                    print(f"    {file}")
                    continue
                
                # Save result to lora folder (save final result from each LoRA)
                lora_final_path = os.path.join(self.intermediate_output_dir, f"{clean_scene_id}.{lora_clean_name}.png")
                shutil.copy2(generated_image, lora_final_path)
                intermediate_paths.append(lora_final_path)
                print(f"  Saved LoRA result: {lora_final_path}")
                
                # Use this output as input for next LoRA
                current_image_path = generated_image
                print(f"  LoRA {i + 1} completed successfully")
                
                # Save progress after each LoRA completion
                if resumable_state:
                    completed_loras.append(lora_unique_id)
                    lora_progress = {
                        "completed_loras": completed_loras,
                        "current_image_path": current_image_path,
                        "intermediate_paths": intermediate_paths,
                        "lora_configs": {f"{i}_{lora["name"]}": lora for i, lora in enumerate(enabled_loras)}  # Save LoRA configs for resuming with unique IDs
                    }
                    resumable_state.state.setdefault("lora_progress", {})[lora_progress_key] = lora_progress
                    resumable_state._save_state()
                    print(f"  Saved LoRA progress: {len(completed_loras)}/{len(enabled_loras)} completed")
            
            if not current_image_path:
                print(f"ERROR: No successful LoRA generations for {scene_id}")
                return None
            
            # Copy final result to output directory
            final_path = os.path.join(self.final_output_dir, f"{scene_id}.png")
            shutil.copy2(current_image_path, final_path)
            print(f"Saved: {final_path}")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                result = {
                    'path': final_path,
                    'scene_id': scene_id,
                    'scene_description': scene_description,
                    'character_names': character_names,
                    'intermediate_paths': intermediate_paths
                }
                resumable_state.set_scene_result(scene_id, result)
                
                # Keep LoRA progress for completed scene (not cleaning up)
                print(f"  Preserved LoRA progress for completed scene")
            
            return final_path

        except Exception as e:
            print(f"ERROR: Failed to generate image for {scene_id}: {e}")
            return None

    def _apply_single_lora(self, workflow: dict, lora_config: dict, lora_index: int) -> None:
        """Apply a single LoRA to the workflow."""
        lora_node_id = f"lora_{lora_index}"
        
        # Get initial model and clip connections
        model_input = ["41", 0]  # Base model node
        clip_input = ["10", 0]   # Base clip node
        
        # Create LoRA node inputs
        lora_inputs = {
            "lora_name": lora_config["name"],
            "model": model_input,
            "clip": clip_input
        }
        
        # Apply strength settings with bypass options
        if lora_config.get("bypass_model", False):
            lora_inputs["strength_model"] = 0.0
            print(f"  Model bypassed")
        else:
            lora_inputs["strength_model"] = lora_config.get("strength_model", 1.0)
            print(f"  Model strength: {lora_inputs['strength_model']}")
        
        if lora_config.get("bypass_clip", False):
            lora_inputs["strength_clip"] = 0.0
            print(f"  CLIP bypassed")
        else:
            lora_inputs["strength_clip"] = lora_config.get("strength_clip", 1.0)
            print(f"  CLIP strength: {lora_inputs['strength_clip']}")
        
        # Create LoRA node
        workflow[lora_node_id] = {
            "inputs": lora_inputs,
            "class_type": "LoraLoader",
            "_meta": {"title": f"Load LoRA {lora_index}: {lora_config['name']}"}
        }
        
        # Connect LoRA outputs to workflow nodes
        self._update_node_connections(workflow, "KSampler", "model", [lora_node_id, 0])
        self._update_node_connections(workflow, ["CLIPTextEncode", "CLIP Text Encode (Prompt)"], "clip", [lora_node_id, 1])


    def _update_node_connections(self, workflow: dict, class_types: str | list[str], input_key: str, value) -> None:
        """Update specific input connections for nodes matching class types."""
        if isinstance(class_types, str):
            class_types = [class_types]
        
        for node_id, node in workflow.items():
            if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
                if node.get("class_type") in class_types and input_key in node["inputs"]:
                    node["inputs"][input_key] = value

    def _generate_scene_image(self, scene_id: str, scene_description: str, character_names: list[str], location_ids: list[str], master_prompt: str, characters_data: dict[str, str], locations_data: dict[str, str] = None, resumable_state=None) -> str | None:
        """Generate a single scene image using ComfyUI."""
        try:
            # Use serial LoRA mode if enabled
            if USE_LORA and LORA_MODE == "serial":
                return self._generate_scene_image_serial(scene_id, scene_description, character_names, location_ids, master_prompt, characters_data, locations_data, resumable_state)
            
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_scene_complete(scene_id):
                cached_result = resumable_state.get_scene_result(scene_id)
                if cached_result and os.path.exists(cached_result.get('path', '')):
                    print(f"Using cached scene image: {scene_id}")
                    return cached_result['path']
                elif cached_result:
                    print(f"Cached file missing, regenerating: {scene_id}")
            
            print(f"Generating scene: {scene_id} with characters: {', '.join(character_names)}")
            workflow = self._build_dynamic_workflow(scene_id, scene_description, character_names, location_ids, master_prompt, characters_data, locations_data)
            if not workflow:
                return None

            # Print workflow summary
            self._print_workflow_summary(workflow, f"Scene: {scene_id}")
            
            # Print prompt before sending
            print(f"\n=== PROMPT FOR SCENE: {scene_id} ===")
            # Get the text prompt from the workflow
            text_prompt = workflow.get("33", {}).get("inputs", {}).get("text", "No text prompt found")
            print(f"Text prompt: {text_prompt}")
            print(f"Workflow nodes: {len(workflow)} nodes")
            print("=" * 50)

            # Submit to ComfyUI
            resp = requests.post(f"{self.comfyui_url}prompt", json={"prompt": workflow}, timeout=120)
            if resp.status_code != 200:
                print(f"ERROR: ComfyUI API error: {resp.status_code} {resp.text}")
                return None
                
            prompt_id = resp.json().get("prompt_id")
            if not prompt_id:
                print("ERROR: No prompt ID returned from ComfyUI")
                return None

            # Wait for completion
            print(f"Waiting for completion (prompt_id: {prompt_id})...")
            while True:
                h = requests.get(f"{self.comfyui_url}history/{prompt_id}")
                if h.status_code == 200 and prompt_id in h.json():
                    status = h.json()[prompt_id].get("status", {})
                    if status.get("exec_info", {}).get("queue_remaining", 0) == 0:
                        time.sleep(2)
                        break
                time.sleep(2)

            # Find and copy generated image
            generated_image = self._find_newest_output_with_prefix(scene_id)
            if not generated_image:
                print(f"ERROR: Could not find generated image for {scene_id}")
                return None

            final_path = os.path.join(self.final_output_dir, f"{scene_id}.png")
            shutil.copy2(generated_image, final_path)
            print(f"Saved: {final_path}")
            
            # Handle intermediate files for serial LoRA mode
            if USE_LORA and LORA_MODE == "serial":
                self._handle_serial_lora_intermediate_files(scene_id)
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                result = {
                    'path': final_path,
                    'scene_id': scene_id,
                    'scene_description': scene_description,
                    'character_names': character_names
                }
                resumable_state.set_scene_result(scene_id, result)
            
            return final_path

        except Exception as e:
            print(f"ERROR: Failed to generate image for {scene_id}: {e}")
            return None
    
    def _handle_serial_lora_intermediate_files(self, scene_id: str) -> None:
        """Handle intermediate files from serial LoRA processing."""
        if not os.path.isdir(self.comfyui_output_folder):
            return
        
        # Find all intermediate LoRA files
        intermediate_files = []
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        
        for root, _, files in os.walk(self.comfyui_output_folder):
            for name in files:
                if name.startswith(f"{scene_id}.") and any(name.lower().endswith(ext) for ext in exts):
                    full_path = os.path.join(root, name)
                    intermediate_files.append(full_path)
        
        if intermediate_files:
            print(f"Found {len(intermediate_files)} intermediate LoRA files:")
            for file_path in intermediate_files:
                # Copy to final output directory for reference
                filename = os.path.basename(file_path)
                dest_path = os.path.join(self.final_output_dir, filename)
                shutil.copy2(file_path, dest_path)
                print(f"  Copied intermediate: {filename}")
                
                # Optionally clean up intermediate files from ComfyUI output
                # Uncomment the next line if you want to clean up intermediate files
                # os.remove(file_path)

    def _find_newest_output_with_prefix(self, prefix: str) -> str | None:
        """Find the newest generated image with the given prefix."""
        if not os.path.isdir(self.comfyui_output_folder):
            return None
        latest, latest_mtime = None, -1.0
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        for root, _, files in os.walk(self.comfyui_output_folder):
            for name in files:
                if name.startswith(prefix) and any(name.lower().endswith(ext) for ext in exts):
                    full_path = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(full_path)
                        if mtime > latest_mtime:
                            latest_mtime, latest = mtime, full_path
                    except OSError:
                        continue
        return latest

    def _get_completed_scenes(self) -> set[str]:
        """Get scene IDs that have already been generated."""
        if not os.path.exists(self.final_output_dir):
            return set()
        return {f[:-4] for f in os.listdir(self.final_output_dir) if f.endswith('.png')}

    def generate_all_scenes(self, force_regenerate: bool = False, resumable_state=None) -> dict[str, str]:
        """Generate images for all scenes."""
        scenes = self._read_scene_data()
        characters = self._read_character_data()

        locations = self._read_location_data()
        master_prompt = self._get_master_prompt()
        
        if not scenes or not master_prompt:
            print("ERROR: Missing scene data or master prompt")
            return {}

        print(f"Character mode: {self.character_mode}")
        if self.character_mode == "NONE":
            print("Character processing disabled (NONE mode)")
        
        print(f"Location mode: {self.location_mode}")
        if self.location_mode == "NONE":
            print("Location processing disabled (NONE mode)")

        # Use resumable state if available, otherwise fall back to file-based checking
        if resumable_state:
            print(f"Validating cache against output/scene directory: {self.final_output_dir}")
            
            # First, sync with output/scene directory to detect any manually added files
            synced_count = resumable_state.sync_with_output_directory(self.final_output_dir)
            if synced_count > 0:
                print(f"Sync completed: {synced_count} scenes auto-detected from output/scene directory")
            
            # Then run precheck to validate file existence and clean up invalid entries
            cleaned_count = resumable_state.validate_and_cleanup_results(self.final_output_dir)
            if cleaned_count > 0:
                print(f"Precheck completed: {cleaned_count} invalid entries removed from checkpoint")
            
            completed_scenes = set()
            for scene_id in scenes.keys():
                if resumable_state.is_scene_complete(scene_id):
                    completed_scenes.add(scene_id)
        else:
            completed_scenes = self._get_completed_scenes()
        
        if not force_regenerate and completed_scenes:
            print(f"Found {len(completed_scenes)} completed scenes: {sorted(completed_scenes)}")

        scenes_to_process = {sid: desc for sid, desc in scenes.items() 
                           if force_regenerate or sid not in completed_scenes}

        if not scenes_to_process:
            print("All scenes already generated!")
            return {}

        print(f"Processing {len(scenes_to_process)} scenes, skipped {len(completed_scenes)}")
        
        # Calculate total words for ETA calculations
        self.total_scene_words = sum(len(description.split()) for description in scenes_to_process.values())
        print(f"📊 Total scene words: {self.total_scene_words:,}")
        
        print("=" * 60)

        results = {}
        # Initialize start time for time estimation
        self.start_time = time.time()
        
        print(f"\n📊 SCENE GENERATION PROGRESS")
        print("=" * 100)
        
        for i, (scene_id, scene_description) in enumerate(scenes_to_process.items(), 1):
            scene_start_time = time.time()
            eta = self.estimate_remaining_time(i, len(scenes_to_process), scene_description=scene_description)
            print(f"🔄 Scene {i}/{len(scenes_to_process)} - {scene_description[:50]} - Processing...")
            print(f"📊 Estimated time remaining: {eta}")
            
            character_names = self._extract_characters_from_scene(scene_description)
            valid_characters = [char for char in character_names if char in characters]

            location_ids = self._extract_location_ids_from_scene(scene_description)
            location_ids = [loc for loc in location_ids if loc in locations]
            
            if not valid_characters:
                print(f"WARNING: No valid characters found in {scene_id}, skipping...")
                continue
                
            output_path = self._generate_scene_image(scene_id, scene_description, valid_characters, location_ids, master_prompt, characters, locations, resumable_state)
            
            scene_processing_time = time.time() - scene_start_time
            self.processing_times.append(scene_processing_time)
            
            if output_path:
                eta = self.estimate_remaining_time(i, len(scenes_to_process), scene_processing_time, scene_description)
                print(f"✅ Scene {i}/{len(scenes_to_process)} - {scene_description[:50]} - Completed in {self.format_processing_time(scene_processing_time)}")
                print(f"📊 Estimated time remaining: {eta}")
                results[scene_id] = output_path
                print(f"[OK] Generated: {scene_id}")
            else:
                eta = self.estimate_remaining_time(i, len(scenes_to_process), scene_processing_time, scene_description)
                print(f"❌ Scene {i}/{len(scenes_to_process)} - {scene_description[:50]} - Failed after {self.format_processing_time(scene_processing_time)}")
                print(f"📊 Estimated time remaining: {eta}")
                print(f"[FAILED] {scene_id}")

        return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate scene images using ComfyUI")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration of all scenes")
    parser.add_argument("--list-completed", "-l", action="store_true", help="List completed scenes")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")
    args = parser.parse_args()
    
    generator = SceneGenerator()
    
    if args.list_completed:
        completed = generator._get_completed_scenes()
        print(f"Completed scenes ({len(completed)}): {sorted(completed)}" if completed else "No completed scenes")
        return 0
    
    # Initialize resumable state if enabled
    resumable_state = None
    if ENABLE_RESUMABLE_MODE:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.normpath(os.path.join(base_dir, "../output/tracking"))
        script_name = Path(__file__).stem  # Automatically get script name without .py extension
        resumable_state = ResumableState(checkpoint_dir, script_name, args.force_start)
        print(f"Resumable mode enabled - checkpoint directory: {checkpoint_dir}")
        if resumable_state.state_file.exists():
            print(f"Found existing checkpoint: {resumable_state.state_file}")
            print(resumable_state.get_progress_summary())
        else:
            print("No existing checkpoint found - starting fresh")
    
    start_time = time.time()
    results = generator.generate_all_scenes(force_regenerate=args.force, resumable_state=resumable_state)
    elapsed = time.time() - start_time
    
    if results:
        print(f"\nGenerated {len(results)} scenes in {elapsed:.2f}s:")
        for scene_id, path in results.items():
            print(f"  {scene_id}: {path}")
        
        # Clean up checkpoint files if resumable mode was used and everything completed successfully
        if resumable_state:
            print("All operations completed successfully")
            print("Final progress:", resumable_state.get_progress_summary())
            resumable_state.cleanup()
    else:
        print("No new scenes generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
