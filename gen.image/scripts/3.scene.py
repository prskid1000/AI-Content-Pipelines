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

# Image resizing configuration (characters only)
# Character image dimensions: 256x1024 (width x height) - Better aspect ratio for stitching
CHARACTER_RESIZE_WIDTH = 256
CHARACTER_RESIZE_HEIGHT = 1024

# Image compression configuration
# JPEG quality: 1-100 (100 = best quality, larger file; 1 = worst quality, smaller file)
IMAGE_COMPRESSION_QUALITY = 100

# Character prompt handling modes
# "IMAGE_TEXT" Send character images + character details appended from characters.txt
# "TEXT" Only character details from characters.txt
# "IMAGE" Only images

# HARDCODED CHARACTER MODE - Change this to switch modes
ACTIVE_CHARACTER_MODE = "IMAGE_TEXT"

# Image Resolution Constants
IMAGE_MEGAPIXEL = "1.2"
IMAGE_ASPECT_RATIO = "16:9 (Panorama)"
IMAGE_DIVISIBLE_BY = "64"
IMAGE_CUSTOM_RATIO = False
IMAGE_CUSTOM_ASPECT_RATIO = "1:1"

# Image Output Dimension Constants
USE_FIXED_DIMENSIONS = True  # Set to True to use fixed width/height, False to use aspect ratio calculation
IMAGE_OUTPUT_WIDTH = 1280
IMAGE_OUTPUT_HEIGHT = 720

# Image Stitching Configuration (1-5)
IMAGE_STITCH_COUNT = 1  # Number of images to stitch together in each group

# LoRA Configuration
USE_LORA = True  # Set to False to disable LoRA usage in workflow
LORA_NAME = "FLUX.1-Turbo-Alpha.safetensors"  # LoRA file name
LORA_STRENGTH_MODEL = 2.0  # LoRA strength for the model (0.0 - 2.0)
LORA_STRENGTH_CLIP = 2.0   # LoRA strength for CLIP (0.0 - 2.0)

# Sampling Configuration
SAMPLING_STEPS = 9 # Number of sampling steps (higher = better quality, slower)

# Negative Prompt Configuration
USE_NEGATIVE_PROMPT = True  # Set to True to enable negative prompts, False to disable
NEGATIVE_PROMPT = "blur, distorted, text, watermark, extra limbs, bad anatomy, poorly drawn, asymmetrical, malformed, disfigured, ugly, bad proportions, plastic texture, artificial looking, cross-eyed, missing fingers, extra fingers, bad teeth, missing teeth, unrealistic"

# Random Seed Configuration
USE_RANDOM_SEED = True  # Set to True to use random seed, False to use fixed seed - > Use when correcting images by regenerating
FIXED_SEED = 333555666  # Fixed seed value when USE_RANDOM_SEED is False

# Location Information Configuration
USE_LOCATION_INFO = True  # Set to True to replace {{loc_1}} with location descriptions from 3.location.txt

ART_STYLE = "3D Animation, often for computer-generated imagery, three-dimensional modeling, or virtual cinematography"


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
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        completed = len(self.state["scenes"]["completed"])
        total = len(self.state["scenes"]["results"]) + len([k for k in self.state["scenes"]["results"].keys() if k not in self.state["scenes"]["completed"]])
        
        return f"Progress: Scenes({completed}/{total})"


class SceneGenerator:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188/"):
        self.comfyui_url = comfyui_url
        self.character_mode = ACTIVE_CHARACTER_MODE
        # ComfyUI saves images under this folder
        self.comfyui_output_folder = "../../ComfyUI/output"
        # ComfyUI input folder where we need to copy character images
        self.comfyui_input_folder = "../../ComfyUI/input"
        # Final destination inside this repo
        self.final_output_dir = "../output/scene"
        self.scene_file = "../input/3.scene.txt"
        self.character_file = "../input/3.character.txt"
        self.location_file = "../input/3.location.txt"
        self.workflow_file = "../workflow/scene.json"
        self.character_images_dir = "../output/characters"

        # Create output directory
        os.makedirs(self.final_output_dir, exist_ok=True)
        # Ensure ComfyUI input directory exists
        os.makedirs(self.comfyui_input_folder, exist_ok=True)

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

    def _get_character_details(self, character_names: list[str], characters_data: dict[str, str]) -> str:
        """Get character details text for the given character names with position information."""
        if not character_names or not characters_data:
            return ""
        
        details = []
        for i, char in enumerate(character_names):
            if char in characters_data:
                # Calculate position information (only for IMAGE and IMAGE_TEXT modes)
                if self.character_mode in ["IMAGE", "IMAGE_TEXT"]:
                    # X = position within group (1-based)
                    # Y = group number (1-based)
                    position_in_group = (i % IMAGE_STITCH_COUNT) + 1
                    group_number = (i // IMAGE_STITCH_COUNT) + 1
                    
                    # Create position description using helper method
                    position_desc = self._get_position_description(position_in_group, group_number)
                    
                    details.append(f"({position_desc}) is (({char})) who looks like {{{characters_data[char]}}}.")
                else:
                    # TEXT mode: use simple format without position information
                    details.append(f"{char} WITH {{{characters_data[char]}}}.")
        
        return "\n".join(details)
    
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

    def _replace_location_references(self, scene_description: str, locations_data: dict[str, str]) -> str:
        """Replace {{loc_id}} references with actual location descriptions."""
        if not USE_LOCATION_INFO or not locations_data:
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

    def _replace_character_references(self, scene_description: str, character_names: list[str]) -> str:
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
                return f"Place (({char_name})), the {position_desc},"
            else:
                # Character not found in the list, keep original
                return full_match
        
        # Replace ((character_name)) patterns with position format
        result = re.sub(r'\(\(([^)]+)\)\)', replace_character_func, scene_description)
        return result

    def _get_master_prompt(self) -> str:
        """Get the master prompt content."""
        return """Create a 16K ultra-high-resolution, illustration in the style of {ART_STYLE}. The artwork should feature fine, intricate details and a natural sense of depth, with carefully chosen camera angle and focus to best frame the Scene. 
Must Always Precisely & Accurately Preserve each Character's identity(all physical features - face, body, height, weight, clothings) from respective specified reference image, though "posture", "expression", "movement", "placement" and "action-performed" is adaptable according to Scene/Character text-description.
Must Always Precisely & Accurately Represent entire Scene and all Non-Living Objects according to scene text-description.
All Non-Living Objects mentioned in Scene text-description must be present in illustration.
Each Object/Character in the illustration must be visually distinct/unique from each other.
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

    def _get_character_image_path(self, character_name: str) -> str | None:
        """Get the path to a character's generated image."""
        clean_name = re.sub(r'[^\w\s-]', '', character_name).strip()
        clean_name = re.sub(r'[-\s]+', '_', clean_name)
        image_path = os.path.join(self.character_images_dir, f"{clean_name}.png")
        return image_path if os.path.exists(image_path) else None

    def _copy_character_images_to_comfyui(self, character_names: list[str]) -> dict[str, str]:
        """Copy, resize, and compress character images to ComfyUI input directory."""
        copied_images = {}
        for char_name in character_names:
            source_path = self._get_character_image_path(char_name)
            if source_path:
                clean_name = re.sub(r'[^\w\s-]', '', char_name).strip()
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
                        
                        # Resize the image before saving - maintain aspect ratio with padding
                        # Calculate the scaling factor to fit within target dimensions
                        scale_w = CHARACTER_RESIZE_WIDTH / img.width
                        scale_h = CHARACTER_RESIZE_HEIGHT / img.height
                        scale = min(scale_w, scale_h)
                        
                        # Calculate new size
                        new_width = int(img.width * scale)
                        new_height = int(img.height * scale)
                        
                        # Resize with aspect ratio preserved
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # Create a new image with target dimensions and white background
                        padded_img = Image.new('RGB', (CHARACTER_RESIZE_WIDTH, CHARACTER_RESIZE_HEIGHT), (255, 255, 255))
                        
                        # Calculate position to center the resized image
                        paste_x = (CHARACTER_RESIZE_WIDTH - new_width) // 2
                        paste_y = (CHARACTER_RESIZE_HEIGHT - new_height) // 2
                        
                        # Paste the resized image onto the padded background
                        padded_img.paste(img, (paste_x, paste_y))
                        img = padded_img
                        
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

    def _load_base_workflow(self) -> dict:
        """Load the base scene workflow and modify based on LoRA settings."""
        try:
            with open(self.workflow_file, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            
            # Modify workflow based on USE_LORA setting
            if USE_LORA:
                # Add LoRA loader node
                workflow["43"] = {
                    "inputs": {
                        "lora_name": LORA_NAME,
                        "strength_model": LORA_STRENGTH_MODEL,
                        "strength_clip": LORA_STRENGTH_CLIP,
                        "model": ["41", 0],
                        "clip": ["10", 0]
                    },
                    "class_type": "LoraLoader",
                    "_meta": {
                        "title": "Load LoRA"
                    }
                }
                # Update KSampler to use LoRA model
                workflow["16"]["inputs"]["model"] = ["43", 0]
                # Update CLIPTextEncode to use LoRA clip
                workflow["33"]["inputs"]["clip"] = ["43", 1]
                print("LoRA enabled in workflow")
            else:
                # Ensure KSampler uses base model directly
                workflow["16"]["inputs"]["model"] = ["41", 0]
                # Ensure CLIPTextEncode uses base clip directly
                workflow["33"]["inputs"]["clip"] = ["10", 0]
                # Remove LoRA node if it exists
                if "43" in workflow:
                    del workflow["43"]
                print("LoRA disabled in workflow")
            
            # Set sampling steps and seed
            workflow["16"]["inputs"]["steps"] = SAMPLING_STEPS
            seed = self._get_seed()
            workflow["16"]["inputs"]["seed"] = seed
            print(f"Sampling steps set to: {SAMPLING_STEPS}")
            print(f"Seed set to: {seed}")
            
            return workflow
        except Exception as e:
            print(f"ERROR: Failed to load workflow: {e}")
            return {}

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

    def _build_dynamic_workflow(self, scene_id: str, scene_description: str, character_names: list[str], master_prompt: str, characters_data: dict[str, str], locations_data: dict[str, str] = None) -> dict:
        """Build a dynamic workflow with N character images."""
        workflow = self._load_base_workflow()
        if not workflow:
            return {}

        # Handle different character modes
        all_images = {}
        if self.character_mode in ["IMAGE", "IMAGE_TEXT"]:
            all_images = self._copy_character_images_to_comfyui(character_names)
            if not all_images and self.character_mode == "IMAGE":
                print("ERROR: No images copied to ComfyUI!")
                return {}
        
        print(f"Character mode: {self.character_mode}, Images: {len(all_images)}")
        
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

        # Replace location references if location data is available
        processed_scene_description = scene_description
        if locations_data:
            processed_scene_description = self._replace_location_references(scene_description, locations_data)
        
        # Replace character references with position format (only in IMAGE and IMAGE_TEXT modes)
        if self.character_mode in ["IMAGE", "IMAGE_TEXT"]:
            processed_scene_description = self._replace_character_references(processed_scene_description, character_names)

        text_prompt += f"\nSCENE TEXT-DESCRIPTION:\n Illustrate an exact scenery like {processed_scene_description}.\n"

        if self.character_mode in ["TEXT", "IMAGE_TEXT"]:
            character_details = self._get_character_details(character_names, characters_data)
            if character_details:
                text_prompt += f"\nCHARACTER TEXT-DESCRIPTION:\n{character_details}"
        
        workflow["33"]["inputs"]["text"] = text_prompt
        workflow["21"]["inputs"]["filename_prefix"] = scene_id
        
        # Set resolution parameters
        workflow["23"]["inputs"]["megapixel"] = IMAGE_MEGAPIXEL
        workflow["23"]["inputs"]["aspect_ratio"] = IMAGE_ASPECT_RATIO
        workflow["23"]["inputs"]["divisible_by"] = IMAGE_DIVISIBLE_BY
        workflow["23"]["inputs"]["custom_ratio"] = IMAGE_CUSTOM_RATIO
        workflow["23"]["inputs"]["custom_aspect_ratio"] = IMAGE_CUSTOM_ASPECT_RATIO
        
        # Override with fixed dimensions if specified
        if USE_FIXED_DIMENSIONS:
            # For fixed dimensions, bypass the FluxResolutionNode and set dimensions directly
            # Disconnect from FluxResolutionNode and set fixed values
            workflow["19"]["inputs"]["width"] = IMAGE_OUTPUT_WIDTH
            workflow["19"]["inputs"]["height"] = IMAGE_OUTPUT_HEIGHT
            print(f"Using fixed dimensions: {IMAGE_OUTPUT_WIDTH}x{IMAGE_OUTPUT_HEIGHT} (bypassing aspect ratio calculation)")
        else:
            print(f"Using aspect ratio calculation: {IMAGE_ASPECT_RATIO} with {IMAGE_MEGAPIXEL} megapixels")
        
        # Handle negative prompt
        if USE_NEGATIVE_PROMPT:
            # Create a new CLIPTextEncode node for negative prompt
            negative_node_id = "35"
            workflow[negative_node_id] = {
                "inputs": {
                    "text": NEGATIVE_PROMPT,
                    "clip": ["10", 0] if not USE_LORA else ["43", 1]
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
        
        print("\n\n\n")
        print(f"Text prompt: {text_prompt}")
        return workflow

    def _generate_scene_image(self, scene_id: str, scene_description: str, character_names: list[str], master_prompt: str, characters_data: dict[str, str], locations_data: dict[str, str] = None, resumable_state=None) -> str | None:
        """Generate a single scene image using ComfyUI."""
        try:
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_scene_complete(scene_id):
                cached_result = resumable_state.get_scene_result(scene_id)
                if cached_result and os.path.exists(cached_result.get('path', '')):
                    print(f"Using cached scene image: {scene_id}")
                    return cached_result['path']
                elif cached_result:
                    print(f"Cached file missing, regenerating: {scene_id}")
            
            print(f"Generating scene: {scene_id} with characters: {', '.join(character_names)}")
            workflow = self._build_dynamic_workflow(scene_id, scene_description, character_names, master_prompt, characters_data, locations_data)
            if not workflow:
                return None

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
        locations = self._read_location_data() if USE_LOCATION_INFO else {}
        master_prompt = self._get_master_prompt()
        
        if not scenes or not master_prompt:
            print("ERROR: Missing scene data or master prompt")
            return {}

        if USE_LOCATION_INFO and locations:
            print(f"Location info enabled: {len(locations)} locations loaded")
        elif USE_LOCATION_INFO:
            print("WARNING: Location info enabled but no location data found")

        # Use resumable state if available, otherwise fall back to file-based checking
        if resumable_state:
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
        print("=" * 60)

        results = {}
        for i, (scene_id, scene_description) in enumerate(scenes_to_process.items(), 1):
            print(f"\n[{i}/{len(scenes_to_process)}] Processing {scene_id}...")
            
            character_names = self._extract_characters_from_scene(scene_description)
            valid_characters = [char for char in character_names if char in characters]
            
            if not valid_characters:
                print(f"WARNING: No valid characters found in {scene_id}, skipping...")
                continue
                
            output_path = self._generate_scene_image(scene_id, scene_description, valid_characters, master_prompt, characters, locations, resumable_state)
            if output_path:
                results[scene_id] = output_path
                print(f"[OK] Generated: {scene_id}")
            else:
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
