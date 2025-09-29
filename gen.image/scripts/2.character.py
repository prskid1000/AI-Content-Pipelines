import os
import json
import time
import shutil
import re
import random
from pathlib import Path
import requests
from functools import partial
import builtins as _builtins
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse
print = partial(_builtins.print, flush=True)

# Feature flags
ENABLE_RESUMABLE_MODE = True
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them

# Image Resolution Constants
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Latent Input Mode Configuration
LATENT_MODE = "LATENT"  # "LATENT" for normal noise generation, "IMAGE" for load image input
LATENT_DENOISING_STRENGTH = 0.8  # Denoising strength when using IMAGE mode (0.0-1.0, higher = more change)

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
        "strength_model": 3.0,    # Model strength (0.0 - 2.0)
        "strength_clip": 3.0,     # CLIP strength (0.0 - 2.0)
        "bypass_model": False,    # Set to True to bypass model part of this LoRA
        "bypass_clip": False,     # Set to True to bypass CLIP part of this LoRA
        "enabled": True,          # Set to False to disable this LoRA entirely
        
        # Serial mode specific settings (only used when LORA_MODE = "serial")
        "steps": 9,               # Sampling steps for this LoRA (serial mode only)
        "denoising_strength": 1,  # Denoising strength (0.0 - 1.0) (serial mode only)
        "save_intermediate": True, # Save intermediate results for debugging (serial mode only)
        "use_only_intermediate": False # Set to True to disable character images and use only intermediate result
    },
    {
        "name": "Ghibli_lora_weights.safetensors",  # Example second LoRA
        "strength_model": 2.0,
        "strength_clip": 2.0,
        "bypass_model": False,
        "bypass_clip": False,
        "enabled": False,  # Disabled by default
        
        # Serial mode specific settings
        "steps": 45,
        "denoising_strength": 0.6,
        "save_intermediate": True,
        "use_only_intermediate": True  # This LoRA will only use intermediate result, no character images
    },
]

# Sampling Configuration
SAMPLING_STEPS = 25  # Number of sampling steps (higher = better quality, slower)

# Negative Prompt Configuration
USE_NEGATIVE_PROMPT = True  # Set to True to enable negative prompts, False to disable
NEGATIVE_PROMPT = "blur, distorted, text, watermark, extra limbs, bad anatomy, poorly drawn, asymmetrical, malformed, disfigured, ugly, bad proportions, plastic texture, artificial looking, cross-eyed, missing fingers, extra fingers, bad teeth, missing teeth, unrealistic"

# Random Seed Configuration
USE_RANDOM_SEED = True  # Set to True to use random seeds, False to use fixed seed
FIXED_SEED = 333555666  # Fixed seed value when USE_RANDOM_SEED is False

ART_STYLE = "Realistic Anime"

# Text overlay settings for character names
USE_CHARACTER_NAME_OVERLAY = False  # Set to False to disable name overlay
CHARACTER_NAME_FONT_SCALE = 1
CHARACTER_NAME_BAND_HEIGHT_RATIO = 0.30  # 15% of image height for name band


class ResumableState:
    """Manages resumable state for expensive character generation operations."""
    
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
            "characters": {
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
    
    def is_character_complete(self, character_name: str) -> bool:
        """Check if character generation is complete."""
        return character_name in self.state["characters"]["completed"]
    
    def get_character_result(self, character_name: str) -> dict:
        """Get character generation result."""
        return self.state["characters"]["results"].get(character_name, {})
    
    def set_character_result(self, character_name: str, result: dict):
        """Set character generation result and mark as complete."""
        self.state["characters"]["results"][character_name] = result
        if character_name not in self.state["characters"]["completed"]:
            self.state["characters"]["completed"].append(character_name)
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
        """Validate that all completed character files actually exist and clean up missing entries.
        
        Returns:
            int: Number of entries cleaned up (removed from completed list)
        """
        cleaned_count = 0
        characters_to_remove = []
        
        # Check each completed character
        for character_name in self.state["characters"]["completed"]:
            result = self.state["characters"]["results"].get(character_name, {})
            file_path = result.get('path', '')
            
            # Check if file actually exists
            if not file_path or not os.path.exists(file_path):
                print(f"Precheck: File missing for {character_name} - marking as not completed")
                characters_to_remove.append(character_name)
                cleaned_count += 1
        
        # Remove invalid entries
        for character_name in characters_to_remove:
            if character_name in self.state["characters"]["completed"]:
                self.state["characters"]["completed"].remove(character_name)
            if character_name in self.state["characters"]["results"]:
                del self.state["characters"]["results"][character_name]
        
        # Save cleaned state if any changes were made
        if cleaned_count > 0:
            self._save_state()
            print(f"Precheck: Cleaned up {cleaned_count} invalid entries from checkpoint")
        
        return cleaned_count
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        completed = len(self.state["characters"]["completed"])
        total = len(self.state["characters"]["results"]) + len([k for k in self.state["characters"]["results"].keys() if k not in self.state["characters"]["completed"]])
        
        return f"Progress: Characters({completed}/{total})"


class CharacterGenerator:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188/", mode: str = "flux"):
        self.comfyui_url = comfyui_url
        self.mode = (mode or "flux").strip().lower()
        # ComfyUI saves images under this folder
        self.comfyui_output_folder = "../../ComfyUI/output"
        # Final destination inside this repo
        self.final_output_dir = "../output/characters"
        self.intermediate_output_dir = "../output/lora"
        self.input_file = "../input/2.character.txt"
        # Latent image input file path
        self.latent_image_path = "../input/2.latent.png"
        # Dynamic workflow file selection based on mode
        self.workflow_file = "../workflow/character.flux.json" if self.mode == "flux" else "../workflow/character.json"

        # Create output directories
        os.makedirs(self.final_output_dir, exist_ok=True)
        os.makedirs(self.intermediate_output_dir, exist_ok=True)

    def _read_character_data(self) -> dict[str, str]:
        """Parse character data from input file.
        
        Returns:
            dict: {character_name: description} mapping
        """
        characters = {}
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
            # Split by double newlines to separate character entries
            entries = [entry.strip() for entry in content.split('\n\n') if entry.strip()]
            
            for entry in entries:
                # Match pattern: ((Character Name)): Description
                match = re.match(r'\(\(([^)]+)\)\):\s*(.+)', entry, re.DOTALL)
                if match:
                    character_name = match.group(1).strip()
                    description = match.group(2).strip()
                    characters[character_name] = description
                    
        except Exception as e:
            print(f"ERROR: Failed to read character data: {e}")
            
        return characters

    def _load_character_workflow(self) -> dict:
        """Load the character generation workflow and apply settings."""
        try:
            with open(self.workflow_file, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            
            print(f"Loaded {self.mode} workflow from: {self.workflow_file}")
            
            # Validate LoRA configuration
            if not self._validate_lora_config():
                print("ERROR: LoRA configuration validation failed")
                return {}
            
            # Apply LoRA and sampling settings
            workflow = self._apply_workflow_settings(workflow)
            
            return workflow
        except Exception as e:
            print(f"ERROR: Failed to load workflow: {e}")
            return {}
    
    def _apply_workflow_settings(self, workflow: dict) -> dict:
        """Apply LoRA and sampling configuration to workflow."""
        # Apply LoRA settings
        if USE_LORA:
            # Handle multiple LoRAs in series
            self._apply_loras(workflow)
            print("LoRA enabled in character workflow")
        else:
            # Remove all LoRA nodes if they exist
            self._remove_all_lora_nodes(workflow)
            print("LoRA disabled in character workflow")
        
        # Apply sampling steps
        self._update_node_connections(workflow, "KSampler", "steps", SAMPLING_STEPS)
        print(f"Sampling steps set to: {SAMPLING_STEPS}")
        
        # Handle negative prompt
        if USE_NEGATIVE_PROMPT:
            # Create a new CLIPTextEncode node for negative prompt
            negative_node_id = "35"
            workflow[negative_node_id] = {
                "inputs": {
                    "text": NEGATIVE_PROMPT,
                    "clip": self._find_node_by_class(workflow, ["DualCLIPLoader", "TripleCLIPLoader"]) or ["10", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIP Text Encode (Negative Prompt)"
                }
            }
            # Connect negative prompt directly to sampler's negative conditioning
            self._update_node_connections(workflow, "KSampler", "negative", [negative_node_id, 0])
            # Remove ConditioningZeroOut from the workflow when using negative prompt
            conditioning_zero_out = self._find_node_by_class(workflow, "ConditioningZeroOut")
            if conditioning_zero_out:
                del workflow[conditioning_zero_out[0]]
            print(f"Negative prompt enabled: {NEGATIVE_PROMPT}")
        else:
            # Keep ConditioningZeroOut for empty negative (it's already connected in base workflow)
            print("Negative prompt disabled - using ConditioningZeroOut")
        
        return workflow
    
    def _apply_loras(self, workflow: dict) -> None:
        """Apply LoRAs based on mode (serial or chained)."""
        if LORA_MODE == "serial":
            self._apply_loras_serial(workflow)
        else:
            self._apply_loras_chained(workflow)
    
    def _apply_loras_chained(self, workflow: dict) -> None:
        """Apply LoRAs in series with individual bypass options (chained mode)."""
        enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
        
        if not enabled_loras:
            print("No enabled LoRAs found in LORAS configuration")
            return
        
        print(f"Applying {len(enabled_loras)} LoRAs in chained mode...")
        
        # Get initial model and clip connections
        model_input = self._find_node_by_class(workflow, "UnetLoaderGGUF") or ["1", 0]
        clip_input = self._find_node_by_class(workflow, ["DualCLIPLoader", "TripleCLIPLoader"]) or ["2", 0]
        
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
        
        print(f"Multiple LoRAs chain completed with {len(enabled_loras)} LoRAs")
    
    def _apply_loras_serial(self, workflow: dict) -> None:
        """Apply LoRAs in serial mode - each LoRA runs independently."""
        enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
        
        if not enabled_loras:
            print("No enabled LoRAs found in LORAS configuration")
            return
        
        print(f"Serial LoRA mode: {len(enabled_loras)} LoRAs will run independently")
        print("Note: Serial mode requires separate workflow execution for each LoRA")
    
    def _remove_all_lora_nodes(self, workflow: dict) -> None:
        """Remove all LoRA nodes from workflow."""
        # Remove LoRA nodes (lora_1, lora_2, etc.)
        nodes_to_remove = []
        for node_id in workflow.keys():
            if node_id.startswith("lora_"):
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            del workflow[node_id]
    
    def _validate_lora_config(self) -> bool:
        """Validate LoRA configuration and print helpful messages."""
        if not USE_LORA:
            print("LoRA disabled globally")
            return True
        
        if not LORAS:
            print("ERROR: USE_LORA is True but LORAS list is empty")
            return False
        
        enabled_count = sum(1 for lora in LORAS if lora.get("enabled", True))
        if enabled_count == 0:
            print("WARNING: No enabled LoRAs found in LORAS configuration")
            return False
        
        print(f"LoRAs configuration: {enabled_count}/{len(LORAS)} LoRAs enabled")
        
        # Validate each LoRA configuration
        for i, lora in enumerate(LORAS):
            if not lora.get("enabled", True):
                continue
                
            if "name" not in lora:
                print(f"ERROR: LoRA {i + 1} missing 'name' field")
                return False
            
            # Check for invalid strength values
            model_strength = lora.get("strength_model", 1.0)
            clip_strength = lora.get("strength_clip", 1.0)
            
            if not (0.0 <= model_strength <= 2.0):
                print(f"WARNING: LoRA {i + 1} ({lora['name']}) model strength {model_strength} outside recommended range (0.0-2.0)")
            
            if not (0.0 <= clip_strength <= 2.0):
                print(f"WARNING: LoRA {i + 1} ({lora['name']}) CLIP strength {clip_strength} outside recommended range (0.0-2.0)")
        
        return True
    
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

    def _update_workflow_prompt(self, workflow: dict, character_name: str, description: str) -> dict:
        """Update the workflow with character-specific prompt."""
        prompt = f"Create a 16K ultra-high-resolution, Full Body Visible, Illustration in the style of {ART_STYLE} in which torso, limbs, hands, feet, face(eyes, nose, mouth, skin), clothes, ornaments, props, precisely and accurately matching character with description and fine-level detailing, and any part not cropped or hidden.Must use White Background.\n\n Character Description = {description}. Strictly, Accurately, Precisely, always must Follow {ART_STYLE} Style."
        self._update_node_connections(workflow, ["CLIPTextEncode", "CLIP Text Encode (Prompt)"], "text", prompt)
        return workflow

    def _update_workflow_filename(self, workflow: dict, character_name: str) -> dict:
        """Update the workflow to save with character name as filename."""
        clean_name = re.sub(r'[^\w\s.-]', '', character_name).strip()
        clean_name = re.sub(r'[-\s]+', '_', clean_name)
        self._update_node_connections(workflow, "SaveImage", "filename_prefix", clean_name)
        return workflow

    def _update_workflow_seed(self, workflow: dict, seed: int = None) -> dict:
        """Update the workflow with a seed based on configuration."""
        if seed is None:
            if USE_RANDOM_SEED:
                seed = random.randint(1, 2**32 - 1)
            else:
                seed = FIXED_SEED
        self._update_node_connections(workflow, "KSampler", "seed", seed)
        return workflow

    def _update_workflow_resolution(self, workflow: dict) -> dict:
        """Update the workflow with resolution settings and handle latent input mode."""
        # Handle EmptySD3LatentImage node
        latent_image_node = self._find_node_by_class(workflow, "EmptySD3LatentImage")
        if latent_image_node:
            node_id = latent_image_node[0]
            
            if LATENT_MODE == "IMAGE":
                # Replace EmptySD3LatentImage with LoadImage + VAEEncode for image input
                self._replace_latent_with_image_input(workflow, node_id, self.latent_image_path)
                print(f"Using image input mode with file: {self.latent_image_path}")
            else:
                # Normal latent mode - set dimensions
                workflow[node_id]["inputs"]["width"] = IMAGE_WIDTH
                workflow[node_id]["inputs"]["height"] = IMAGE_HEIGHT
                print(f"Using latent mode with dimensions: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        
        return workflow

    def _replace_latent_with_image_input(self, workflow: dict, latent_node_id: str, image_path: str) -> None:
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
                        sampler_node["inputs"]["denoise"] = LATENT_DENOISING_STRENGTH
                        break
            
            # Remove the original EmptySD3LatentImage node
            del workflow[latent_node_id]
            
            print(f"  Replaced EmptySD3LatentImage with LoadImage + VAEEncode")
            print(f"  LoadImage node: {load_image_node_id}, VAEEncode node: {encode_node_id}")
            print(f"  Denoising strength set to: {LATENT_DENOISING_STRENGTH}")
            
        except Exception as e:
            print(f"WARNING: Failed to replace latent with image input: {e}")

    def _replace_latent_with_previous_output(self, workflow: dict, image_path: str) -> None:
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
                        # Keep the existing denoising strength from LoRA config
                        break
            
            # Remove the original EmptySD3LatentImage node
            if "19" in workflow:
                del workflow["19"]
            
            print(f"  Replaced EmptySD3LatentImage with LoadImage + VAEEncode")
            print(f"  LoadImage node: {load_image_node_id}, VAEEncode node: {encode_node_id}")
            
        except Exception as e:
            print(f"WARNING: Failed to replace latent with previous output: {e}")

    def _generate_character_image_serial(self, character_name: str, description: str, resumable_state=None) -> str | None:
        """Generate character image using serial LoRA mode with intermediate storage."""
        try:
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_character_complete(character_name):
                cached_result = resumable_state.get_character_result(character_name)
                if cached_result and os.path.exists(cached_result.get('path', '')):
                    print(f"Using cached character image: {character_name}")
                    return cached_result['path']
                elif cached_result:
                    print(f"Cached file missing, regenerating: {character_name}")
            
            print(f"Generating image for: {character_name} (Serial LoRA mode)")
            
            enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
            if not enabled_loras:
                print("ERROR: No enabled LoRAs found for serial mode")
                return None
            
            # Clean character name for filenames (preserve dots for version numbers like 1.1)
            clean_name = re.sub(r'[^\w\s.-]', '', character_name).strip()
            clean_name = re.sub(r'[-\s]+', '_', clean_name)
            
            # Check for existing LoRA progress
            lora_progress_key = f"{character_name}_lora_progress"
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
            
            # Process each LoRA in sequence, using previous output as input
            for i, lora_config in enumerate(enabled_loras):
                lora_name = lora_config['name']
                lora_clean_name = re.sub(r'[^\w\s-]', '', lora_name).strip()
                lora_clean_name = re.sub(r'[-\s]+', '_', lora_clean_name)
                
                # Skip if this LoRA was already completed
                if lora_name in completed_loras:
                    print(f"Skipping completed LoRA {i + 1}/{len(enabled_loras)}: {lora_name}")
                    continue
                
                print(f"\nProcessing LoRA {i + 1}/{len(enabled_loras)}: {lora_name}")
                
                # Load base workflow for this LoRA
                workflow = self._load_character_workflow()
                if not workflow:
                    print(f"ERROR: Failed to load workflow for LoRA {i + 1}")
                    continue
                
                # Apply only this LoRA to the workflow
                self._apply_single_lora(workflow, lora_config, i + 1)
                
                # Update workflow with character-specific settings
                workflow = self._update_workflow_prompt(workflow, character_name, description)
                workflow = self._update_workflow_seed(workflow)
                workflow = self._update_workflow_resolution(workflow)
                
                # Set LoRA-specific sampling steps and denoising
                steps = lora_config.get("steps", SAMPLING_STEPS)
                denoising_strength = lora_config.get("denoising_strength", 1.0)
                self._update_node_connections(workflow, "KSampler", "steps", steps)
                self._update_node_connections(workflow, "KSampler", "denoise", denoising_strength)
                
                # Handle input for this LoRA (first LoRA can use latent/image mode, subsequent LoRAs use previous output)
                if i > 0 and current_image_path:
                    # For subsequent LoRAs, use previous output as input
                    self._replace_latent_with_previous_output(workflow, current_image_path)
                    print(f"  Using previous LoRA output as latent input")
                elif i == 0 and LATENT_MODE == "IMAGE":
                    # For first LoRA, use image input mode if configured
                    self._replace_latent_with_image_input(workflow, "19", self.latent_image_path)
                    print(f"  Using image input mode for first LoRA with file: {self.latent_image_path}")
                
                # Generate filename for this LoRA step
                lora_clean_name = re.sub(r'[^\w\s-]', '', lora_config['name']).strip()
                lora_clean_name = re.sub(r'[-\s]+', '_', lora_clean_name)
                lora_filename = f"{clean_name}.{lora_clean_name}"
                self._update_node_connections(workflow, "SaveImage", "filename_prefix", lora_filename)
                
                print(f"  Steps: {steps}, Denoising: {denoising_strength}")
                
                # Print prompt before sending
                print(f"\n=== PROMPT FOR LoRA {i + 1}: {lora_name} ===")
                # Get the text prompt from the workflow
                text_prompt = workflow.get("33", {}).get("inputs", {}).get("text", "No text prompt found")
                print(f"Text prompt: {text_prompt}")
                print(f"Workflow nodes: {len(workflow)} nodes")
                print("=" * 50)
                
                # Submit workflow to ComfyUI
                resp = requests.post(f"{self.comfyui_url}prompt", json={"prompt": workflow}, timeout=60)
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
                    continue
                
                # Save result to lora folder (save final result from each LoRA)
                lora_final_path = os.path.join(self.intermediate_output_dir, f"{clean_name}.{lora_clean_name}.png")
                shutil.copy2(generated_image, lora_final_path)
                intermediate_paths.append(lora_final_path)
                print(f"  Saved LoRA result: {lora_final_path}")
                
                # Use this output as input for next LoRA
                current_image_path = generated_image
                print(f"  LoRA {i + 1} completed successfully")
                
                # Save progress after each LoRA completion
                if resumable_state:
                    completed_loras.append(lora_name)
                    lora_progress = {
                        "completed_loras": completed_loras,
                        "current_image_path": current_image_path,
                        "intermediate_paths": intermediate_paths,
                        "lora_configs": {lora["name"]: lora for lora in enabled_loras}  # Save LoRA configs for resuming
                    }
                    resumable_state.state.setdefault("lora_progress", {})[lora_progress_key] = lora_progress
                    resumable_state._save_state()
                    print(f"  Saved LoRA progress: {len(completed_loras)}/{len(enabled_loras)} completed")
            
            if not current_image_path:
                print(f"ERROR: No successful LoRA generations for {character_name}")
                return None
            
            # Copy final result to output directory
            final_path = os.path.join(self.final_output_dir, f"{character_name}.png")
            shutil.copy2(current_image_path, final_path)
            
            # Apply character name overlay if enabled
            if USE_CHARACTER_NAME_OVERLAY:
                print(f"Adding character name overlay...")
                overlay_success = self._overlay_character_name(final_path, character_name)
                if overlay_success:
                    print(f"Saved with name overlay: {final_path}")
                else:
                    print(f"Saved (overlay failed): {final_path}")
            else:
                print(f"Saved: {final_path}")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                result = {
                    'path': final_path,
                    'character_name': character_name,
                    'description': description,
                    'intermediate_paths': intermediate_paths
                }
                resumable_state.set_character_result(character_name, result)
                
                # Keep LoRA progress for completed character (not cleaning up)
                print(f"  Preserved LoRA progress for completed character")
            
            return final_path

        except Exception as e:
            print(f"ERROR: Failed to generate image for {character_name}: {e}")
            return None

    def _apply_single_lora(self, workflow: dict, lora_config: dict, lora_index: int) -> None:
        """Apply a single LoRA to the workflow."""
        lora_node_id = f"lora_{lora_index}"
        
        # Get initial model and clip connections
        model_input = self._find_node_by_class(workflow, "UnetLoaderGGUF") or ["1", 0]
        clip_input = self._find_node_by_class(workflow, ["DualCLIPLoader", "TripleCLIPLoader"]) or ["2", 0]
        
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

    def _set_image_input(self, workflow: dict, image_path: str) -> None:
        """Set an image as input for the workflow (for chaining LoRA outputs)."""
        try:
            # Copy the image to ComfyUI input folder
            image_filename = os.path.basename(image_path)
            comfyui_input_path = os.path.join("../../ComfyUI/input", image_filename)
            shutil.copy2(image_path, comfyui_input_path)
            
            # Find existing LoadImage node or create one
            load_image_node_id = None
            for node_id, node in workflow.items():
                if isinstance(node, dict) and node.get("class_type") == "LoadImage":
                    load_image_node_id = node_id
                    break
            
            # If no LoadImage node exists, create one
            if not load_image_node_id:
                # Find the next available node ID
                max_id = max(int(k) for k in workflow.keys() if k.isdigit())
                load_image_node_id = str(max_id + 1)
                
                # Create LoadImage node
                workflow[load_image_node_id] = {
                    "inputs": {"image": image_filename},
                    "class_type": "LoadImage",
                    "_meta": {"title": "Load Image (LoRA Chain Input)"}
                }
                print(f"  Created LoadImage node: {load_image_node_id}")
            else:
                # Update existing LoadImage node
                workflow[load_image_node_id]["inputs"]["image"] = image_filename
                print(f"  Updated LoadImage node: {load_image_node_id}")
            
            # Find and replace EmptySD3LatentImage with LoadImage
            for node_id, node in workflow.items():
                if isinstance(node, dict) and node.get("class_type") == "EmptySD3LatentImage":
                    # Replace the latent_image input in KSampler
                    for sampler_id, sampler_node in workflow.items():
                        if isinstance(sampler_node, dict) and sampler_node.get("class_type") == "KSampler":
                            if "latent_image" in sampler_node.get("inputs", {}):
                                # Create VAEEncode node to convert image to latent
                                encode_node_id = str(int(load_image_node_id) + 1)
                                workflow[encode_node_id] = {
                                    "inputs": {
                                        "pixels": [load_image_node_id, 0],
                                        "vae": ["11", 0]  # Use existing VAE
                                    },
                                    "class_type": "VAEEncode",
                                    "_meta": {"title": "VAE Encode (LoRA Chain Input)"}
                                }
                                
                                # Update KSampler to use encoded latent
                                sampler_node["inputs"]["latent_image"] = [encode_node_id, 0]
                                print(f"  Connected LoadImage → VAEEncode → KSampler")
                                break
                    break
                    
        except Exception as e:
            print(f"WARNING: Failed to set image input: {e}")

    def _generate_character_image(self, character_name: str, description: str, resumable_state=None) -> str | None:
        """Generate a single character image using ComfyUI."""
        try:
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_character_complete(character_name):
                cached_result = resumable_state.get_character_result(character_name)
                if cached_result and os.path.exists(cached_result.get('path', '')):
                    print(f"Using cached character image: {character_name}")
                    return cached_result['path']
                elif cached_result:
                    print(f"Cached file missing, regenerating: {character_name}")
            
            # Use serial LoRA mode if enabled
            if USE_LORA and LORA_MODE == "serial":
                return self._generate_character_image_serial(character_name, description, resumable_state)
            
            print(f"Generating image for: {character_name}")
            
            # Load and update workflow
            workflow = self._load_character_workflow()
            if not workflow:
                return None
                
            workflow = self._update_workflow_prompt(workflow, character_name, description)
            workflow = self._update_workflow_filename(workflow, character_name)
            workflow = self._update_workflow_seed(workflow)
            workflow = self._update_workflow_resolution(workflow)

            # Print prompt before sending
            print(f"\n=== PROMPT FOR CHARACTER: {character_name} ===")
            # Get the text prompt from the workflow
            text_prompt = workflow.get("33", {}).get("inputs", {}).get("text", "No text prompt found")
            print(f"Text prompt: {text_prompt}")
            print(f"Workflow nodes: {len(workflow)} nodes")
            print("=" * 50)

            # Submit workflow to ComfyUI
            resp = requests.post(f"{self.comfyui_url}prompt", json={"prompt": workflow}, timeout=60)
            if resp.status_code != 200:
                print(f"ERROR: ComfyUI API error: {resp.status_code} {resp.text}")
                return None
                
            prompt_id = resp.json().get("prompt_id")
            if not prompt_id:
                print("ERROR: No prompt ID returned from ComfyUI")
                return None

            # Wait for completion
            print(f"Waiting for generation to complete (prompt_id: {prompt_id})...")
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
            generated_image = self._find_newest_output_with_prefix(character_name)
            if not generated_image:
                print(f"ERROR: Could not find generated image for {character_name}")
                return None

            # Copy to final output directory
            final_path = os.path.join(self.final_output_dir, f"{character_name}.png")
            shutil.copy2(generated_image, final_path)
            
            # Apply character name overlay if enabled
            if USE_CHARACTER_NAME_OVERLAY:
                print(f"Adding character name overlay...")
                overlay_success = self._overlay_character_name(final_path, character_name)
                if overlay_success:
                    print(f"Saved with name overlay: {final_path}")
                else:
                    print(f"Saved (overlay failed): {final_path}")
            else:
                print(f"Saved: {final_path}")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                result = {
                    'path': final_path,
                    'character_name': character_name,
                    'description': description
                }
                resumable_state.set_character_result(character_name, result)
            
            return final_path

        except Exception as e:
            print(f"ERROR: Failed to generate image for {character_name}: {e}")
            return None

    def _find_newest_output_with_prefix(self, prefix: str) -> str | None:
        """Find the newest generated image with the given prefix."""
        if not os.path.isdir(self.comfyui_output_folder):
            return None
            
        latest = None
        latest_mtime = -1.0
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        
        for root, _dirs, files in os.walk(self.comfyui_output_folder):
            for name in files:
                if name.startswith(prefix) and Path(name).suffix.lower() in exts:
                    full = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(full)
                    except OSError:
                        continue
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest = full
        return latest

    def _load_font(self, size: int) -> ImageFont.ImageFont:
        """Load a suitable font for character name overlay."""
        candidates = [
            "Impact.ttf",
            "Anton-Regular.ttf",
            "BebasNeue-Regular.ttf",
            "Arial Bold.ttf",
            "arialbd.ttf",
            "DejaVuSans-Bold.ttf",
        ]
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _wrap_lines(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, stroke_width: int = 0) -> list[str]:
        """Wrap text to fit within the specified width, breaking at word boundaries."""
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            trial = word if not current else current + " " + word
            bbox = draw.textbbox((0, 0), trial, font=font, stroke_width=stroke_width)
            if (bbox[2] - bbox[0]) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _normalize_character_name(self, name: str) -> str:
        """Clean and format character name for display."""
        if not name:
            return ""
        cleaned = re.sub(r"[^A-Za-z0-9\s]+", " ", name)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return " ".join(part.capitalize() for part in cleaned.split(" "))

    def _overlay_character_name(self, image_path: str, character_name: str) -> bool:
        """Add character name overlay at the bottom of the image with text wrapping support."""
        try:
            img = Image.open(image_path).convert("RGBA")
            w, h = img.size
            
            # Prepare text
            name_text = self._normalize_character_name(character_name).upper()
            if not name_text:
                return True  # No text to overlay
            
            # Dynamic sizing and stroke
            scale = float(CHARACTER_NAME_FONT_SCALE)
            stroke_w = max(1, int(h * 0.002 * scale))  # Stroke width based on image size
            base_font_size = max(12, int(h * 0.025 * scale))  # 2.5% of height
            min_font_size = max(8, int(base_font_size * 0.6))
            padding = int(h * 0.01 * max(1.0, min(scale, 2.0)))  # Padding based on scale
            max_text_width = int(w * 0.90)  # 90% of image width for text
            
            # Find suitable font size with text wrapping
            chosen_font = None
            lines = None
            line_height = None
            block_height = None
            
            dummy_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(dummy_img)
            
            for size in range(base_font_size, min_font_size - 1, -1):
                font = self._load_font(size)
                lines_try = self._wrap_lines(draw, name_text, font, max_text_width, stroke_width=stroke_w)
                # Measure line height with stroke
                sample_bbox = draw.textbbox((0, 0), "Ag", font=font, stroke_width=stroke_w)
                lh = (sample_bbox[3] - sample_bbox[1]) + int(h * 0.005)
                bh = len(lines_try) * lh + padding * 2
                # Limit to 3 lines max and reasonable band height
                if len(lines_try) <= 3 and bh <= int(h * 0.5):  # Max 50% of image height
                    chosen_font = font
                    lines = lines_try
                    line_height = lh
                    block_height = bh
                    break
            
            # Fallback to minimum size if no suitable size found
            if chosen_font is None:
                chosen_font = self._load_font(max(min_font_size, 1))
                lines = self._wrap_lines(draw, name_text, chosen_font, max_text_width, stroke_width=stroke_w)
                sample_bbox = draw.textbbox((0, 0), "Ag", font=chosen_font, stroke_width=stroke_w)
                line_height = (sample_bbox[3] - sample_bbox[1]) + int(h * 0.005)
                block_height = len(lines) * line_height + padding * 2
            
            # Calculate final band height
            band_height = min(block_height, int(h * CHARACTER_NAME_BAND_HEIGHT_RATIO))
            if band_height < 20:  # Minimum band height
                band_height = 20
            
            # Create band and text layers
            band_opacity = int(0.70 * 255)  # Semi-transparent dark band
            band = Image.new("RGBA", (w, band_height), (0, 0, 0, band_opacity))
            
            # Add slight blur to band for better text readability
            shadow = band.copy().filter(ImageFilter.GaussianBlur(radius=1))
            
            text_layer = Image.new("RGBA", (w, band_height), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_layer)
            
            # Draw multi-line text with stroke
            y_cursor = padding
            for line in lines:
                bbox = text_draw.textbbox((0, 0), line, font=chosen_font, stroke_width=stroke_w)
                tw = bbox[2] - bbox[0]
                x = (w - tw) // 2
                # Draw text with stroke for better visibility
                text_draw.text((x, y_cursor), line, font=chosen_font, fill=(255, 255, 255, 255), stroke_width=stroke_w, stroke_fill=(0, 0, 0, 128))
                y_cursor += line_height
            
            # Composite layers onto original image
            # Position band at bottom
            band_y = h - band_height
            
            # Add shadow first
            shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            shadow_layer.paste(shadow, (0, band_y), shadow)
            img = Image.alpha_composite(img, shadow_layer)
            
            # Add band
            band_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            band_layer.paste(band, (0, band_y), band)
            img = Image.alpha_composite(img, band_layer)
            
            # Add text
            text_overlay_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            text_overlay_layer.paste(text_layer, (0, band_y), text_layer)
            img = Image.alpha_composite(img, text_overlay_layer)
            
            # Save back to file
            if image_path.lower().endswith((".jpg", ".jpeg")):
                img.convert("RGB").save(image_path, quality=95)
            else:
                img.save(image_path)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to overlay character name '{character_name}': {e}")
            return False

    def _get_completed_characters(self) -> set[str]:
        """Get character names that have already been generated."""
        if not os.path.exists(self.final_output_dir):
            return set()
        return {f[:-4] for f in os.listdir(self.final_output_dir) if f.endswith('.png')}

    def generate_all_characters(self, force_regenerate: bool = False, resumable_state=None) -> dict[str, str]:
        """Generate images for all characters.
        
        Returns:
            dict: {character_name: output_path} mapping of successful generations
        """
        characters = self._read_character_data()
        if not characters:
            print("ERROR: No character data found")
            return {}

        # Use resumable state if available, otherwise fall back to file-based checking
        if resumable_state:
            # Run precheck to validate file existence and clean up invalid entries
            cleaned_count = resumable_state.validate_and_cleanup_results()
            if cleaned_count > 0:
                print(f"Precheck completed: {cleaned_count} invalid entries removed from checkpoint")
            
            completed_characters = set()
            for char_name in characters.keys():
                if resumable_state.is_character_complete(char_name):
                    completed_characters.add(char_name)
        else:
            completed_characters = self._get_completed_characters()
        
        if not force_regenerate and completed_characters:
            print(f"Found {len(completed_characters)} completed characters: {sorted(completed_characters)}")

        characters_to_process = {name: desc for name, desc in characters.items() 
                               if force_regenerate or name not in completed_characters}

        if not characters_to_process:
            print("All characters already generated!")
            return {}

        print(f"Processing {len(characters_to_process)} characters, skipped {len(completed_characters)}")
        print("=" * 60)

        results = {}
        for i, (character_name, description) in enumerate(characters_to_process.items(), 1):
            print(f"\n[{i}/{len(characters_to_process)}] Processing {character_name}...")
            output_path = self._generate_character_image(character_name, description, resumable_state)
            if output_path:
                results[character_name] = output_path
                print(f"[OK] Generated: {character_name}")
            else:
                print(f"[FAILED] {character_name}")

        return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate character images using flux (default) or diffusion workflow.")
    parser.add_argument("--mode", "-m", choices=["flux", "diffusion"], default="flux", help="Select workflow: flux (default) or diffusion")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration of all characters")
    parser.add_argument("--list-completed", "-l", action="store_true", help="List completed characters")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")
    args = parser.parse_args()
    
    generator = CharacterGenerator(mode=args.mode)
    
    if args.list_completed:
        completed = generator._get_completed_characters()
        print(f"Completed characters ({len(completed)}): {sorted(completed)}" if completed else "No completed characters")
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
    results = generator.generate_all_characters(force_regenerate=args.force, resumable_state=resumable_state)
    elapsed = time.time() - start_time
    
    if results:
        print(f"\nGenerated {len(results)} character images in {elapsed:.2f}s using {args.mode} mode:")
        for name, path in results.items():
            print(f"  {name}: {path}")
        
        # Clean up checkpoint files if resumable mode was used and everything completed successfully
        if resumable_state:
            print("All operations completed successfully")
            print("Final progress:", resumable_state.get_progress_summary())
            resumable_state.cleanup()
        
        return 0
    else:
        print("No new character images generated")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
