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
WORKFLOW_SUMMARY_ENABLED = False  # Set to True to enable workflow summary printing

# Variation Configuration
VARIATIONS_PER_LOCATION = 1  # Number of variations to generate per location (in addition to original)

# Image Resolution Constants
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Latent Input Mode Configuration
LATENT_MODE = "LATENT"  # "LATENT" for normal noise generation, "IMAGE" for load image input
IMAGE_LATENT_SIZE = "large"
LATENT_DENOISING_STRENGTH = 0.82  # Denoising strength when using IMAGE mode (0.0-1.0, higher = more change)

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
        "denoising_strength": 1,  # Denoising strength (0.0 - 1.0) (serial mode only)
        "save_intermediate": True, # Save intermediate results for debugging (serial mode only)
        "use_only_intermediate": False # Set to True to disable character images and use only intermediate result
    },
    {
        "name": "FLUX.1-Turbo-Alpha.safetensors",
        "strength_model": 3.6,    # Model strength (0.0 - 2.0)
        "strength_clip": 3.6,     # CLIP strength (0.0 - 2.0)
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
SAMPLING_STEPS = 25  # Number of sampling steps (higher = better quality, slower)

# Negative Prompt Configuration
USE_NEGATIVE_PROMPT = False  # Set to True to enable negative prompts, False to disable
NEGATIVE_PROMPT = "blur, distorted, text, watermark, extra limbs, bad anatomy, poorly drawn, asymmetrical, malformed, disfigured, ugly, bad proportions, plastic texture, artificial looking, cross-eyed, missing fingers, extra fingers, bad teeth, missing teeth, unrealistic"

# Random Seed Configuration
USE_RANDOM_SEED = True  # Set to True to use random seeds, False to use fixed seed
FIXED_SEED = 333555666  # Fixed seed value when USE_RANDOM_SEED is False

ART_STYLE = "Realistic Anime"

# Text overlay settings for character names
USE_CHARACTER_NAME_OVERLAY = False  # Set to False to disable name overlay
CHARACTER_NAME_FONT_SCALE = 1
CHARACTER_NAME_BAND_HEIGHT_RATIO = 0.30  # 15% of image height for name band

USE_SUMMARY_TEXT = False  # Set to True to use summary text


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
            "locations": {
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
    
    def is_location_complete(self, location_name: str) -> bool:
        """Check if location generation is complete (including all variations)."""
        if location_name not in self.state["locations"]["results"]:
            return False
        
        result = self.state["locations"]["results"][location_name]
        variations = result.get("variations", {})
        
        # Check if original exists
        if not result.get("path") or not os.path.exists(result.get("path", "")):
            return False
        
        # Check if all variations exist
        expected_variations = VARIATIONS_PER_LOCATION
        for i in range(1, expected_variations + 1):
            var_key = f"v{i}"
            if var_key not in variations or not os.path.exists(variations[var_key].get("path", "")):
                return False
        
        return True
    
    def is_location_original_complete(self, location_name: str) -> bool:
        """Check if the original location image is complete."""
        if location_name not in self.state["locations"]["results"]:
            return False
        
        result = self.state["locations"]["results"][location_name]
        return result.get("path") and os.path.exists(result.get("path", ""))
    
    def is_location_variation_complete(self, location_name: str, variation_suffix: str) -> bool:
        """Check if a specific variation is complete."""
        if location_name not in self.state["locations"]["results"]:
            return False
        
        result = self.state["locations"]["results"][location_name]
        variations = result.get("variations", {})
        
        if variation_suffix not in variations:
            return False
        
        return os.path.exists(variations[variation_suffix].get("path", ""))
    
    def get_location_result(self, location_name: str) -> dict:
        """Get location generation result."""
        return self.state["locations"]["results"].get(location_name, {})
    
    def set_location_result(self, location_name: str, result: dict):
        """Set location generation result and mark as complete."""
        self.state["locations"]["results"][location_name] = result
        if location_name not in self.state["locations"]["completed"]:
            self.state["locations"]["completed"].append(location_name)
        self._save_state()
    
    def set_location_variation(self, location_name: str, variation_key: str, variation_result: dict):
        """Set a specific variation result for a location."""
        if location_name not in self.state["locations"]["results"]:
            self.state["locations"]["results"][location_name] = {"variations": {}}
        
        if "variations" not in self.state["locations"]["results"][location_name]:
            self.state["locations"]["results"][location_name]["variations"] = {}
        
        self.state["locations"]["results"][location_name]["variations"][variation_key] = variation_result
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
    
    def validate_and_cleanup_results(self, output_characters_dir: str = None) -> int:
        """Validate that all completed character files actually exist and clean up missing entries.
        
        Args:
            output_characters_dir: Path to the output/characters directory to check for actual files
        
        Returns:
            int: Number of entries cleaned up (removed from completed list)
        """
        cleaned_count = 0
        characters_to_remove = []
        
        print(f"Validating {len(self.state['locations']['completed'])} completed locations against output/locations directory...")
        
        # Check each completed location
        for location_name in self.state["locations"]["completed"]:
            result = self.state["locations"]["results"].get(location_name, {})
            file_path = result.get('path', '')
            
            # Check if file actually exists
            main_exists = file_path and os.path.exists(file_path)
            
            if not main_exists:
                print(f"Precheck: File missing for {location_name} - marking as not completed")
                print(f"  Main file exists: {main_exists} ({file_path})")
                characters_to_remove.append(location_name)
                cleaned_count += 1
            elif output_characters_dir:
                # Additional check: verify file exists in output/locations directory
                expected_location_file = os.path.join(output_characters_dir, f"{location_name}.png")
                if not os.path.exists(expected_location_file):
                    print(f"Precheck: Location file missing in output/locations directory for {location_name} - marking as not completed")
                    print(f"  Expected: {expected_location_file}")
                    characters_to_remove.append(location_name)
                    cleaned_count += 1
                else:
                    print(f"Precheck: ✓ {location_name} validated in output/locations directory")
        
        # Remove invalid entries
        for location_name in characters_to_remove:
            if location_name in self.state["locations"]["completed"]:
                self.state["locations"]["completed"].remove(location_name)
            if location_name in self.state["locations"]["results"]:
                del self.state["locations"]["results"][location_name]
            
            # Also clear any LoRA progress for this location (including all variations)
            if "lora_progress" in self.state:
                # Clear base location progress
                base_key = f"{location_name}_lora_progress"
                if base_key in self.state["lora_progress"]:
                    del self.state["lora_progress"][base_key]
                    print(f"Precheck: Cleared LoRA progress for {location_name}")
                
                # Clear all variation progress
                keys_to_remove = [k for k in self.state["lora_progress"].keys() if k.startswith(f"{location_name}_v") and k.endswith("_lora_progress")]
                for key in keys_to_remove:
                    del self.state["lora_progress"][key]
                    print(f"Precheck: Cleared LoRA progress for {key}")
        
        # Save cleaned state if any changes were made
        if cleaned_count > 0:
            self._save_state()
            print(f"Precheck: Cleaned up {cleaned_count} invalid entries from checkpoint")
        
        return cleaned_count
    
    def sync_with_output_directory(self, output_characters_dir: str) -> int:
        """Sync resumable state with actual files in output directory.
        
        This method finds files that exist in the output directory but aren't tracked
        in the resumable state, and adds them to the completed list.
        
        Args:
            output_characters_dir: Path to the output/characters directory to check for actual files
        
        Returns:
            int: Number of files found and added to completed list
        """
        if not os.path.exists(output_characters_dir):
            print(f"Output/characters directory does not exist: {output_characters_dir}")
            return 0
            
        added_count = 0
        tracked_locations = set(self.state["locations"]["completed"])
        
        print(f"Scanning output/locations directory for untracked files: {output_characters_dir}")
        
        # Find all .png files in the output directory
        for filename in os.listdir(output_characters_dir):
            if filename.endswith('.png'):
                # Extract location name from filename (remove .png extension)
                location_name = filename[:-4]
                
                # If this location isn't tracked, add it to completed
                if location_name not in tracked_locations:
                    file_path = os.path.join(output_characters_dir, filename)
                    result = {
                        'path': file_path,
                        'location_name': location_name,
                        'auto_detected': True
                    }
                    self.state["locations"]["results"][location_name] = result
                    self.state["locations"]["completed"].append(location_name)
                    added_count += 1
                    print(f"Auto-detected completed location: {location_name} -> {file_path}")
                else:
                    print(f"Location already tracked: {location_name}")
        
        # Save state if any files were added
        if added_count > 0:
            self._save_state()
            print(f"Auto-detection: Added {added_count} locations from output/locations directory")
        else:
            print("No untracked location files found in output/locations directory")
        
        return added_count
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        completed = len(self.state["locations"]["completed"])
        total = len(self.state["locations"]["results"]) + len([k for k in self.state["locations"]["results"].keys() if k not in self.state["locations"]["completed"]])
        
        return f"Progress: Locations({completed}/{total})"


class LocationGenerator:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188/"):
        self.comfyui_url = comfyui_url
        # ComfyUI saves images under this folder
        self.comfyui_output_folder = "../../ComfyUI/output"
        # Final destination inside this repo
        self.final_output_dir = "../output/locations"
        self.intermediate_output_dir = "../output/lora"
        self.input_file = "../input/3.location.txt" if USE_SUMMARY_TEXT else "../input/2.location.txt"
        # Latent image input file path
        self.latent_image_path = f"../input/2.latent.location.{IMAGE_LATENT_SIZE}.png"
        # Always use Flux workflow - use character workflow
        self.workflow_file = "../workflow/character_location.json"

        # Create output directories
        os.makedirs(self.final_output_dir, exist_ok=True)
        os.makedirs(self.intermediate_output_dir, exist_ok=True)

    def _read_location_data(self) -> dict[str, str]:
        """Parse location data from input file.
        
        Returns:
            dict: {location_name: description} mapping
        """
        locations = {}
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
            # Split by single newlines to separate location entries
            entries = [entry.strip() for entry in content.split('\n') if entry.strip()]
            
            for entry in entries:
                # Match pattern: {{loc_X}}: Description
                match = re.match(r'\{\{([^}]+)\}\}\s*(.+)', entry, re.DOTALL)
                if match:
                    location_name = match.group(1).strip()
                    description = match.group(2).strip()
                    locations[location_name] = description
                    
        except Exception as e:
            print(f"ERROR: Failed to read location data: {e}")
            
        return locations

    def _load_location_workflow(self) -> dict:
        """Load the location generation workflow and apply settings."""
        try:
            with open(self.workflow_file, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            
            print(f"Loaded Flux workflow from: {self.workflow_file}")
            
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
            print("LoRA enabled in location workflow")
        else:
            # Remove all LoRA nodes if they exist
            self._remove_all_lora_nodes(workflow)
            print("LoRA disabled in location workflow")
        
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
        """Apply LoRAs in chained mode - all LoRAs stitched together in single workflow.
        
        Chained mode logic:
        - All LoRAs are connected in a chain: Model -> LoRA1 -> LoRA2 -> LoRA3 -> KSampler
        - Single workflow execution with all LoRAs applied together
        - For IMAGE mode: EmptySD3LatentImage replaced with image input + LATENT_DENOISING_STRENGTH
        - For LATENT mode: EmptySD3LatentImage remains with width/height settings
        """
        enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
        
        if not enabled_loras:
            print("No enabled LoRAs found in LORAS configuration")
            return
        
        print(f"Applying {len(enabled_loras)} LoRAs in chained mode...")
        print("All LoRAs will be stitched together in a single workflow execution")
        
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
        """Apply LoRAs in serial mode - each LoRA runs independently.
        
        Serial mode logic:
        - Each LoRA runs in a separate workflow execution
        - LoRA 0: Uses EmptySD3LatentImage (latent mode) or image input (image mode)
        - LoRA 1+: Uses previous LoRA output as input
        - Each LoRA uses its own denoising_strength setting
        """
        enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
        
        if not enabled_loras:
            print("No enabled LoRAs found in LORAS configuration")
            return
        
        print(f"Serial LoRA mode: {len(enabled_loras)} LoRAs will run independently")
        print("Note: Serial mode requires separate workflow execution for each LoRA")
        print("Each LoRA will use its own denoising_strength setting")
    
    def _remove_all_lora_nodes(self, workflow: dict) -> None:
        """Remove all LoRA nodes from workflow."""
        # Remove LoRA nodes (lora_1, lora_2, etc.)
        nodes_to_remove = []
        for node_id in workflow.keys():
            if node_id.startswith("lora"):
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

    def _update_workflow_prompt(self, workflow: dict, location_name: str, description: str) -> dict:
        """Update the workflow with location-specific prompt."""
        prompt = f"Create a 16K ultra-high-resolution, detailed location environment illustration in the style of {ART_STYLE}. The scene should be atmospheric, immersive, and precisely matching the location description with fine-level detailing, rich color palette, proper lighting, depth of field, and environmental storytelling. No characters should be visible in this location scene.\n\n Location Description = {description}. Strictly, Accurately, Precisely, always must Follow {ART_STYLE} Style for environmental art.All Colourings, Styles, Shapes, Textures, Lightings and Detailing, must be **exactly same/ identical/as it is** in the scene text-description"
        self._update_node_connections(workflow, ["CLIPTextEncode", "CLIP Text Encode (Prompt)"], "text", prompt)
        return workflow

    def _update_workflow_filename(self, workflow: dict, location_name: str, variation_suffix: str = "") -> dict:
        """Update the workflow to save with location name as filename."""
        clean_name = re.sub(r'[^\w\s.-]', '', location_name).strip()
        clean_name = re.sub(r'[-\s]+', '_', clean_name)
        
        if variation_suffix:
            filename = f"{clean_name}_{variation_suffix}"
        else:
            filename = clean_name
            
        self._update_node_connections(workflow, "SaveImage", "filename_prefix", filename)
        return workflow

    def _update_workflow_seed(self, workflow: dict, seed: int = None, variation_number: int = 0) -> dict:
        """Update the workflow with a seed based on configuration."""
        if seed is None:
            if USE_RANDOM_SEED:
                # Generate different seed for each variation
                base_seed = random.randint(1, 2**32 - 1)
                seed = base_seed + variation_number
            else:
                # Use fixed seed with variation offset
                seed = FIXED_SEED + variation_number
        self._update_node_connections(workflow, "KSampler", "seed", seed)
        return workflow

    def _update_workflow_resolution(self, workflow: dict) -> dict:
        """Update the workflow with resolution settings and handle latent input mode.

        Logic:
        - For chained mode: Apply latent/image logic based on LATENT_MODE
        - For serial mode: This method is called for each LoRA individually
        """
        # Handle EmptySD3LatentImage node
        latent_image_node = self._find_node_by_class(workflow, "EmptySD3LatentImage")
        if latent_image_node:
            node_id = latent_image_node[0]
            
            if LATENT_MODE == "IMAGE":
                # For chained mode: Replace EmptySD3LatentImage with LoadImage + VAEEncode
                # For serial mode: This will be handled individually in _generate_character_image_serial
                if LORA_MODE == "chained" or not USE_LORA:
                    self._replace_latent_with_image_input(workflow, node_id, self.latent_image_path, LATENT_DENOISING_STRENGTH)
                    print(f"Using image input mode with file: {self.latent_image_path}")
                else:
                    # Serial mode: Just set dimensions, individual LoRA handling will replace this
                    workflow[node_id]["inputs"]["width"] = IMAGE_WIDTH
                    workflow[node_id]["inputs"]["height"] = IMAGE_HEIGHT
                    print(f"Serial mode: Set latent dimensions, individual LoRA handling will replace with image input")
            else:
                # Normal latent mode - set dimensions
                workflow[node_id]["inputs"]["width"] = IMAGE_WIDTH
                workflow[node_id]["inputs"]["height"] = IMAGE_HEIGHT
                print(f"Using latent mode with dimensions: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        
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
                        final_denoising = denoising_strength if denoising_strength is not None else LATENT_DENOISING_STRENGTH
                        sampler_node["inputs"]["denoise"] = final_denoising
                        break
            
            # Remove the original EmptySD3LatentImage node
            del workflow[latent_node_id]
            
            print(f"  Replaced EmptySD3LatentImage with LoadImage + VAEEncode")
            print(f"  LoadImage node: {load_image_node_id}, VAEEncode node: {encode_node_id}")
            print(f"  Denoising strength set to: {final_denoising}")
            
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
                        # Use provided denoising strength or keep existing
                        if denoising_strength is not None:
                            sampler_node["inputs"]["denoise"] = denoising_strength
                        break
            
            # Remove the original EmptySD3LatentImage node
            if "19" in workflow:
                del workflow["19"]
            
            print(f"  Replaced EmptySD3LatentImage with LoadImage + VAEEncode")
            print(f"  LoadImage node: {load_image_node_id}, VAEEncode node: {encode_node_id}")
            if denoising_strength is not None:
                print(f"  Denoising strength set to: {denoising_strength}")
            
        except Exception as e:
            print(f"WARNING: Failed to replace latent with previous output: {e}")

    def _generate_location_image_serial(self, location_name: str, description: str, resumable_state=None, variation_suffix: str = "") -> str | None:
        """Generate location image using serial LoRA mode with intermediate storage."""
        try:
            # Check if location file already exists
            if variation_suffix:
                location_filename = f"{location_name}_{variation_suffix}.png"
            else:
                location_filename = f"{location_name}.png"
            location_output_path = os.path.join(self.final_output_dir, location_filename)
            if os.path.exists(location_output_path):
                print(f"⏭️  Skipping location - File already exists: {location_filename}")
                return location_output_path
            
            # Check if resumable and already complete
            if resumable_state:
                if variation_suffix:
                    # Check if this specific variation is complete
                    if resumable_state.is_location_variation_complete(location_name, variation_suffix):
                        result = resumable_state.get_location_result(location_name)
                        variations = result.get("variations", {})
                        if variation_suffix in variations and os.path.exists(variations[variation_suffix].get("path", "")):
                            print(f"Using cached variation: {location_name}_{variation_suffix}")
                            return variations[variation_suffix]["path"]
                        else:
                            print(f"Cached variation file missing, regenerating: {location_name}_{variation_suffix}")
                else:
                    # Check if original is complete
                    if resumable_state.is_location_original_complete(location_name):
                        cached_result = resumable_state.get_location_result(location_name)
                        if cached_result and os.path.exists(cached_result.get('path', '')):
                            print(f"Using cached location image: {location_name}")
                            return cached_result['path']
                        elif cached_result:
                            print(f"Cached file missing, regenerating: {location_name}")
            
            display_name = f"{location_name}_{variation_suffix}" if variation_suffix else location_name
            print(f"Generating image for: {display_name} (Serial LoRA mode)")
            
            enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
            if not enabled_loras:
                print("ERROR: No enabled LoRAs found for serial mode")
                return None
            
            # Clean location name for filenames (preserve dots for version numbers like 1.1)
            clean_name = re.sub(r'[^\w\s.-]', '', location_name).strip()
            clean_name = re.sub(r'[-\s]+', '_', clean_name)
            
            # Add variation suffix to clean name if provided
            if variation_suffix:
                clean_name = f"{clean_name}_{variation_suffix}"
            
            # Check for existing LoRA progress (include variation suffix for unique tracking)
            lora_progress_key = f"{location_name}{'_' + variation_suffix if variation_suffix else ''}_lora_progress"
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
                
                # Load base workflow for this LoRA
                workflow = self._load_location_workflow()
                if not workflow:
                    print(f"ERROR: Failed to load workflow for LoRA {i + 1}")
                    continue
                
                # Apply only this LoRA to the workflow
                self._apply_single_lora(workflow, lora_config, i + 1)
                
                # Update workflow with location-specific settings
                workflow = self._update_workflow_prompt(workflow, location_name, description)
                # Determine variation number for seed generation (0 = original, 1+ = variations)
                variation_num = 0 if not variation_suffix else int(variation_suffix.replace('v', ''))
                workflow = self._update_workflow_seed(workflow, variation_number=variation_num)
                workflow = self._update_workflow_resolution(workflow)
                
                # Append dots to prompt based on LoRA position (LoRA 0 = no dots, LoRA 1 = 1 dot, etc.)
                if i > 0:  # Skip first LoRA (index 0)
                    dots = "." * i
                    # Find the CLIPTextEncode node and append dots to its text
                    for node_id, node in workflow.items():
                        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                            if "text" in node.get("inputs", {}):
                                current_prompt = node["inputs"]["text"]
                                node["inputs"]["text"] = current_prompt + dots
                                print(f"  Added {i} dots to prompt: {dots}")
                                break
                
                # Set LoRA-specific sampling steps
                steps = lora_config.get("steps", SAMPLING_STEPS)
                denoising_strength = lora_config.get("denoising_strength", 1.0)
                self._update_node_connections(workflow, "KSampler", "steps", steps)
                
                # Handle input for this LoRA based on serial mode logic
                if i == 0:
                    # First LoRA: Use latent/image mode based on LATENT_MODE setting
                    if LATENT_MODE == "IMAGE":
                        # Replace EmptySD3LatentImage with image input + LATENT_DENOISING_STRENGTH
                        self._replace_latent_with_image_input(workflow, "19", self.latent_image_path, LATENT_DENOISING_STRENGTH)
                        # Apply LATENT_DENOISING_STRENGTH to KSampler for first LoRA in IMAGE mode
                        self._update_node_connections(workflow, "KSampler", "denoise", LATENT_DENOISING_STRENGTH)
                        denoising_strength = LATENT_DENOISING_STRENGTH
                        print(f"  Using image input mode for first LoRA with file: {self.latent_image_path}")
                        print(f"  Using LATENT_DENOISING_STRENGTH: {LATENT_DENOISING_STRENGTH}")
                    else:
                        # Keep EmptySD3LatentImage with width/height, use LoRA's denoising_strength
                        self._update_node_connections(workflow, "KSampler", "denoise", denoising_strength)
                        print(f"  Using latent mode for first LoRA with dimensions: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
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
                
                # Generate filename for this LoRA step
                lora_clean_name = re.sub(r'[^\w\s-]', '', lora_config['name']).strip()
                lora_clean_name = re.sub(r'[-\s]+', '_', lora_clean_name)
                lora_filename = f"{clean_name}.{lora_clean_name}"
                self._update_node_connections(workflow, "SaveImage", "filename_prefix", lora_filename)
                
                print(f"  Steps: {steps}, Denoising: {denoising_strength}")
                
                # Print workflow summary before sending
                self._print_workflow_summary(workflow, f"LoRA {i + 1}: {lora_name}")
                
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
                print(f"ERROR: No successful LoRA generations for {location_name}")
                return None
            
            # Copy final result to output directory
            if variation_suffix:
                final_path = os.path.join(self.final_output_dir, f"{location_name}_{variation_suffix}.png")
            else:
                final_path = os.path.join(self.final_output_dir, f"{location_name}.png")
            shutil.copy2(current_image_path, final_path)
            
            # Apply location name overlay if enabled
            if USE_CHARACTER_NAME_OVERLAY:
                print(f"Adding location name overlay...")
                overlay_success = self._overlay_character_name(final_path, location_name)
                if overlay_success:
                    print(f"Saved with name overlay: {final_path}")
                else:
                    print(f"Saved (overlay failed): {final_path}")
            else:
                print(f"Saved: {final_path}")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                if variation_suffix:
                    # Save as variation
                    variation_result = {
                        'path': final_path,
                        'location_name': location_name,
                        'description': description,
                        'variation_suffix': variation_suffix,
                        'intermediate_paths': intermediate_paths
                    }
                    resumable_state.set_location_variation(location_name, variation_suffix, variation_result)
                else:
                    # Save as main result
                    result = {
                        'path': final_path,
                        'location_name': location_name,
                        'description': description,
                        'intermediate_paths': intermediate_paths
                    }
                    resumable_state.set_location_result(location_name, result)
                
                # Keep LoRA progress for completed location (not cleaning up)
                print(f"  Preserved LoRA progress for completed location")
            
            return final_path

        except Exception as e:
            print(f"ERROR: Failed to generate image for {location_name}: {e}")
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

    def _generate_location_image(self, location_name: str, description: str, resumable_state=None, variation_suffix: str = "") -> str | None:
        """Generate a single location image using ComfyUI."""
        try:
            # Check if resumable and already complete
            if resumable_state:
                if variation_suffix:
                    # Check if this specific variation is complete
                    if resumable_state.is_location_variation_complete(location_name, variation_suffix):
                        result = resumable_state.get_location_result(location_name)
                        variations = result.get("variations", {})
                        if variation_suffix in variations and os.path.exists(variations[variation_suffix].get("path", "")):
                            print(f"Using cached variation: {location_name}_{variation_suffix}")
                            return variations[variation_suffix]["path"]
                        else:
                            print(f"Cached variation file missing, regenerating: {location_name}_{variation_suffix}")
                else:
                    # Check if original is complete
                    if resumable_state.is_location_original_complete(location_name):
                        cached_result = resumable_state.get_location_result(location_name)
                        if cached_result and os.path.exists(cached_result.get('path', '')):
                            print(f"Using cached location image: {location_name}")
                            return cached_result['path']
                        elif cached_result:
                            print(f"Cached file missing, regenerating: {location_name}")
            
            # Use serial LoRA mode if enabled
            if USE_LORA and LORA_MODE == "serial":
                return self._generate_location_image_serial(location_name, description, resumable_state, variation_suffix)
            
            display_name = f"{location_name}_{variation_suffix}" if variation_suffix else location_name
            print(f"Generating image for: {display_name}")
            
            # Load and update workflow
            workflow = self._load_location_workflow()
            if not workflow:
                return None
                
            workflow = self._update_workflow_prompt(workflow, location_name, description)
            workflow = self._update_workflow_filename(workflow, location_name, variation_suffix)
            # Determine variation number for seed generation (0 = original, 1+ = variations)
            variation_num = 0 if not variation_suffix else int(variation_suffix.replace('v', ''))
            workflow = self._update_workflow_seed(workflow, variation_number=variation_num)
            workflow = self._update_workflow_resolution(workflow)

            # Print workflow summary
            self._print_workflow_summary(workflow, f"Location: {location_name}")
            
            # Print prompt before sending
            print(f"\n=== PROMPT FOR LOCATION: {location_name} ===")
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
            search_prefix = f"{location_name}_{variation_suffix}" if variation_suffix else location_name
            generated_image = self._find_newest_output_with_prefix(search_prefix)
            if not generated_image:
                print(f"ERROR: Could not find generated image for {display_name}")
                return None

            # Copy to final output directory
            if variation_suffix:
                final_path = os.path.join(self.final_output_dir, f"{location_name}_{variation_suffix}.png")
            else:
                final_path = os.path.join(self.final_output_dir, f"{location_name}.png")
            shutil.copy2(generated_image, final_path)
            
            # Apply location name overlay if enabled
            if USE_CHARACTER_NAME_OVERLAY:
                print(f"Adding location name overlay...")
                overlay_success = self._overlay_character_name(final_path, location_name)
                if overlay_success:
                    print(f"Saved with name overlay: {final_path}")
                else:
                    print(f"Saved (overlay failed): {final_path}")
            else:
                print(f"Saved: {final_path}")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                if variation_suffix:
                    # Save as variation
                    variation_result = {
                        'path': final_path,
                        'location_name': location_name,
                        'description': description,
                        'variation_suffix': variation_suffix
                    }
                    resumable_state.set_location_variation(location_name, variation_suffix, variation_result)
                else:
                    # Save as main result
                    result = {
                        'path': final_path,
                        'location_name': location_name,
                        'description': description
                    }
                    resumable_state.set_location_result(location_name, result)
            
            return final_path

        except Exception as e:
            print(f"ERROR: Failed to generate image for {location_name}: {e}")
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

    def _get_completed_locations(self) -> set[str]:
        """Get location names that have already been generated."""
        if not os.path.exists(self.final_output_dir):
            return set()
        return {f[:-4] for f in os.listdir(self.final_output_dir) if f.endswith('.png')}

    def generate_all_locations(self, force_regenerate: bool = False, resumable_state=None) -> dict[str, str]:
        """Generate images for all locations.
        
        Returns:
            dict: {location_name: output_path} mapping of successful generations
        """
        locations = self._read_location_data()
        if not locations:
            print("ERROR: No location data found")
            return {}

        # Use resumable state if available, otherwise fall back to file-based checking
        if resumable_state:
            print(f"Validating and syncing with output/locations directory: {self.final_output_dir}")
            
            # First, sync with output/locations directory to detect any manually added files
            synced_count = resumable_state.sync_with_output_directory(self.final_output_dir)
            if synced_count > 0:
                print(f"Sync completed: {synced_count} locations auto-detected from output/locations directory")
            
            # Then run precheck to validate file existence and clean up invalid entries
            cleaned_count = resumable_state.validate_and_cleanup_results(self.final_output_dir)
            if cleaned_count > 0:
                print(f"Precheck completed: {cleaned_count} invalid entries removed from checkpoint")
            
            completed_locations = set()
            for loc_name in locations.keys():
                if resumable_state.is_location_complete(loc_name):
                    completed_locations.add(loc_name)
        else:
            completed_locations = self._get_completed_locations()
        
        if not force_regenerate and completed_locations:
            print(f"Found {len(completed_locations)} fully completed locations: {sorted(completed_locations)}")

        locations_to_process = {name: desc for name, desc in locations.items() 
                               if force_regenerate or name not in completed_locations}

        if not locations_to_process:
            print("All locations already generated!")
            return {}

        print(f"Processing {len(locations_to_process)} locations with {VARIATIONS_PER_LOCATION} variations each, skipped {len(completed_locations)}")
        print("=" * 60)

        results = {}
        for i, (location_name, description) in enumerate(locations_to_process.items(), 1):
            print(f"\n[{i}/{len(locations_to_process)}] Processing {location_name}...")
            
            # Generate original image (only if not already complete)
            if not resumable_state or not resumable_state.is_location_original_complete(location_name):
                print(f"  Generating original image for {location_name}...")
                output_path = self._generate_location_image(location_name, description, resumable_state)
                if output_path:
                    results[location_name] = output_path
                    print(f"  [OK] Generated original: {location_name}")
                else:
                    print(f"  [FAILED] Original: {location_name}")
                    continue
            else:
                print(f"  [SKIP] Original already exists: {location_name}")
                # Get existing path for results
                if resumable_state:
                    result = resumable_state.get_location_result(location_name)
                    if result and result.get("path"):
                        results[location_name] = result["path"]
            
            # Generate variations (only missing ones)
            for v in range(1, VARIATIONS_PER_LOCATION + 1):
                variation_suffix = f"v{v}"
                
                # Check if this variation already exists
                if resumable_state and resumable_state.is_location_variation_complete(location_name, variation_suffix):
                    print(f"  [SKIP] Variation {v} already exists: {location_name}_{variation_suffix}")
                    continue
                
                print(f"  Generating variation {v}/{VARIATIONS_PER_LOCATION} for {location_name}...")
                var_output_path = self._generate_location_image(location_name, description, resumable_state, variation_suffix)
                if var_output_path:
                    print(f"  [OK] Generated variation {v}: {location_name}_{variation_suffix}")
                else:
                    print(f"  [FAILED] Variation {v}: {location_name}_{variation_suffix}")

        return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate location images using Flux workflow.")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration of all locations")
    parser.add_argument("--list-completed", "-l", action="store_true", help="List completed locations")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")
    args = parser.parse_args()
    
    generator = LocationGenerator()
    
    if args.list_completed:
        completed = generator._get_completed_locations()
        print(f"Completed locations ({len(completed)}): {sorted(completed)}" if completed else "No completed locations")
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
    results = generator.generate_all_locations(force_regenerate=args.force, resumable_state=resumable_state)
    elapsed = time.time() - start_time
    
    if results:
        print(f"\nGenerated {len(results)} location images in {elapsed:.2f}s using Flux mode:")
        for name, path in results.items():
            print(f"  {name}: {path}")
        
        # Clean up checkpoint files if resumable mode was used and everything completed successfully
        if resumable_state:
            print("All operations completed successfully")
            print("Final progress:", resumable_state.get_progress_summary())
            resumable_state.cleanup()
        
        return 0
    else:
        print("No new location images generated")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
