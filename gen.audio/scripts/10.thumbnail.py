import glob
import json
import os
import time
import shutil
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from functools import partial
import argparse
import builtins as _builtins
print = partial(_builtins.print, flush=True)
import re
import random

# Feature flags
ENABLE_RESUMABLE_MODE = True
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them

# Random seed configuration
USE_RANDOM_SEED = True  # Set to True to use random seeds for each generation
RANDOM_SEED = 333555666

# Workflow configuration
WORKFLOW_SUMMARY_ENABLED = False  # Set to True to enable workflow summary printing

# Controls text generation method:
# - True: use text overlay after image generation (generates 1 image: thumbnail.png)
# - False: let Flux generate text in the image itself (generates 5 versions: thumbnail.flux.v1-v5.png)
USE_TITLE_TEXT = True

# Controls where the title band + text appears: "top", "middle", or "bottom"
TITLE_POSITION = "middle"

# Controls title text scaling: 1 = default, 2 = 2x, 3 = 3x, any numeric
TITLE_FONT_SCALE = 1.5

# Controls layout strategy for the title band:
# - "overlay": draw band over the image (current behavior)
# - "expand": increase canvas height to place band outside the image (no overlap)
# - "fit": shrink the image to leave room for the band within the same canvas (no overlap)
TITLE_LAYOUT = "overlay"

# Target output canvas size
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720

# YouTube Shorts format (9:16 aspect ratio)
SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920

# Number of shorts variations to generate
SHORTS_VARIATIONS = 5

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
SAMPLING_STEPS = 25  # Number of sampling steps (higher = better quality, slower)

# Negative Prompt Configuration
USE_NEGATIVE_PROMPT = True  # Set to True to enable negative prompts, False to disable
NEGATIVE_PROMPT = "blur, distorted, text, watermark, extra limbs, bad anatomy, poorly drawn, asymmetrical, malformed, disfigured, ugly, bad proportions, plastic texture, artificial looking, cross-eyed, missing fingers, extra fingers, bad teeth, missing teeth, unrealistic"

ART_STYLE = "Realistic Anime"

class ResumableState:
    """Manages resumable state for expensive thumbnail generation operations."""
    
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
            "thumbnails": {
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
    
    def is_thumbnail_complete(self, thumbnail_key: str) -> bool:
        """Check if thumbnail generation is complete (including all variations)."""
        if thumbnail_key not in self.state["thumbnails"]["results"]:
            return False
        
        result = self.state["thumbnails"]["results"][thumbnail_key]
        variations = result.get("variations", {})
        
        # Check if original exists
        if not result.get("path") or not os.path.exists(result.get("path", "")):
            return False
        
        # Check if all variations exist
        expected_variations = 6  # 6 main variations (0-5)
        for i in range(1, expected_variations + 1):
            var_key = f"v{i}"
            if var_key not in variations or not os.path.exists(variations[var_key].get("path", "")):
                return False
        
        return True
    
    def is_thumbnail_original_complete(self, thumbnail_key: str) -> bool:
        """Check if the original thumbnail is complete."""
        if thumbnail_key not in self.state["thumbnails"]["results"]:
            return False
        
        result = self.state["thumbnails"]["results"][thumbnail_key]
        return result.get("path") and os.path.exists(result.get("path", ""))
    
    def is_thumbnail_variation_complete(self, thumbnail_key: str, variation_suffix: str) -> bool:
        """Check if a specific variation is complete."""
        if thumbnail_key not in self.state["thumbnails"]["results"]:
            return False
        
        result = self.state["thumbnails"]["results"][thumbnail_key]
        variations = result.get("variations", {})
        
        if variation_suffix not in variations:
            return False
        
        return os.path.exists(variations[variation_suffix].get("path", ""))
    
    def is_shorts_thumbnail_complete(self, shorts_thumbnail_key: str) -> bool:
        """Check if a specific shorts thumbnail is complete."""
        if shorts_thumbnail_key not in self.state["thumbnails"]["results"]:
            return False
        
        result = self.state["thumbnails"]["results"][shorts_thumbnail_key]
        return result.get("path") and os.path.exists(result.get("path", ""))
    
    def get_thumbnail_result(self, thumbnail_key: str) -> dict:
        """Get thumbnail generation result."""
        return self.state["thumbnails"]["results"].get(thumbnail_key, {})
    
    def set_thumbnail_result(self, thumbnail_key: str, result: dict):
        """Set thumbnail generation result and mark as complete."""
        self.state["thumbnails"]["results"][thumbnail_key] = result
        if thumbnail_key not in self.state["thumbnails"]["completed"]:
            self.state["thumbnails"]["completed"].append(thumbnail_key)
        self._save_state()
    
    def set_thumbnail_variation(self, thumbnail_key: str, variation_key: str, variation_result: dict):
        """Set a specific variation result for a thumbnail."""
        if thumbnail_key not in self.state["thumbnails"]["results"]:
            self.state["thumbnails"]["results"][thumbnail_key] = {"variations": {}}
        
        if "variations" not in self.state["thumbnails"]["results"][thumbnail_key]:
            self.state["thumbnails"]["results"][thumbnail_key]["variations"] = {}
        
        self.state["thumbnails"]["results"][thumbnail_key]["variations"][variation_key] = variation_result
        self._save_state()
    
    def set_shorts_thumbnail_result(self, shorts_thumbnail_key: str, result: dict):
        """Set shorts thumbnail generation result and mark as complete."""
        self.state["thumbnails"]["results"][shorts_thumbnail_key] = result
        if shorts_thumbnail_key not in self.state["thumbnails"]["completed"]:
            self.state["thumbnails"]["completed"].append(shorts_thumbnail_key)
        self._save_state()
    
    def cleanup_lora_progress(self, thumbnail_key: str):
        """Clean up LoRA progress for a completed thumbnail."""
        lora_progress_key = f"{thumbnail_key}_lora_progress"
        if "lora_progress" in self.state and lora_progress_key in self.state["lora_progress"]:
            del self.state["lora_progress"][lora_progress_key]
            if not self.state["lora_progress"]:
                del self.state["lora_progress"]
            self._save_state()
            print(f"Cleaned up LoRA progress for {lora_progress_key}")
    
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
    
    def validate_and_cleanup_results(self, output_dir: str = None) -> int:
        """Validate that all completed thumbnail files actually exist and clean up missing entries."""
        cleaned_count = 0
        thumbnails_to_remove = []
        
        print(f"Validating {len(self.state['thumbnails']['completed'])} completed thumbnails against output directory...")
        
        # Check each completed thumbnail
        for thumbnail_key in self.state["thumbnails"]["completed"]:
            result = self.state["thumbnails"]["results"].get(thumbnail_key, {})
            file_path = result.get('path', '')
            
            # Check if file actually exists
            main_exists = file_path and os.path.exists(file_path)
            
            if not main_exists:
                print(f"Precheck: File missing for {thumbnail_key} - marking as not completed")
                print(f"  Main file exists: {main_exists} ({file_path})")
                thumbnails_to_remove.append(thumbnail_key)
                cleaned_count += 1
            elif output_dir:
                # Additional check: verify file exists in output directory
                expected_thumbnail_file = os.path.join(output_dir, f"{thumbnail_key}.png")
                if not os.path.exists(expected_thumbnail_file):
                    print(f"Precheck: Thumbnail file missing in output directory for {thumbnail_key} - marking as not completed")
                    print(f"  Expected: {expected_thumbnail_file}")
                    thumbnails_to_remove.append(thumbnail_key)
                    cleaned_count += 1
                else:
                    print(f"Precheck: ✓ {thumbnail_key} validated in output directory")
        
        # Remove invalid entries
        for thumbnail_key in thumbnails_to_remove:
            if thumbnail_key in self.state["thumbnails"]["completed"]:
                self.state["thumbnails"]["completed"].remove(thumbnail_key)
            if thumbnail_key in self.state["thumbnails"]["results"]:
                del self.state["thumbnails"]["results"][thumbnail_key]
            
            # Also clear LoRA progress for this thumbnail
            lora_progress_key = f"{thumbnail_key}_lora_progress"
            if "lora_progress" in self.state and lora_progress_key in self.state["lora_progress"]:
                del self.state["lora_progress"][lora_progress_key]
                print(f"Precheck: Cleared LoRA progress for {thumbnail_key}")
        
        # Save cleaned state if any changes were made
        if cleaned_count > 0:
            self._save_state()
            print(f"Precheck: Cleaned up {cleaned_count} invalid entries from checkpoint")
        
        return cleaned_count
    
    def sync_with_output_directory(self, output_dir: str) -> int:
        """Sync resumable state with actual files in output directory."""
        if not os.path.exists(output_dir):
            print(f"Output directory does not exist: {output_dir}")
            return 0
            
        added_count = 0
        tracked_thumbnails = set(self.state["thumbnails"]["completed"])
        
        print(f"Scanning output directory for untracked files: {output_dir}")
        
        # Find all thumbnail-related files in the output directory
        for filename in os.listdir(output_dir):
            if filename.startswith('thumbnail') and filename.endswith('.png'):
                # Extract thumbnail key from filename
                thumbnail_key = filename[:-4]  # Remove .png extension
                
                # If this thumbnail isn't tracked, add it to completed
                if thumbnail_key not in tracked_thumbnails:
                    file_path = os.path.join(output_dir, filename)
                    result = {
                        'path': file_path,
                        'thumbnail_key': thumbnail_key,
                        'auto_detected': True
                    }
                    self.state["thumbnails"]["results"][thumbnail_key] = result
                    self.state["thumbnails"]["completed"].append(thumbnail_key)
                    added_count += 1
                    print(f"Auto-detected completed thumbnail: {thumbnail_key} -> {file_path}")
                else:
                    print(f"Thumbnail already tracked: {thumbnail_key}")
        
        # Save state if any files were added
        if added_count > 0:
            self._save_state()
            print(f"Auto-detection: Added {added_count} thumbnails from output directory")
        else:
            print("No untracked thumbnail files found in output directory")
        
        return added_count
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        completed = len(self.state["thumbnails"]["completed"])
        total = len(self.state["thumbnails"]["results"]) + len([k for k in self.state["thumbnails"]["results"].keys() if k not in self.state["thumbnails"]["completed"]])
        
        return f"Progress: Thumbnails({completed}/{total})"

class ThumbnailProcessor:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188/", mode: str = "diffusion"):
        self.comfyui_url = comfyui_url
        self.mode = (mode or "diffusion").strip().lower()
        # ComfyUI saves images under this folder
        self.comfyui_output_folder = "../../ComfyUI/output"
        # Final destination inside this repo
        self.final_output_dir = "../output"
        self.intermediate_output_dir = "../output/lora"
        # Latent image input file path
        self.latent_image_path = "../input/10.latent.png"

        os.makedirs(self.final_output_dir, exist_ok=True)
        os.makedirs(self.intermediate_output_dir, exist_ok=True)


        if not ENABLE_RESUMABLE_MODE:
            pattern = self.final_output_dir + "/*thumbnail*"
            files_to_remove = glob.glob(pattern, recursive=True)
            for file in files_to_remove:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Removed: {file}")

    def _load_thumbnail_workflow(self) -> dict:
        filename = "thumbnail.flux.json" if self.mode == "flux" else "thumbnail.json"
        workflow_path = Path(f"../workflow/{filename}")
        if not workflow_path.exists():
            raise FileNotFoundError(f"'../workflow/{filename}' not found.")
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        
        # Apply LoRA and sampling settings
        workflow = self._apply_workflow_settings(workflow)
        return workflow
    
    def _apply_workflow_settings(self, workflow: dict) -> dict:
        """Apply LoRA and sampling configuration to workflow."""
        # Apply LoRA settings
        if USE_LORA:
            # Handle multiple LoRAs in series
            self._apply_loras(workflow)
            print("LoRA enabled in thumbnail workflow")
        else:
            # Remove all LoRA nodes if they exist
            self._remove_all_lora_nodes(workflow)
            print("LoRA disabled in thumbnail workflow")
        
        # Apply sampling steps
        self._update_node_connections(workflow, "KSampler", "steps", SAMPLING_STEPS)
        print(f"Sampling steps set to: {SAMPLING_STEPS}")
        
        # Handle negative prompt
        if USE_NEGATIVE_PROMPT:
            # Create a new CLIPTextEncode node for negative prompt
            negative_node_id = "35"
            
            # Find the final CLIP output from LoRAs or use base CLIP
            final_clip_output = self._find_node_by_class(workflow, ["DualCLIPLoader", "TripleCLIPLoader"]) or ["10", 0]
            if USE_LORA:
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
        
        print(f"LoRAs chain completed with {len(enabled_loras)} LoRAs")
    
    def _remove_all_lora_nodes(self, workflow: dict) -> None:
        """Remove all LoRA nodes from workflow."""
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

    def _update_prompt_text(self, workflow: dict, prompt_text: str) -> dict:
        """Update the workflow with prompt text."""
        self._update_node_connections(workflow, ["CLIPTextEncode", "CLIP Text Encode (Prompt)"], "text", prompt_text)
        return workflow

    def _update_saveimage_prefix(self, workflow: dict, filename_prefix: str = "thumbnail") -> dict:
        """Update the workflow to save with specified filename prefix."""
        self._update_node_connections(workflow, "SaveImage", "filename_prefix", filename_prefix)
        # Also set extension to PNG if available
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == "SaveImage":
                if "extension" in node.get("inputs", {}):
                    node["inputs"]["extension"] = "png"
                break
        return workflow

    def _update_workflow_resolution(self, workflow: dict, width: int, height: int) -> dict:
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
                    workflow[node_id]["inputs"]["width"] = width
                    workflow[node_id]["inputs"]["height"] = height
                    print(f"Serial mode: Set latent dimensions, individual LoRA handling will replace with image input")
            else:
                # Normal latent mode - set dimensions
                workflow[node_id]["inputs"]["width"] = width
                workflow[node_id]["inputs"]["height"] = height
                print(f"Using latent mode with dimensions: {width}x{height}")
        
        return workflow

    def _get_seed(self, variation_number: int = 0) -> int:
        """Get seed value based on configuration."""
        if USE_RANDOM_SEED:
            # Generate different seed for each variation
            base_seed = random.randint(0, 2**32 - 1)
            return base_seed + variation_number
        else:
            # Use fixed seed with variation offset
            return RANDOM_SEED + variation_number

    def _update_workflow_seed(self, workflow: dict, seed: int, variation_number: int = 0) -> dict:
        """Set seed inputs across nodes when available."""
        # Update KSampler seed
        self._update_node_connections(workflow, "KSampler", "seed", int(seed))
        
        # Also try other common seed input names
        try:
            for node_id, node in workflow.items():
                if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
                    continue
                inputs = node["inputs"]
                for key in ("noise_seed", "noiseSeed"):
                    if key in inputs and not isinstance(inputs.get(key), (list, tuple)):
                        inputs[key] = int(seed)
        except Exception:
            pass  # Best-effort; ignore if structure unexpected
        
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

    def _generate_thumbnail_serial(self, prompt_text: str, shorts: bool = False, variation_number: str = "", resumable_state=None) -> str | None:
        """Generate thumbnail using serial LoRA mode with intermediate storage."""
        try:
            print(f"Generating thumbnail (Serial LoRA mode)")
            
            enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
            if not enabled_loras:
                print("ERROR: No enabled LoRAs found for serial mode")
                return None
            
            current_image_path = None
            intermediate_paths = []
            
            # Check for existing LoRA progress if resumable mode enabled
            lora_progress_key = f"thumbnail{".shorts" if shorts else ""}{variation_number}_lora_progress"
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
                workflow = self._load_thumbnail_workflow()
                if not workflow:
                    print(f"ERROR: Failed to load workflow for LoRA {i + 1}")
                    continue
                
                # Apply only this LoRA to the workflow
                self._apply_single_lora(workflow, lora_config, i + 1)
                
                # Update workflow with thumbnail-specific settings
                workflow = self._update_prompt_text(workflow, prompt_text)
                
                # Generate filename for this LoRA step
                lora_filename = f"thumbnail{".shorts" if shorts else ""}{variation_number}.{lora_clean_name}"
                workflow = self._update_saveimage_prefix(workflow, lora_filename)
                workflow = self._update_workflow_resolution(workflow, SHORTS_WIDTH if shorts else OUTPUT_WIDTH, SHORTS_HEIGHT if shorts else OUTPUT_HEIGHT)
                
                # Set LoRA-specific sampling steps, seed, and denoising
                steps = lora_config.get("steps", SAMPLING_STEPS)
                denoising_strength = lora_config.get("denoising_strength", 1.0)
                seed = self._get_seed()  # Serial mode uses base seed for each LoRA
                self._update_node_connections(workflow, "KSampler", "steps", steps)
                self._update_node_connections(workflow, "KSampler", "seed", seed)
                print(f"  Seed set to: {seed}")
                
                # Handle input for this LoRA based on serial mode logic
                if i == 0:
                    # First LoRA: Use latent/image mode based on LATENT_MODE setting
                    if LATENT_MODE == "IMAGE":
                        # Replace EmptySD3LatentImage with image input + LATENT_DENOISING_STRENGTH
                        self._replace_latent_with_image_input(workflow, "19", self.latent_image_path, LATENT_DENOISING_STRENGTH)
                        # Apply LATENT_DENOISING_STRENGTH to KSampler for first LoRA in IMAGE mode
                        self._update_node_connections(workflow, "KSampler", "denoise", LATENT_DENOISING_STRENGTH)
                        print(f"  Using image input mode for first LoRA with file: {self.latent_image_path}")
                        print(f"  Using LATENT_DENOISING_STRENGTH: {LATENT_DENOISING_STRENGTH}")
                    else:
                        # Normal latent mode - set dimensions and use LoRA's denoising strength
                        workflow["19"]["inputs"]["width"] = SHORTS_WIDTH if shorts else OUTPUT_WIDTH
                        workflow["19"]["inputs"]["height"] = SHORTS_HEIGHT if shorts else OUTPUT_HEIGHT
                        # Apply LoRA's denoising_strength to KSampler for first LoRA in LATENT mode
                        self._update_node_connections(workflow, "KSampler", "denoise", denoising_strength)
                        denoising_strength = LATENT_DENOISING_STRENGTH 
                        print(f"  Using latent mode with dimensions: {SHORTS_WIDTH if shorts else OUTPUT_WIDTH}x{SHORTS_HEIGHT if shorts else OUTPUT_HEIGHT}")
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
                
                print(f"  Steps: {steps}, Denoising: {denoising_strength}")
                
                # Print workflow summary before sending
                self._print_workflow_summary(workflow, f"LoRA {i + 1}: {lora_config['name']}")
                
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
                lora_final_path = os.path.join(self.intermediate_output_dir, f"thumbnail{".shorts" if shorts else ""}{variation_number}.{lora_clean_name}.png")
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
                        "lora_configs": {f"{j}_{lora["name"]}": lora for j, lora in enumerate(enabled_loras)}  # Save LoRA configs for resuming with unique IDs
                    }
                    resumable_state.state.setdefault("lora_progress", {})[lora_progress_key] = lora_progress
                    resumable_state._save_state()
                    print(f"  Saved LoRA progress: {len(completed_loras)}/{len(enabled_loras)} completed")
            
            if not current_image_path:
                print(f"ERROR: No successful LoRA generations for thumbnail")
                return None

            output_path = os.path.join(self.final_output_dir, f"thumbnail{".shorts" if shorts else ""}{variation_number}.png")
            
            # Copy final result to output directory
            shutil.copy2(current_image_path, output_path)
            
            # Apply text overlay if enabled
            title = self._read_title_from_file()
            if title and USE_TITLE_TEXT:
                self._overlay_title(output_path, title)
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                if shorts:
                    # Save as individual shorts thumbnail
                    thumbnail_key = f"thumbnail{'.shorts' if shorts else ''}{variation_number}"
                    shorts_result = {
                        'path': output_path,
                        'thumbnail_key': thumbnail_key,
                        'shorts': shorts,
                        'variation_number': variation_number,
                        'intermediate_paths': intermediate_paths
                    }
                    resumable_state.set_shorts_thumbnail_result(thumbnail_key, shorts_result)
                else:
                    # Save as main variation
                    var_key = variation_number or "original"
                    variation_result = {
                        'path': output_path,
                        'thumbnail_key': f"thumbnail{variation_number}",
                        'shorts': shorts,
                        'variation_number': variation_number,
                        'intermediate_paths': intermediate_paths
                    }
                    resumable_state.set_thumbnail_variation("thumbnail", var_key, variation_result)
                
                # Keep LoRA progress for completed thumbnail (not cleaning up)
                print(f"  Preserved LoRA progress for completed thumbnail")
            
            print(f"Saved: {output_path}")
            return output_path

        except Exception as e:
            print(f"ERROR: Failed to generate thumbnail: {e}")
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

    def generate_thumbnail(self, prompt_text: str, shorts: bool = False, variation_number: str = "", resumable_state=None) -> str | list[str] | None:
        try:
            # Check if resumable and already complete
            if resumable_state:
                thumbnail_key = f"thumbnail{'.shorts' if shorts else ''}{variation_number}"
                
                if shorts:
                    # For shorts thumbnails, check if this specific shorts thumbnail is complete
                    if resumable_state.is_shorts_thumbnail_complete(thumbnail_key):
                        result = resumable_state.get_thumbnail_result(thumbnail_key)
                        if result and os.path.exists(result.get("path", "")):
                            print(f"Using cached shorts thumbnail: {thumbnail_key}")
                            return result["path"]
                else:
                    # For regular thumbnails, check if this specific variation is complete
                    if resumable_state.is_thumbnail_variation_complete("thumbnail", variation_number or "original"):
                        result = resumable_state.get_thumbnail_result("thumbnail")
                        variations = result.get("variations", {})
                        var_key = variation_number or "original"
                        if var_key in variations and os.path.exists(variations[var_key].get("path", "")):
                            print(f"Using cached thumbnail: {thumbnail_key}")
                            return variations[var_key]["path"]
            
            # Use serial LoRA mode if enabled
            if USE_LORA and LORA_MODE == "serial":
                return self._generate_thumbnail_serial(prompt_text, shorts, variation_number, resumable_state)
            
            # Determine if we should use text overlay or let the model generate text
            use_overlay = USE_TITLE_TEXT
            title = None
            band_height = 0
            
            if use_overlay:
                title = self._read_title_from_file()
                # Compute for the final canvas size
                band_height, _padding, _lines, _font, _line_height, _stroke_w = self._measure_title_block(
                    title or "", SHORTS_WIDTH if shorts else OUTPUT_WIDTH, SHORTS_HEIGHT if shorts else OUTPUT_HEIGHT
                )

            # Decide generation image area based on layout/position
            gen_width = SHORTS_WIDTH if shorts else OUTPUT_WIDTH
            gen_height = SHORTS_HEIGHT if shorts else OUTPUT_HEIGHT
            if use_overlay:
                position = (TITLE_POSITION or "bottom").strip().lower()
                layout = (TITLE_LAYOUT or "overlay").strip().lower()
                if layout in ("expand",):
                    # Keep base generation at full size; we'll expand canvas after
                    gen_width, gen_height = SHORTS_WIDTH if shorts else OUTPUT_WIDTH, SHORTS_HEIGHT if shorts else OUTPUT_HEIGHT
                elif layout in ("fit",):
                    # Reserve band space within the canvas for top/bottom
                    if position in ("top", "upper", "bottom"):
                        gen_height = max(1, SHORTS_HEIGHT if shorts else OUTPUT_HEIGHT - band_height)
                    else:
                        gen_height = SHORTS_HEIGHT if shorts else OUTPUT_HEIGHT
                else:
                    # overlay uses full image
                    gen_width, gen_height = SHORTS_WIDTH if shorts else OUTPUT_WIDTH, SHORTS_HEIGHT if shorts else OUTPUT_HEIGHT

            workflow = self._load_thumbnail_workflow()
            workflow = self._update_prompt_text(workflow, prompt_text)
            workflow = self._update_workflow_resolution(workflow, gen_width, gen_height)
            workflow = self._update_saveimage_prefix(workflow, "thumbnail" + ".shorts" if shorts else "" + variation_number)
            
            # Set seed based on configuration (variation 0 = original)
            seed = self._get_seed()
            workflow = self._update_workflow_seed(workflow, seed)
            print(f"Seed set to: {seed}")

            # Print workflow summary
            self._print_workflow_summary(workflow, "Thumbnail" + ".shorts" if shorts else "" + variation_number)
            
            # Print prompt before sending
            print(f"\n=== PROMPT FOR THUMBNAIL{".shorts" if shorts else "" + variation_number} ===")
            # Get the text prompt from the workflow
            text_prompt = workflow.get("33", {}).get("inputs", {}).get("text", "No text prompt found")
            print(f"Text prompt: {text_prompt}")
            print(f"Workflow nodes: {len(workflow)} nodes")
            print("=" * 50)

            # Single generation path with text overlay
            resp = requests.post(f"{self.comfyui_url}prompt", json={"prompt": workflow}, timeout=60)
            if resp.status_code != 200:
                return None
            prompt_id = resp.json().get("prompt_id")
            if not prompt_id:
                return None

            while True:
                h = requests.get(f"{self.comfyui_url}history/{prompt_id}")
                if h.status_code == 200:
                    data = h.json()
                    if prompt_id in data:
                        status = data[prompt_id].get("status", {})
                        if status.get("exec_info", {}).get("queue_remaining", 0) == 0:
                            time.sleep(2)
                            break
                time.sleep(2)

            newest_path = self._find_newest_output_with_prefix("thumbnail" + ".shorts" if shorts else "" + variation_number)
            if not newest_path:
                return None

            src_ext = Path(newest_path).suffix.lower()
            final_path = os.path.join(self.final_output_dir, f"thumbnail{".shorts" if shorts else "" + variation_number}{src_ext}")
            shutil.copy2(newest_path, final_path)
            # Apply text overlay if enabled
            if title and use_overlay:
                self._overlay_title(final_path, title)
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                if shorts:
                    # Save as individual shorts thumbnail
                    shorts_result = {
                        'path': final_path,
                        'thumbnail_key': thumbnail_key,
                        'shorts': shorts,
                        'variation_number': variation_number,
                        'intermediate_paths': []  # Empty for non-serial modes
                    }
                    resumable_state.set_shorts_thumbnail_result(thumbnail_key, shorts_result)
                else:
                    # Save as main variation
                    var_key = variation_number or "original"
                    variation_result = {
                        'path': final_path,
                        'thumbnail_key': f"thumbnail{variation_number}",
                        'shorts': shorts,
                        'variation_number': variation_number,
                        'intermediate_paths': []  # Empty for non-serial modes
                    }
                    resumable_state.set_thumbnail_variation("thumbnail", var_key, variation_result)
            
            return final_path
        except Exception:
            return None

    def _find_newest_output_with_prefix(self, prefix: str) -> str | None:
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

    def _read_title_from_file(self, filename: str = "../input/10.title.txt") -> str | None:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                text = f.read().strip()
                return text if text else None
        except Exception:
            return None

    def _normalize_title(self, title: str) -> str:
        """Remove special characters and convert to Title Case per word.
        Keeps only A-Z, a-z, 0-9 and spaces; collapses multiple spaces.
        """
        if not title:
            return ""
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", title)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return " ".join(part.capitalize() for part in cleaned.lower().split(" "))

    def _wrap_lines(self, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int, stroke_width: int = 0) -> list[str]:
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

    def _load_font(self, size: int) -> ImageFont.ImageFont:
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

    def _measure_title_block(self, title: str, canvas_width: int, canvas_height: int) -> tuple[int, int, list[str], ImageFont.ImageFont, int, int]:
        """Compute the band height and wrapped lines for the given title on a target canvas.
        Returns (band_height, padding, lines, chosen_font, line_height, stroke_w)
        """
        dummy_img = Image.new("RGBA", (max(1, canvas_width), max(1, canvas_height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy_img)
        scale = float(TITLE_FONT_SCALE) if isinstance(TITLE_FONT_SCALE, (int, float)) else 1.0
        if not (scale > 0):
            scale = 1.0
        stroke_w = max(2, int(canvas_height * 0.008 * scale))
        base_font_size = max(28, int(canvas_height * 0.06 * scale))
        min_font_size = max(1, int(((18 * 3) * 0.75) * scale))
        padding = int(canvas_height * 0.04 * max(1.0, min(scale, 2.0)))
        max_text_width = int(canvas_width * 0.92)

        title_text = (title or "").strip().upper()
        chosen_font: ImageFont.ImageFont | None = None
        lines: list[str] | None = None
        line_height: int | None = None
        block_height: int | None = None

        for size in range(base_font_size, min_font_size - 1, -2):
            font = self._load_font(size)
            lines_try = self._wrap_lines(draw, title_text, font, max_text_width, stroke_width=stroke_w)
            sample_bbox = draw.textbbox((0, 0), "Ag", font=font, stroke_width=stroke_w)
            lh = (sample_bbox[3] - sample_bbox[1]) + int(canvas_height * 0.01)
            bh = len(lines_try) * lh + padding * 2
            if len(lines_try) <= 3 and bh <= int(canvas_height * 0.85):
                chosen_font = font
                lines = lines_try
                line_height = lh
                block_height = bh
                break

        if chosen_font is None:
            chosen_font = self._load_font(min_font_size)
            lines = self._wrap_lines(draw, title_text, chosen_font, max_text_width, stroke_width=stroke_w)
            sample_bbox = draw.textbbox((0, 0), "Ag", font=chosen_font, stroke_width=stroke_w)
            line_height = (sample_bbox[3] - sample_bbox[1]) + int(canvas_height * 0.01)
            block_height = len(lines) * line_height + padding * 2

        band_height = min(block_height or 0, canvas_height)
        return band_height, padding, lines or [], chosen_font, line_height or 0, stroke_w

    def _overlay_title(self, image_path: str, title: str) -> bool:
        try:
            img = Image.open(image_path).convert("RGBA")
            w, h = img.size
            draw = ImageDraw.Draw(img)

            # Uppercase for impact
            title_text = title.strip().upper()

            # Dynamic sizing and stroke
            scale = float(TITLE_FONT_SCALE) if isinstance(TITLE_FONT_SCALE, (int, float)) else 1.0
            if not (scale > 0):
                scale = 1.0
            stroke_w = max(2, int(h * 0.008 * scale))
            base_font_size = max(28, int(h * 0.06 * scale))
            min_font_size = int(((18 * 3) * 0.75) * scale)
            padding = int(h * 0.04 * max(1.0, min(scale, 2.0)))
            max_text_width = int(w * 0.92)

            chosen_font = None
            lines = None
            line_height = None
            block_height = None

            for size in range(base_font_size, max(min_font_size, 1) - 1, -2):
                font = self._load_font(size)
                lines_try = self._wrap_lines(draw, title_text, font, max_text_width, stroke_width=stroke_w)
                # Measure line height with stroke
                sample_bbox = draw.textbbox((0, 0), "Ag", font=font, stroke_width=stroke_w)
                lh = (sample_bbox[3] - sample_bbox[1]) + int(h * 0.01)
                bh = len(lines_try) * lh + padding * 2
                if len(lines_try) <= 3 and bh <= int(h * 0.85):
                    chosen_font = font
                    lines = lines_try
                    line_height = lh
                    block_height = bh
                    break

            if chosen_font is None:
                chosen_font = self._load_font(max(min_font_size, 1))
                lines = self._wrap_lines(draw, title_text, chosen_font, max_text_width, stroke_width=stroke_w)
                sample_bbox = draw.textbbox((0, 0), "Ag", font=chosen_font, stroke_width=stroke_w)
                line_height = (sample_bbox[3] - sample_bbox[1]) + int(h * 0.01)
                block_height = len(lines) * line_height + padding * 2

            # Constant-opacity band positioned by TITLE_POSITION and layout by TITLE_LAYOUT
            band_height = min(block_height, h)
            pos = (TITLE_POSITION or "bottom").strip().lower()
            layout = (TITLE_LAYOUT or "overlay").strip().lower()
            if layout not in ("overlay", "expand", "fit"):
                layout = "overlay"
            if pos in ("middle", "center", "centre") and layout in ("expand", "fit"):
                # Fallback to overlay for middle placement when avoiding overlap
                layout = "overlay"

            band_opacity = int(0.60 * 255)
            # Prebuild band and text layers
            band = Image.new("RGBA", (w, band_height), (0, 0, 0, band_opacity))
            shadow = band.copy().filter(ImageFilter.GaussianBlur(radius=int(h * 0.01)))
            text_layer = Image.new("RGBA", (w, band_height), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_layer)
            y_cursor = padding
            for line in lines:
                bbox = text_draw.textbbox((0, 0), line, font=chosen_font, stroke_width=0)
                tw = bbox[2] - bbox[0]
                x = (w - tw) // 2
                text_draw.text((x, y_cursor), line, font=chosen_font, fill=(255, 255, 255, 255))
                y_cursor += line_height

            if layout == "overlay":
                # Overlay band and text onto original image
                if pos in ("top", "upper"):
                    band_y = 0
                elif pos in ("middle", "center", "centre"):
                    band_y = max(0, (h - band_height) // 2)
                else:
                    band_y = h - band_height
                layer_shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
                layer_shadow.paste(shadow, (0, band_y), shadow)
                img = Image.alpha_composite(img, layer_shadow)
                layer_band = Image.new("RGBA", img.size, (0, 0, 0, 0))
                layer_band.paste(band, (0, band_y), band)
                img = Image.alpha_composite(img, layer_band)
                layer_text = Image.new("RGBA", img.size, (0, 0, 0, 0))
                layer_text.paste(text_layer, (0, band_y), text_layer)
                img = Image.alpha_composite(img, layer_text)

            elif layout == "expand":
                # Increase canvas height to place band outside the image
                new_h = h + band_height
                new_img = Image.new("RGBA", (w, new_h), (0, 0, 0, 0))
                if pos in ("top", "upper"):
                    band_y = 0
                    image_y = band_height
                else:
                    image_y = 0
                    band_y = h
                new_img.paste(img, (0, image_y))
                layer_shadow = Image.new("RGBA", new_img.size, (0, 0, 0, 0))
                layer_shadow.paste(shadow, (0, band_y), shadow)
                new_img = Image.alpha_composite(new_img, layer_shadow)
                layer_band = Image.new("RGBA", new_img.size, (0, 0, 0, 0))
                layer_band.paste(band, (0, band_y), band)
                new_img = Image.alpha_composite(new_img, layer_band)
                layer_text = Image.new("RGBA", new_img.size, (0, 0, 0, 0))
                layer_text.paste(text_layer, (0, band_y), text_layer)
                new_img = Image.alpha_composite(new_img, layer_text)
                img = new_img
                w, h = img.size

            else:  # layout == "fit"
                # Fill reserved area for the image within the same canvas size (no aspect ratio)
                if pos in ("top", "upper"):
                    band_y = 0
                    area_y0 = band_height
                else:
                    band_y = h - band_height
                    area_y0 = 0
                area_h = max(1, h - band_height)
                area_w = max(1, w)
                # Stretch to exactly fit the target area
                scaled = img.resize((area_w, area_h), Image.LANCZOS)
                new_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                paste_x = 0
                paste_y = area_y0
                new_img.paste(scaled, (paste_x, paste_y))
                layer_shadow = Image.new("RGBA", new_img.size, (0, 0, 0, 0))
                layer_shadow.paste(shadow, (0, band_y), shadow)
                new_img = Image.alpha_composite(new_img, layer_shadow)
                layer_band = Image.new("RGBA", new_img.size, (0, 0, 0, 0))
                layer_band.paste(band, (0, band_y), band)
                new_img = Image.alpha_composite(new_img, layer_band)
                layer_text = Image.new("RGBA", new_img.size, (0, 0, 0, 0))
                layer_text.paste(text_layer, (0, band_y), text_layer)
                new_img = Image.alpha_composite(new_img, layer_text)
                img = new_img

            # Save back
            if image_path.lower().endswith((".jpg", ".jpeg")):
                img.convert("RGB").save(image_path, quality=95)
            else:
                img.save(image_path)
            return True
        except Exception:
            return False


    def _get_master_prompt(self) -> str:
        """Get the master prompt content."""
        return """Create a 16K ultra-high-resolution, illustration (with non-black and non-white background) in the style of {ART_STYLE}, with shot taken with camera placed at very large distance(at least 12 meters away) and ultra wide angle(160 degrees) lens.
        """.format(ART_STYLE=ART_STYLE)
        
def read_prompt_from_file(filename: str = "../input/10.thumbnail.txt") -> str | None:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a thumbnail using diffusion (default) or flux workflow.")
    parser.add_argument("--mode", "-m", choices=["diffusion", "flux"], default="diffusion", help="Select workflow: diffusion (default) or flux")
    parser.add_argument("--force", "-f", action="store_true", help="Force regeneration of all thumbnails")
    parser.add_argument("--force-start", action="store_true", help="Force start from beginning, ignoring any existing checkpoint files")
    args = parser.parse_args()

    prompt = read_prompt_from_file()
    if not prompt:
        raise SystemExit(1)

    prompt =  "SCENE DESCRIPTION:" + prompt

    processor = ThumbnailProcessor(mode=args.mode)

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
        
        # Validate and sync with output directory
        print(f"Validating and syncing with output directory: {processor.final_output_dir}")
        
        # First, sync with output directory to detect any manually added files
        synced_count = resumable_state.sync_with_output_directory(processor.final_output_dir)
        if synced_count > 0:
            print(f"Sync completed: {synced_count} thumbnails auto-detected from output directory")
        
        # Then run precheck to validate file existence and clean up invalid entries
        cleaned_count = resumable_state.validate_and_cleanup_results(processor.final_output_dir)
        if cleaned_count > 0:
            print(f"Precheck completed: {cleaned_count} invalid entries removed from checkpoint")

    title = processor._read_title_from_file()
    if title and not USE_TITLE_TEXT:
        # Only include title in prompt when not using overlay (let model generate text)
        prompt = "TITLE DESCRIPTION: ADD A very large semi-transparent floating newspaper at top-center with arial bold font & grammatically correct english-only legible engraving as \"" + processor._normalize_title(title) + "\"\n\n" + prompt

    # Generate all thumbnails following character script pattern
    results = {}
    master_prompt = processor._get_master_prompt() + "\n\n " + prompt
    
    if USE_TITLE_TEXT:
        # Generate main thumbnail (original)
        print("Generating main thumbnail...")
        result = processor.generate_thumbnail(master_prompt, shorts=False, variation_number="", resumable_state=resumable_state)
        if result:
            results["thumbnail"] = result
            print(f"Generated: {result}")
        else:
            print("Failed to generate main thumbnail")
        
        # Generate shorts variations as separate entities
        for i in range(1, SHORTS_VARIATIONS + 1):
            shorts_key = f"thumbnail.shorts.v{i}"
            print(f"Generating shorts thumbnail {i}/{SHORTS_VARIATIONS}...")
            result = processor.generate_thumbnail(master_prompt, shorts=True, variation_number=".v" + str(i), resumable_state=resumable_state)
            if result:
                results[shorts_key] = result
                print(f"Generated: {result}")
            else:
                print(f"Failed to generate shorts thumbnail {i}")
    else:
        # Generate multiple variations like character script
        for i in range(0, 6):
            thumbnail_key = f"thumbnail{'.v' + str(i) if i > 0 else ''}"
            print(f"Generating thumbnail {i + 1} of 6: {thumbnail_key}")
            result = processor.generate_thumbnail(master_prompt, shorts=False, variation_number=(".v" + str(i) if i > 0 else ""), resumable_state=resumable_state)
            if result:
                results[thumbnail_key] = result
                print(f"Generated: {result}")
            else:
                print(f"Failed to generate thumbnail {i + 1}")
        
        # Generate shorts variations for each main variation
        for i in range(0, 6):
            for j in range(1, SHORTS_VARIATIONS + 1):
                shorts_key = f"thumbnail.shorts.v{j}.v{i}" if i > 0 else f"thumbnail.shorts.v{j}"
                print(f"Generating shorts {j}/{SHORTS_VARIATIONS} for variation {i + 1}/6: {shorts_key}")
                variation_number = ".v" + str(j) + (".v" + str(i) if i > 0 else "")
                result = processor.generate_thumbnail(master_prompt, shorts=True, variation_number=variation_number, resumable_state=resumable_state)
                if result:
                    results[shorts_key] = result
                    print(f"Generated: {result}")
                else:
                    print(f"Failed to generate shorts {j} for variation {i + 1}")
    
    if not results:
        print("No thumbnails were generated successfully")
        raise SystemExit(1)
    
    # Print summary like character script
    print(f"\nGenerated {len(results)} thumbnail images:")
    for key, path in results.items():
        print(f"  {key}: {path}")

    # Clean up checkpoint files if resumable mode was used and everything completed successfully
    if resumable_state:
        print("All thumbnail generation completed successfully")
        print("Final progress:", resumable_state.get_progress_summary())
        # Save final state before cleanup
        print("Final state saved to checkpoint")
        resumable_state.cleanup()

