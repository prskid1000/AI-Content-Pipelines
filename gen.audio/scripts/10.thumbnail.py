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

# Random seed configuration
USE_RANDOM_SEED = True  # Set to True to use random seeds for each generation
RANDOM_SEED = 333555666

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

# Image Output Dimension Constants
USE_FIXED_DIMENSIONS = True  # Set to True to use fixed width/height, False to use aspect ratio calculation
IMAGE_OUTPUT_WIDTH = 1280
IMAGE_OUTPUT_HEIGHT = 720

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
        "steps": 8,               # Sampling steps for this LoRA (serial mode only)
        "denoising_strength": 1,  # Denoising strength (0.0 - 1.0) (serial mode only)
        "save_intermediate": True # Save intermediate results for debugging (serial mode only)
    },
    {
        "name": "Ghibli_lora_weights.safetensors",  # Example second LoRA
        "strength_model": 2.0,
        "strength_clip": 2.0,
        "bypass_model": False,
        "bypass_clip": False,
        "enabled": False,  # Disabled by default
        
        # Serial mode specific settings
        "steps": 8,
        "denoising_strength": 0.6,
        "save_intermediate": True
    },
]

# Sampling Configuration
SAMPLING_STEPS = 9  # Number of sampling steps (higher = better quality, slower)

# Negative Prompt Configuration
USE_NEGATIVE_PROMPT = True  # Set to True to enable negative prompts, False to disable
NEGATIVE_PROMPT = "blur, distorted, text, watermark, extra limbs, bad anatomy, poorly drawn, asymmetrical, malformed, disfigured, ugly, bad proportions, plastic texture, artificial looking, cross-eyed, missing fingers, extra fingers, bad teeth, missing teeth, unrealistic"

ART_STYLE = "Realistic Anime"

class ThumbnailProcessor:
    def __init__(self, comfyui_url: str = "http://127.0.0.1:8188/", mode: str = "diffusion"):
        self.comfyui_url = comfyui_url
        self.mode = (mode or "diffusion").strip().lower()
        # ComfyUI saves images under this folder
        self.comfyui_output_folder = "../../ComfyUI/output"
        # Final destination inside this repo
        self.final_output_dir = "../output"
        self.intermediate_output_dir = "../output/lora"
        self.final_output_path = os.path.join(self.final_output_dir, "thumbnail.png")

        os.makedirs(self.final_output_dir, exist_ok=True)
        os.makedirs(self.intermediate_output_dir, exist_ok=True)
        if os.path.exists(self.final_output_path):
            os.remove(self.final_output_path)

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
        """Update the workflow with specified resolution."""
        # Use fixed dimensions if specified, otherwise use the provided width/height
        if USE_FIXED_DIMENSIONS:
            final_width = IMAGE_OUTPUT_WIDTH
            final_height = IMAGE_OUTPUT_HEIGHT
            print(f"Using fixed dimensions: {final_width}x{final_height} (bypassing aspect ratio calculation)")
        else:
            final_width = width
            final_height = height
            print(f"Using calculated dimensions: {final_width}x{final_height}")
        
        # Find EmptySD3LatentImage node and update its dimensions
        latent_image_node = self._find_node_by_class(workflow, "EmptySD3LatentImage")
        if latent_image_node:
            node_id = latent_image_node[0]
            node = workflow[node_id]
            node.setdefault("inputs", {})
            w_in = node["inputs"].get("width")
            h_in = node["inputs"].get("height")
            # Only override if inputs are not wired from another node
            if not isinstance(w_in, (list, tuple)) and not isinstance(h_in, (list, tuple)):
                node["inputs"]["width"] = int(final_width)
                node["inputs"]["height"] = int(final_height)
                print(f"Updated width/height to {final_width}/{final_height} for node {node_id}")
        else:
            print("No EmptySD3LatentImage node found in workflow")
        
        return workflow

    def _update_workflow_seed(self, workflow: dict, seed: int) -> dict:
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

    def _generate_thumbnail_serial(self, prompt_text: str) -> str | None:
        """Generate thumbnail using serial LoRA mode with intermediate storage."""
        try:
            print(f"Generating thumbnail (Serial LoRA mode)")
            
            enabled_loras = [lora for lora in LORAS if lora.get("enabled", True)]
            if not enabled_loras:
                print("ERROR: No enabled LoRAs found for serial mode")
                return None
            
            current_image_path = None
            
            # Process each LoRA in sequence, using previous output as input
            for i, lora_config in enumerate(enabled_loras):
                print(f"\nProcessing LoRA {i + 1}/{len(enabled_loras)}: {lora_config['name']}")
                
                # Load base workflow for this LoRA
                workflow = self._load_thumbnail_workflow()
                if not workflow:
                    print(f"ERROR: Failed to load workflow for LoRA {i + 1}")
                    continue
                
                # Apply only this LoRA to the workflow
                self._apply_single_lora(workflow, lora_config, i + 1)
                
                # Update workflow with thumbnail-specific settings
                workflow = self._update_prompt_text(workflow, prompt_text)
                workflow = self._update_saveimage_prefix(workflow, f"thumbnail_lora_{i + 1}")
                workflow = self._update_workflow_resolution(workflow, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                
                # Set LoRA-specific sampling steps and denoising
                steps = lora_config.get("steps", SAMPLING_STEPS)
                denoising_strength = lora_config.get("denoising_strength", 1.0)
                self._update_node_connections(workflow, "KSampler", "steps", steps)
                self._update_node_connections(workflow, "KSampler", "denoise", denoising_strength)
                
                # If this is not the first LoRA, use previous output as input
                if i > 0 and current_image_path:
                    # Load previous image as input for this LoRA
                    self._set_image_input(workflow, current_image_path)
                    print(f"  Using previous LoRA output as input")
                
                print(f"  Steps: {steps}, Denoising: {denoising_strength}")
                
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
                generated_image = self._find_newest_output_with_prefix(f"thumbnail_lora_{i + 1}")
                if not generated_image:
                    print(f"ERROR: Could not find generated image for LoRA {i + 1}")
                    continue
                
                # Save result to lora folder (save final result from each LoRA)
                lora_clean_name = re.sub(r'[^\w\s-]', '', lora_config['name']).strip()
                lora_clean_name = re.sub(r'[-\s]+', '_', lora_clean_name)
                lora_final_path = os.path.join(self.intermediate_output_dir, f"thumbnail.{lora_clean_name}.png")
                shutil.copy2(generated_image, lora_final_path)
                print(f"  Saved LoRA result: {lora_final_path}")
                
                # Use this output as input for next LoRA
                current_image_path = generated_image
                print(f"  LoRA {i + 1} completed successfully")
            
            if not current_image_path:
                print(f"ERROR: No successful LoRA generations for thumbnail")
                return None
            
            # Copy final result to output directory
            shutil.copy2(current_image_path, self.final_output_path)
            
            # Apply text overlay if enabled
            title = self._read_title_from_file()
            if title and USE_TITLE_TEXT:
                self._overlay_title(self.final_output_path, title)
            
            print(f"Saved: {self.final_output_path}")
            return self.final_output_path

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

    def generate_thumbnail(self, prompt_text: str) -> str | list[str] | None:
        try:
            # Use serial LoRA mode if enabled
            if USE_LORA and LORA_MODE == "serial":
                return self._generate_thumbnail_serial(prompt_text)
            
            # Determine if we should use text overlay or let the model generate text
            use_overlay = USE_TITLE_TEXT
            title = None
            band_height = 0
            
            if use_overlay:
                title = self._read_title_from_file()
                # Compute for the final canvas size
                band_height, _padding, _lines, _font, _line_height, _stroke_w = self._measure_title_block(
                    title or "", OUTPUT_WIDTH, OUTPUT_HEIGHT
                )

            # Decide generation image area based on layout/position
            gen_width = OUTPUT_WIDTH
            gen_height = OUTPUT_HEIGHT
            if use_overlay:
                position = (TITLE_POSITION or "bottom").strip().lower()
                layout = (TITLE_LAYOUT or "overlay").strip().lower()
                if layout in ("expand",):
                    # Keep base generation at full size; we'll expand canvas after
                    gen_width, gen_height = OUTPUT_WIDTH, OUTPUT_HEIGHT
                elif layout in ("fit",):
                    # Reserve band space within the canvas for top/bottom
                    if position in ("top", "upper", "bottom"):
                        gen_height = max(1, OUTPUT_HEIGHT - band_height)
                    else:
                        gen_height = OUTPUT_HEIGHT
                else:
                    # overlay uses full image
                    gen_width, gen_height = OUTPUT_WIDTH, OUTPUT_HEIGHT

            workflow = self._load_thumbnail_workflow()
            workflow = self._update_prompt_text(workflow, prompt_text)
            workflow = self._update_workflow_resolution(workflow, gen_width, gen_height)
            workflow = self._update_saveimage_prefix(workflow, "thumbnail")
            
            # Set seed based on configuration
            if not USE_RANDOM_SEED:
                workflow = self._update_workflow_seed(workflow, RANDOM_SEED)

            # If not using overlay, generate multiple variants (like flux mode)
            if not use_overlay:
                saved_paths: list[str] = []
                for idx in range(1, 6):
                    # Use fixed seed or random seed based on configuration
                    if USE_RANDOM_SEED:
                        seed_value = random.randint(1, 2**31 - 1)
                    else:
                        seed_value = RANDOM_SEED
                    workflow = self._update_workflow_seed(workflow, seed_value)
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

                    newest_path = self._find_newest_output_with_prefix("thumbnail")
                    if not newest_path:
                        return None

                    # Store outputs as PNG with version suffix
                    final_path = os.path.join(self.final_output_dir, f"thumbnail.v{idx}.png")
                    try:
                        img = Image.open(newest_path)
                        img.save(final_path, format="PNG")
                    except Exception:
                        shutil.copy2(newest_path, final_path)

                    saved_paths.append(final_path)

                return saved_paths

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

            newest_path = self._find_newest_output_with_prefix("thumbnail")
            if not newest_path:
                return None

            src_ext = Path(newest_path).suffix.lower()
            final_path = self.final_output_path if src_ext == ".png" else os.path.join(self.final_output_dir, f"thumbnail{src_ext}")
            shutil.copy2(newest_path, final_path)
            # Apply text overlay if enabled
            if title and use_overlay:
                self._overlay_title(final_path, title)
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
        return """Create a 16K ultra-high-resolution, illustration in the style of {ART_STYLE}. The artwork should feature fine, intricate details and a natural sense of depth, with carefully chosen camera angle and focus to best frame the Scene. 
Must Always Precisely & Accurately Preserve each Character's identity(all physical features - face, body, height, weight, clothings) from respective specified reference image, though "posture", "expression", "movement", "placement" and "action-performed" is adaptable according to Scene/Character text-description.
Must Always Precisely & Accurately Represent entire Scene and all Non-Living Objects according to scene text-description.
All Non-Living Objects mentioned in Scene text-description must be present in illustration.
Each Object/Character in the illustration must be visually distinct/unique from each other.
Strictly, Accurately, Precisely, always must Follow {ART_STYLE} Style.
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
    args = parser.parse_args()

    prompt = read_prompt_from_file()
    if not prompt:
        raise SystemExit(1)

    prompt =  "SCENE DESCRIPTION:" + prompt

    processor = ThumbnailProcessor(mode=args.mode)

    title = processor._read_title_from_file()
    if title and not USE_TITLE_TEXT:
        # Only include title in prompt when not using overlay (let model generate text)
        prompt = "TITLE DESCRIPTION: ADD A very large semi-transparent floating newspaper at top-center with arial bold font & grammatically correct english-only legible engraving as \"" + processor._normalize_title(title) + "\"\n\n" + prompt

    result = processor.generate_thumbnail(_get_master_prompt() + "\n\n " + prompt)
    if result:
        if isinstance(result, list):
            for p in result:
                print(p)
        else:
            print(result)
    else:
        raise SystemExit(1)


