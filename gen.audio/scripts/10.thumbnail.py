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
        "strength_model": 3.6,    # Model strength (0.0 - 2.0)
        "strength_clip": 3.6,     # CLIP strength (0.0 - 2.0)
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
        "steps": 6,               # Sampling steps for this LoRA (serial mode only)
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
        self.shorts_output_path = os.path.join(self.final_output_dir, "thumbnail.short.png")
        # Latent image input file path
        self.latent_image_path = "../input/10.latent.png"

        os.makedirs(self.final_output_dir, exist_ok=True)
        os.makedirs(self.intermediate_output_dir, exist_ok=True)
        if os.path.exists(self.final_output_path):
            os.remove(self.final_output_path)
        if os.path.exists(self.shorts_output_path):
            os.remove(self.shorts_output_path)

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
        
        # Print comprehensive sampler summary
        self._print_sampler_summary(workflow, sampler_id)
        
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

    def _build_forward_path(self, workflow: dict, start_node_id: str, end_node_id: str) -> list:
        """Build a path from start node to end node by tracing forwards."""
        path = []
        visited = set()
        current = start_node_id
        
        while current and current not in visited:
            visited.add(current)
            path.append(current)
            
            if current == end_node_id:
                break
                
            # Find the next node that this node connects to
            # Look for nodes that have this node as an input
            found_next = False
            for node_id, node_data in workflow.items():
                if node_id != current and node_id not in visited:
                    node_inputs = node_data.get("inputs", {})
                    for input_name, input_value in node_inputs.items():
                        if isinstance(input_value, list) and len(input_value) >= 2:
                            if input_value[0] == current:
                                current = node_id
                                found_next = True
                                break
                    if found_next:
                        break
            
            if not found_next:
                break
        
        return path

    def _print_forward_path(self, workflow: dict, path: list, sampler_id: str) -> None:
        """Print the forward path from source to sampler."""
        if not path:
            print("      ❌ No path found")
            return
            
        for i, node_id in enumerate(path):
            if node_id not in workflow:
                continue
                
            node = workflow[node_id]
            node_type = node.get("class_type", "Unknown")
            node_inputs = node.get("inputs", {})
            
            # Indent based on position in path
            indent = "      " + "   " * i
            
            if i == 0:
                # First node (source)
                print(f"{indent}📤 {node_type}({node_id})")
            elif i == len(path) - 1:
                # Last node (sampler)
                print(f"{indent}📥 {node_type}({node_id})")
            else:
                # Middle nodes
                print(f"{indent}⬇️  {node_type}({node_id})")
            
            # Show node parameters
            self._show_node_parameters(node_type, node_inputs, indent + "   ")

    def _print_sampler_summary(self, workflow: dict, sampler_id: str) -> None:
        """Print a comprehensive summary of all sampler values and connections."""
        print(f"\n📊 SAMPLER SUMMARY:")
        
        sampler_node = workflow.get(sampler_id, {})
        sampler_inputs = sampler_node.get("inputs", {})
        
        # Core sampler parameters
        print(f"   🎲 Core Parameters:")
        print(f"      Steps: {sampler_inputs.get('steps', 'N/A')}")
        print(f"      Denoising: {sampler_inputs.get('denoise', 'N/A')}")
        print(f"      Seed: {sampler_inputs.get('seed', 'N/A')}")
        print(f"      CFG: {sampler_inputs.get('cfg', 'N/A')}")
        print(f"      Sampler: {sampler_inputs.get('sampler_name', 'N/A')}")
        print(f"      Scheduler: {sampler_inputs.get('scheduler', 'N/A')}")
        
        

    def _trace_node_backwards(self, workflow: dict, node_id: str, target_id: str, depth: int) -> None:
        """Recursively trace backwards through the workflow graph."""
        if node_id not in workflow:
            return
            
        node = workflow[node_id]
        node_type = node.get("class_type", "Unknown")
        node_inputs = node.get("inputs", {})
        
        # Indent based on depth
        indent = "      " + "   " * depth
        
        # Show current node
        if depth == 0:
            print(f"      {node_type}({node_id}) → KSampler({target_id})")
        else:
            print(f"{indent}⬆️  {node_type}({node_id})")
        
        # Show node parameters
        self._show_node_parameters(node_type, node_inputs, indent + "   ")
        
        # Continue tracing backwards for each input
        for input_name, input_value in node_inputs.items():
            if isinstance(input_value, list) and len(input_value) >= 2:
                upstream_node_id = input_value[0]
                if upstream_node_id in workflow and upstream_node_id != node_id:  # Avoid infinite loops
                    self._trace_node_backwards(workflow, upstream_node_id, target_id, depth + 1)

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

    def _get_seed(self) -> int:
        """Get seed value based on configuration."""
        if USE_RANDOM_SEED:
            return random.randint(0, 2**32 - 1)
        else:
            return RANDOM_SEED

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
                
                # Generate filename for this LoRA step
                lora_clean_name = re.sub(r'[^\w\s.-]', '', lora_config['name']).strip()
                lora_clean_name = re.sub(r'[-\s]+', '_', lora_clean_name)
                lora_filename = f"thumbnail.{lora_clean_name}"
                workflow = self._update_saveimage_prefix(workflow, lora_filename)
                workflow = self._update_workflow_resolution(workflow, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                
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
                        # Replace EmptySD3LatentImage with image input + LATENT_DENOISING_STRENGTH
                        self._replace_latent_with_image_input(workflow, "19", self.latent_image_path, LATENT_DENOISING_STRENGTH)
                        # Apply LATENT_DENOISING_STRENGTH to KSampler for first LoRA in IMAGE mode
                        self._update_node_connections(workflow, "KSampler", "denoise", LATENT_DENOISING_STRENGTH)
                        print(f"  Using image input mode for first LoRA with file: {self.latent_image_path}")
                        print(f"  Using LATENT_DENOISING_STRENGTH: {LATENT_DENOISING_STRENGTH}")
                    else:
                        # Normal latent mode - set dimensions and use LoRA's denoising strength
                        workflow["19"]["inputs"]["width"] = OUTPUT_WIDTH
                        workflow["19"]["inputs"]["height"] = OUTPUT_HEIGHT
                        # Apply LoRA's denoising_strength to KSampler for first LoRA in LATENT mode
                        self._update_node_connections(workflow, "KSampler", "denoise", denoising_strength)
                        denoising_strength = LATENT_DENOISING_STRENGTH 
                        print(f"  Using latent mode with dimensions: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")
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
            
            # Create YouTube Shorts versions
            shorts_paths = self._create_shorts_versions(prompt_text)
            if shorts_paths:
                print(f"Also created {len(shorts_paths)} shorts variations: {shorts_paths}")
            
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
            seed = self._get_seed()
            workflow = self._update_workflow_seed(workflow, seed)
            print(f"Seed set to: {seed}")

            # Print workflow summary
            self._print_workflow_summary(workflow, "Thumbnail")
            
            # Print prompt before sending
            print(f"\n=== PROMPT FOR THUMBNAIL ===")
            # Get the text prompt from the workflow
            text_prompt = workflow.get("33", {}).get("inputs", {}).get("text", "No text prompt found")
            print(f"Text prompt: {text_prompt}")
            print(f"Workflow nodes: {len(workflow)} nodes")
            print("=" * 50)

            # If not using overlay, generate multiple variants (like flux mode)
            if not use_overlay:
                saved_paths: list[str] = []
                for idx in range(1, 6):
                    # Use seed based on configuration
                    seed_value = self._get_seed()
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
                    
                    # Create YouTube Shorts versions for each variant
                    # When USE_TITLE_TEXT = False, each main variant gets 5 Shorts variations
                    shorts_paths = self._create_shorts_versions_for_main_variant(prompt_text, idx)
                    if shorts_paths:
                        print(f"Also created {len(shorts_paths)} shorts variations for variant {idx}: {shorts_paths}")

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
            
            # Create YouTube Shorts versions
            shorts_paths = self._create_shorts_versions(prompt_text)
            if shorts_paths:
                print(f"Also created {len(shorts_paths)} shorts variations: {shorts_paths}")
            
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

    def _create_shorts_versions(self, prompt_text: str) -> list[str]:
        """Generate multiple YouTube Shorts versions using ComfyUI in proper 9:16 dimensions."""
        shorts_paths = []
        
        for i in range(1, SHORTS_VARIATIONS + 1):
            shorts_path = os.path.join(self.final_output_dir, f"thumbnail.short.v{i}.png")
            success = self._generate_single_shorts_version(prompt_text, shorts_path, i)
            if success:
                shorts_paths.append(shorts_path)
        
        return shorts_paths

    def _create_shorts_versions_for_main_variant(self, prompt_text: str, main_variant_num: int) -> list[str]:
        """Generate multiple YouTube Shorts versions for a specific main variant using ComfyUI."""
        shorts_paths = []
        
        for i in range(1, SHORTS_VARIATIONS + 1):
            shorts_path = os.path.join(self.final_output_dir, f"thumbnail.short.v{main_variant_num}.v{i}.png")
            success = self._generate_single_shorts_version(prompt_text, shorts_path, f"{main_variant_num}.{i}")
            if success:
                shorts_paths.append(shorts_path)
        
        return shorts_paths

    def _generate_single_shorts_version(self, prompt_text: str, output_path: str, variation_num: int) -> bool:
        """Generate a YouTube Shorts version using ComfyUI in proper 9:16 dimensions."""
        try:
            print(f"Generating YouTube Shorts version {variation_num} using ComfyUI...")
            
            # Load base workflow
            workflow = self._load_thumbnail_workflow()
            if not workflow:
                print(f"ERROR: Failed to load workflow for Shorts version {variation_num}")
                return False
            
            # Update workflow for Shorts (9:16 aspect ratio)
            workflow = self._update_prompt_text(workflow, prompt_text)
            workflow = self._update_workflow_resolution(workflow, SHORTS_WIDTH, SHORTS_HEIGHT)
            
            # Generate unique filename for this Shorts variation
            if isinstance(variation_num, str) and "." in str(variation_num):
                # Format: "1.1", "1.2", etc. for main variant.sub-variant
                shorts_filename = f"thumbnail.short.v{variation_num}"
            else:
                # Format: "1", "2", etc. for single variant
                shorts_filename = f"thumbnail.short.v{variation_num}"
            workflow = self._update_saveimage_prefix(workflow, shorts_filename)
            
            # Set unique seed for this variation
            seed = self._get_seed()
            workflow = self._update_workflow_seed(workflow, seed)
            print(f"  Seed set to: {seed}")
            
            # Print workflow summary
            self._print_workflow_summary(workflow, f"YouTube Shorts Version {variation_num}")
            
            # Submit workflow to ComfyUI
            resp = requests.post(f"{self.comfyui_url}prompt", json={"prompt": workflow}, timeout=60)
            if resp.status_code != 200:
                print(f"ERROR: ComfyUI API error for Shorts version {variation_num}: {resp.status_code} {resp.text}")
                return False
                
            prompt_id = resp.json().get("prompt_id")
            if not prompt_id:
                print(f"ERROR: No prompt ID returned for Shorts version {variation_num}")
                return False

            # Wait for completion
            print(f"  Waiting for Shorts version {variation_num} generation to complete...")
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
            generated_image = self._find_newest_output_with_prefix(shorts_filename)
            if not generated_image:
                print(f"ERROR: Could not find generated image for Shorts version {variation_num}")
                return False
            
            # Copy to final output location
            shutil.copy2(generated_image, output_path)
            
            # Apply title overlay if enabled
            title = self._read_title_from_file()
            if title and USE_TITLE_TEXT:
                self._overlay_title_on_shorts_image(output_path, title)
            
            print(f"Created YouTube Shorts version {variation_num}: {output_path}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to generate Shorts version {variation_num}: {e}")
            return False

    def _overlay_title_on_shorts_image(self, image_path: str, title: str) -> bool:
        """Apply title overlay to the shorts image file (9:16 format)."""
        try:
            img = Image.open(image_path).convert("RGBA")
            w, h = img.size  # Should be SHORTS_WIDTH x SHORTS_HEIGHT (1080x1920)
            
            # Measure title block for the full shorts canvas
            band_height, padding, lines, chosen_font, line_height, stroke_w = self._measure_title_block(
                title, w, h
            )
            
            if not lines:
                return True
            
            # Create title band and text
            band_opacity = int(0.60 * 255)
            band = Image.new("RGBA", (w, band_height), (0, 0, 0, band_opacity))
            shadow = band.copy().filter(ImageFilter.GaussianBlur(radius=int(h * 0.01)))
            text_layer = Image.new("RGBA", (w, band_height), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_layer)
            
            # Draw text lines
            y_cursor = padding
            for line in lines:
                bbox = text_draw.textbbox((0, 0), line, font=chosen_font, stroke_width=0)
                tw = bbox[2] - bbox[0]
                x = (w - tw) // 2
                text_draw.text((x, y_cursor), line, font=chosen_font, fill=(255, 255, 255, 255))
                y_cursor += line_height
            
            # Position the title band using same logic as main thumbnail
            pos = (TITLE_POSITION or "bottom").strip().lower()
            if pos in ("top", "upper"):
                band_y = 0
            elif pos in ("middle", "center", "centre"):
                band_y = max(0, (h - band_height) // 2)
            else:
                band_y = h - band_height
            
            # Apply shadow, band, and text to image
            shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            shadow_layer.paste(shadow, (0, band_y), shadow)
            img = Image.alpha_composite(img, shadow_layer)
            
            band_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            band_layer.paste(band, (0, band_y), band)
            img = Image.alpha_composite(img, band_layer)
            
            text_overlay_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            text_overlay_layer.paste(text_layer, (0, band_y), text_layer)
            img = Image.alpha_composite(img, text_overlay_layer)
            
            # Save back
            img.save(image_path)
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to overlay title on shorts image: {e}")
            return False

    def _overlay_title_on_shorts_canvas(self, canvas: Image.Image, title: str) -> bool:
        """Apply title overlay to the shorts canvas (9:16 format)."""
        try:
            w, h = canvas.size  # Should be SHORTS_WIDTH x SHORTS_HEIGHT (1080x1920)
            
            # Measure title block for the full shorts canvas
            band_height, padding, lines, chosen_font, line_height, stroke_w = self._measure_title_block(
                title, w, h
            )
            
            if not lines:
                return True
            
            # Create title band and text
            band_opacity = int(0.60 * 255)
            band = Image.new("RGBA", (w, band_height), (0, 0, 0, band_opacity))
            shadow = band.copy().filter(ImageFilter.GaussianBlur(radius=int(h * 0.01)))
            text_layer = Image.new("RGBA", (w, band_height), (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_layer)
            
            # Draw text lines
            y_cursor = padding
            for line in lines:
                bbox = text_draw.textbbox((0, 0), line, font=chosen_font, stroke_width=0)
                tw = bbox[2] - bbox[0]
                x = (w - tw) // 2
                text_draw.text((x, y_cursor), line, font=chosen_font, fill=(255, 255, 255, 255))
                y_cursor += line_height
            
            # Position the title band at the bottom of the shorts canvas
            band_y = h - band_height
            
            # Apply shadow, band, and text to canvas
            shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            shadow_layer.paste(shadow, (0, band_y), shadow)
            canvas.paste(shadow_layer, (0, 0), shadow_layer)
            
            band_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            band_layer.paste(band, (0, band_y), band)
            canvas.paste(band_layer, (0, 0), band_layer)
            
            text_overlay_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            text_overlay_layer.paste(text_layer, (0, band_y), text_layer)
            canvas.paste(text_overlay_layer, (0, 0), text_overlay_layer)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to overlay title on shorts canvas: {e}")
            return False

    def _get_master_prompt(self) -> str:
        """Get the master prompt content."""
        return """Create a 16K ultra-high-resolution, illustration in the style of {ART_STYLE}. The artwork should feature fine, intricate details and a natural sense of depth, with carefully chosen camera angle and focus to best frame the Scene. 
All Non-Living Objects mentioned in Scene text-description must be present in illustration.Must Always Precisely & Accurately Represent entire Scene including all Non-Living Objects according to scene text-description.
Must Always Precisely & Accurately Preserve each Character's Identity and Appearance(Properties like "Color", "Texture", "Shape", "Details", "Style", "Type") of Facial and Body Features as well as entire Clothing) from their respective reference image or image-section specified in Character's or Scene's text-description.
All other aspects of Characters is adaptable/must change according to Scene and Character text-description.Keep each Character's all "Features Separate and Discrete" from each other.
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

    result = processor.generate_thumbnail(processor._get_master_prompt() + "\n\n " + prompt)
    if result:
        if isinstance(result, list):
            for p in result:
                print(p)
        else:
            print(result)
    else:
        raise SystemExit(1)


