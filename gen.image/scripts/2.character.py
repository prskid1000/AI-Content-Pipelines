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
IMAGE_MEGAPIXEL = "0.3"
IMAGE_ASPECT_RATIO = "9:32 (Skyline)"
IMAGE_DIVISIBLE_BY = "64"
IMAGE_CUSTOM_RATIO = False
IMAGE_CUSTOM_ASPECT_RATIO = "1:1"

# Image Output Dimension Constants
USE_FIXED_DIMENSIONS = False  # Set to True to use fixed width/height, False to use aspect ratio calculation
IMAGE_OUTPUT_WIDTH = 256
IMAGE_OUTPUT_HEIGHT = 1024

# LoRA Configuration
USE_LORA = True  # Set to False to disable LoRA usage in workflow
LORA_NAME = "FLUX.1-Turbo-Alpha.safetensors"  # LoRA file name
LORA_STRENGTH_MODEL = 2.0  # LoRA strength for the model (0.0 - 2.0)
LORA_STRENGTH_CLIP = 2.0   # LoRA strength for CLIP (0.0 - 2.0)

# Sampling Configuration
SAMPLING_STEPS = 9  # Number of sampling steps (higher = better quality, slower)

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
        self.input_file = "../input/2.character.txt"
        # Dynamic workflow file selection based on mode
        self.workflow_file = "../workflow/character.flux.json" if self.mode == "flux" else "../workflow/character.json"

        # Create output directory
        os.makedirs(self.final_output_dir, exist_ok=True)

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
            # Add LoRA loader node
            workflow["43"] = {
                "inputs": {
                    "lora_name": LORA_NAME,
                    "strength_model": LORA_STRENGTH_MODEL,
                    "strength_clip": LORA_STRENGTH_CLIP,
                    "model": self._find_node_by_class(workflow, "UnetLoaderGGUF") or ["1", 0],
                    "clip": self._find_node_by_class(workflow, ["DualCLIPLoader", "TripleCLIPLoader"]) or ["2", 0]
                },
                "class_type": "LoraLoader",
                "_meta": {"title": "Load LoRA"}
            }
            
            # Update nodes to use LoRA outputs
            self._update_node_connections(workflow, "KSampler", "model", ["43", 0])
            self._update_node_connections(workflow, ["CLIPTextEncode", "CLIP Text Encode (Prompt)"], "clip", ["43", 1])
            
            print("LoRA enabled in character workflow")
        else:
            # Remove LoRA node if it exists
            if "43" in workflow:
                del workflow["43"]
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
        prompt = f"Create a 16K ultra-high-resolution, Full Body Visible, Illustration in the style of {ART_STYLE} in which torso, limbs, hands, feet, face(eyes, nose, mouth, skin), clothes, ornaments, props, precisely and accurately matching character with description, fine-level detailing and vibrant colors, and any part not cropped or hidden.Must use White Background.\n\n Character Name = {character_name}. \n\n Character Description = {description}."
        self._update_node_connections(workflow, ["CLIPTextEncode", "CLIP Text Encode (Prompt)"], "text", prompt)
        return workflow

    def _update_workflow_filename(self, workflow: dict, character_name: str) -> dict:
        """Update the workflow to save with character name as filename."""
        clean_name = re.sub(r'[^\w\s-]', '', character_name).strip()
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
        """Update the workflow with dynamic resolution settings."""
        # Handle Flux workflow with FluxResolutionNode
        flux_resolution_node = self._find_node_by_class(workflow, "FluxResolutionNode")
        if flux_resolution_node:
            node_id = flux_resolution_node[0]
            workflow[node_id]["inputs"]["megapixel"] = IMAGE_MEGAPIXEL
            workflow[node_id]["inputs"]["aspect_ratio"] = IMAGE_ASPECT_RATIO
            workflow[node_id]["inputs"]["divisible_by"] = IMAGE_DIVISIBLE_BY
            workflow[node_id]["inputs"]["custom_ratio"] = IMAGE_CUSTOM_RATIO
            workflow[node_id]["inputs"]["custom_aspect_ratio"] = IMAGE_CUSTOM_ASPECT_RATIO
            print(f"Updated Flux resolution settings: {IMAGE_MEGAPIXEL}MP, {IMAGE_ASPECT_RATIO}")
        
        # Handle Diffusion workflow with EmptySD3LatentImage
        latent_image_node = self._find_node_by_class(workflow, "EmptySD3LatentImage")
        if latent_image_node:
            node_id = latent_image_node[0]
            
            # Use fixed dimensions if specified
            if USE_FIXED_DIMENSIONS:
                # Bypass aspect ratio calculation and set fixed dimensions directly
                workflow[node_id]["inputs"]["width"] = IMAGE_OUTPUT_WIDTH
                workflow[node_id]["inputs"]["height"] = IMAGE_OUTPUT_HEIGHT
                print(f"Using fixed dimensions: {IMAGE_OUTPUT_WIDTH}x{IMAGE_OUTPUT_HEIGHT} (bypassing aspect ratio calculation)")
            else:
                # Calculate dimensions based on aspect ratio and megapixels
                try:
                    if ":" in IMAGE_ASPECT_RATIO:
                        # Extract just the ratio part before any parentheses
                        ratio_parts = IMAGE_ASPECT_RATIO.split("(")[0].strip()
                        width_ratio, height_ratio = map(int, ratio_parts.split(":"))
                        total_pixels = float(IMAGE_MEGAPIXEL) * 1000000
                        aspect_ratio = width_ratio / height_ratio
                        height = int((total_pixels / aspect_ratio) ** 0.5)
                        width = int(height * aspect_ratio)
                        divisible_by = int(IMAGE_DIVISIBLE_BY)
                        width = (width // divisible_by) * divisible_by
                        height = (height // divisible_by) * divisible_by
                        workflow[node_id]["inputs"]["width"] = width
                        workflow[node_id]["inputs"]["height"] = height
                        print(f"Using calculated dimensions: {width}x{height}")
                except Exception as e:
                    print(f"Warning: Could not parse aspect ratio {IMAGE_ASPECT_RATIO}: {e}")
        
        return workflow

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
            
            print(f"Generating image for: {character_name}")
            
            # Load and update workflow
            workflow = self._load_character_workflow()
            if not workflow:
                return None
                
            workflow = self._update_workflow_prompt(workflow, character_name, description)
            workflow = self._update_workflow_filename(workflow, character_name)
            workflow = self._update_workflow_seed(workflow)
            workflow = self._update_workflow_resolution(workflow)

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
