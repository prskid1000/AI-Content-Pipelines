import requests
import json
import time
import os
import re
import base64
from typing import List, Dict, Any, Optional
from functools import partial
import builtins as _builtins
from pathlib import Path
print = partial(_builtins.print, flush=True)

# Model constants for easy switching
MODEL_MOTION_GENERATION = "qwen_qwen3-vl-30b-a3b-instruct"  # Vision model for motion generation

# Feature flags
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them

# Prompt generation feature flags
USE_SCENE_IMAGE = True  # Set to True to include scene image in prompt generation
USE_DIALOGUE = False  # Set to True to include dialogue lines in prompt generation
USE_LOCATION = False  # Set to True to include location data from 3.location.txt
USE_CHARACTER = False  # Set to True to include character data from 2.character.txt
USE_SUMMARY_TEXT = False  # Set to True to use summary text files (3.character.txt, 3.location.txt) instead of 2.character.txt, 2.location.txt

# Resumable state management
class ResumableState:
    """Manages resumable state for expensive LLM operations."""
    
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
                print(f"WARNING: Failed to load checkpoint state: {ex}")
        return {
            "prompt_entries": {"completed": [], "results": {}},
            "metadata": {"start_time": time.time(), "last_update": time.time()}
        }
    
    def _save_state(self):
        """Save current state to checkpoint file."""
        try:
            self.state["metadata"]["last_update"] = time.time()
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as ex:
            print(f"WARNING: Failed to save checkpoint state: {ex}")
    
    def is_prompt_entry_complete(self, entry_key: str) -> bool:
        """Check if specific prompt entry is complete."""
        return entry_key in self.state["prompt_entries"]["completed"]
    
    def get_prompt_entry(self, entry_key: str) -> str | None:
        """Get cached prompt entry if available."""
        return self.state["prompt_entries"]["results"].get(entry_key)
    
    def set_prompt_entry(self, entry_key: str, master_prompt: str):
        """Set prompt entry and mark as complete."""
        if entry_key not in self.state["prompt_entries"]["completed"]:
            self.state["prompt_entries"]["completed"].append(entry_key)
        self.state["prompt_entries"]["results"][entry_key] = master_prompt
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
        prompt_done = len(self.state["prompt_entries"]["completed"])
        prompt_total = len(self.state["prompt_entries"]["results"]) + len([k for k in self.state["prompt_entries"]["results"].keys() if k not in self.state["prompt_entries"]["completed"]])
        
        return f"Progress: Prompt Entries({prompt_done}/{prompt_total})"

class PromptGenerator:
    def __init__(self, lm_studio_url="http://localhost:1234/v1", model=MODEL_MOTION_GENERATION, output_file="../input/2.motion.txt"):
        self.lm_studio_url = lm_studio_url
        self.output_file = output_file
        self.model = model
        
        # Scene image path configuration
        self.scene_image_base_path = "../../gen.image/output/scene"
        
        # Data file paths
        if USE_SUMMARY_TEXT:
            self.character_file = "../../gen.image/input/3.character.txt"
            self.location_file = "../../gen.image/input/3.location.txt"
        else:
            self.character_file = "../../gen.image/input/2.character.txt"
            self.location_file = "../../gen.image/input/2.location.txt"
        
        # Time estimation tracking
        self.processing_times = []
        self.start_time = None
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64 format for API input"""
        try:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_encoded = base64.b64encode(image_data).decode('utf-8')
                
                # Determine image format
                if image_path.lower().endswith('.png'):
                    mime_type = 'image/png'
                elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
                    mime_type = 'image/jpeg'
                elif image_path.lower().endswith('.webp'):
                    mime_type = 'image/webp'
                else:
                    mime_type = 'image/jpeg'  # Default fallback
                
                return f"data:{mime_type};base64,{base64_encoded}"
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def read_story_content(self, filename="../input/1.story.txt") -> str:
        """Read story content from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Story file '{filename}' not found.")
            return None
        except Exception as e:
            print(f"Error reading story file: {e}")
            return None
    
    def read_character_data(self) -> Dict[str, str]:
        """Read character data from file.
        
        Format: Each character on a single line as: ((character_name)): description
        """
        characters = {}
        if not USE_CHARACTER:
            return characters
        
        try:
            with open(self.character_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            entries = [entry.strip() for entry in content.split('\n') if entry.strip()]
            for entry in entries:
                match = re.match(r'\(\(([^)]+)\)\):\s*(.+)', entry, re.DOTALL)
                if match:
                    characters[match.group(1).strip()] = match.group(2).strip()
            if characters:
                print(f"📖 Loaded {len(characters)} character descriptions")
        except FileNotFoundError:
            print(f"⚠️ Character file not found: {self.character_file}")
        except Exception as e:
            print(f"ERROR: Failed to read character data: {e}")
        return characters
    
    def read_location_data(self) -> Dict[str, str]:
        """Read location data from file.
        
        Format: Each location on a single line as: {{loc_id}} description
        """
        locations = {}
        if not USE_LOCATION:
            return locations
        
        try:
            with open(self.location_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            for line in lines:
                match = re.match(r'\{\{([^}]+)\}\}\s*(.+)', line)
                if match:
                    locations[match.group(1).strip()] = match.group(2).strip()
            if locations:
                print(f"📍 Loaded {len(locations)} location descriptions")
        except FileNotFoundError:
            print(f"⚠️ Location file not found: {self.location_file}")
        except Exception as e:
            print(f"ERROR: Failed to read location data: {e}")
        return locations
    
    def parse_story_scenes(self, content: str) -> List[Dict[str, Any]]:
        """Parse story content into structured scene entries with dialogue"""
        scenes = []
        lines = content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for dialogue lines (character speaking)
            if line.startswith('[') and ']' in line:
                # Extract speaker and dialogue text separately (like 3.av.py does)
                dialogue_end = line.find(']')
                speaker = line[1:dialogue_end]
                dialogue_text = line[dialogue_end + 1:].strip()
                
                # Look for the next scene line
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('(scene_'):
                    j += 1
                
                if j < len(lines):
                    scene_line = lines[j].strip()
                    if scene_line.startswith('(scene_'):
                        # Extract scene number and content
                        scene_match = re.match(r'\(scene_(\d+\.\d+)\)', scene_line)
                        if scene_match:
                            scene_number = scene_match.group(1)
                            scene_content = scene_line[scene_match.end():].strip()
                            
                            # Check if corresponding scene image exists
                            scene_image_path = f"{self.scene_image_base_path}/scene_{scene_number}.png"
                            if os.path.exists(scene_image_path) or not USE_SCENE_IMAGE:
                                scenes.append({
                                    'scene_number': scene_number,
                                    'dialogue': dialogue_text,  # Store just the dialogue text (without speaker tag)
                                    'speaker': speaker,  # Store speaker separately
                                    'scene_content': scene_content,
                                    'image_path': scene_image_path if os.path.exists(scene_image_path) else None,
                                    'original_scene_line': scene_line
                                })
                            else:
                                print(f"Warning: Scene image not found for scene_{scene_number}.png")
                            i = j  # Skip to after the scene line
                        else:
                            i += 1
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
        
        print(f"📋 Parsed {len(scenes)} story scenes with dialogue and corresponding images")
        return scenes
    
    def build_raw_input_prompt(self, scene: Dict[str, Any], characters_data: Dict[str, str], locations_data: Dict[str, str]) -> str:
        """Build the raw input prompt from scene, dialogue, and optional data"""
        parts = []
        
        # Add dialogue (with speaker if available)
        if USE_DIALOGUE and scene.get('dialogue'):
            speaker = scene.get('speaker', '')
            if speaker:
                parts.append(f"DIALOGUE: [{speaker}] {scene['dialogue']}")
            else:
                parts.append(f"DIALOGUE: {scene['dialogue']}")
        
        # Add scene description
        if scene.get('scene_content'):
            scene_desc = scene['scene_content']
            
            # Replace location references if enabled
            if USE_LOCATION and locations_data:
                def replace_location(match):
                    full_match = match.group(0)
                    content = match.group(1).strip()
                    loc_id = content.split(',')[0].strip() if ',' in content else content
                    if loc_id in locations_data:
                        return locations_data[loc_id]
                    return full_match
                scene_desc = re.sub(r'\{\{([^}]+)\}\}', replace_location, scene_desc)
            
            parts.append(f"SCENE DESCRIPTION: {scene_desc}")
        
        # Add character information if enabled
        if USE_CHARACTER and characters_data:
            # Extract character references from speaker, dialogue, and scene
            char_refs = set()
            # Add speaker as character reference
            if scene.get('speaker'):
                char_refs.add(scene['speaker'])
            if scene.get('dialogue'):
                # Dialogue text might still have character references in quotes or elsewhere
                char_refs.update(re.findall(r'\[([^\]]+)\]', scene['dialogue']))
            if scene.get('scene_content'):
                char_refs.update(re.findall(r'\(\(([^)]+)\)\)', scene['scene_content']))
            
            if char_refs:
                char_info = []
                for char_name in char_refs:
                    if char_name in characters_data:
                        char_info.append(f"{char_name}: {characters_data[char_name]}")
                if char_info:
                    parts.append(f"CHARACTERS: {'; '.join(char_info)}")
        
        return "\n\n".join(parts)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for video prompt generation"""
        if USE_SCENE_IMAGE:
            # System prompt for when scene image is used (image-to-video)
            return """You are a Creative Assistant writing SIMPLIFIED, ESSENTIAL-ONLY image-to-video prompts. Given an image (first frame) and user Raw Input Prompt, generate a MINIMAL prompt with ONLY what can actually be rendered visually.

#### CRITICAL SIMPLIFICATION RULES:
- RETAIN ONLY: Main character (the one actually visible in image), essential setting (indoor/outdoor, key elements like fireplace/pond), basic clothing, weather/atmosphere (fog/sunlight) if clearly visible.
- REMOVE COMPLETELY: All audio descriptions (sounds, voices, footsteps, music). All secondary characters not visible in image. All specific objects that don't render (rings, lanterns, padlocks, tools, magnifying glasses, books, letters, carriages). All detailed actions that can't be captured (pointing, examining, speaking, reaching, trembling). All dialogue and speech. All emotional descriptions. All off-screen elements.

#### Guidelines:
- Analyze the Image: Identify ONLY what is actually visible - main subject, setting, essential elements.
- Follow image as ground truth: If the image shows one person, describe only that person. If no objects are visible, don't mention objects.
- Style: Include visual style at beginning: "Style: <style>, <rest of prompt>." If unclear, omit to avoid conflicts.
- Visual only: Describe ONLY what can be seen. NO audio, NO sounds, NO dialogue, NO actions that aren't visible.
- Ultra-minimal: Keep prompts SHORT. Only essential visual elements that actually appear in the image.

#### Important notes:
- DO NOT include audio descriptions of any kind.
- DO NOT include secondary characters unless clearly visible in image.
- DO NOT include specific objects (rings, tools, books, etc.) unless clearly visible.
- DO NOT include actions like "speaking", "pointing", "examining" - only static poses.
- DO NOT include dialogue or speech.
- Format: Start directly with Style (optional) and minimal scene description.
- Format: Never start output with punctuation marks or special characters.

#### Output Format (Strict):
- Single SHORT paragraph in natural English. NO titles, headings, prefaces, sections, code fences, or Markdown.
- Maximum 2-3 sentences. Only essential visual elements.

#### Example output:
Style: realistic - cinematic - The man sits in an armchair near the fireplace. Outside the large window, a cold winter morning with bare trees against a pale sky."""
        else:
            # System prompt for when scene image is NOT used (text-to-video)
            return """You are a Creative Assistant writing SIMPLIFIED, ESSENTIAL-ONLY video prompts. Given a user's raw input prompt, generate a MINIMAL prompt with ONLY what can actually be rendered visually.

#### CRITICAL SIMPLIFICATION RULES:
- RETAIN ONLY: Main character (one primary subject), essential setting (indoor/outdoor, key elements), basic clothing, weather/atmosphere if relevant.
- REMOVE COMPLETELY: All audio descriptions (sounds, voices, footsteps, music). All secondary characters. All specific objects that don't render reliably (rings, lanterns, padlocks, tools, magnifying glasses, books, letters, carriages). All detailed actions that can't be captured (pointing, examining, speaking, reaching). All dialogue and speech. All emotional descriptions.

#### Guidelines
- Focus on ONE main character only. If input mentions multiple characters, describe only the primary one.
- Essential setting only: Basic location (living room, park, garden) and key visible elements (fireplace, pond, trees).
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified.
- Visual only: Describe ONLY what can be seen. NO audio, NO sounds, NO dialogue, NO complex actions.
- Ultra-minimal: Keep prompts SHORT. Maximum 2-3 sentences.

#### Important notes:
- DO NOT include audio descriptions of any kind.
- DO NOT include secondary characters.
- DO NOT include specific objects (rings, tools, books, etc.) unless absolutely essential.
- DO NOT include actions like "speaking", "pointing", "examining" - only static poses.
- DO NOT include dialogue or speech.
- Format: Start directly with Style (optional) and minimal scene description.
- Format: DO NOT start your response with special characters.

#### Output Format (Strict):
- Single SHORT paragraph in natural language (English). Maximum 2-3 sentences.
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

#### Example
Input: "A woman at a coffee shop talking on the phone"
Output:
Style: realistic - cinematic - A woman sits at a table by a window. She wears a cream-colored sweater and holds a smartphone to her ear."""
    
    def call_lm_studio_api(self, raw_input_prompt: str, image_path: str = None) -> str:
        """Call LM Studio API to generate master prompt with optional image input"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Build user content - image first (if provided), then text
            user_content = []
            
            # Add image if provided and enabled (place image BEFORE text for vision models)
            if USE_SCENE_IMAGE and image_path and os.path.exists(image_path):
                base64_image = self.encode_image_to_base64(image_path)
                if base64_image:
                    # Verify base64 string is valid (starts with data: and contains base64)
                    if base64_image.startswith("data:") and len(base64_image) > 100:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        })
                        print(f"🖼️  Including image: {os.path.basename(image_path)} (base64 length: {len(base64_image)} chars)")
                    else:
                        print(f"⚠️  Invalid base64 image format: {image_path}")
                else:
                    print(f"⚠️  Failed to encode image: {image_path}")
            elif USE_SCENE_IMAGE and image_path and not os.path.exists(image_path):
                print(f"⚠️  Scene image not found: {image_path}")
            
            # Add text content (after image if image was added)
            user_content.append({
                "type": "text",
                "text": f"Raw Input Prompt:\n{raw_input_prompt}\n\nOnly use English Language for Input, Thinking, and Output\n/no_think"
            })
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": f"{self._build_system_prompt()}\nOnly use English Language for Input, Thinking, and Output\n/no_think"
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                "temperature": 1,
                "stream": False
            }
            
            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    return content
                else:
                    raise Exception("No content in API response")
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to LM Studio API. Make sure LM Studio is running on localhost:1234")
        except requests.exceptions.Timeout:
            raise Exception("API call timed out")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def parse_prompt_response(self, response: str) -> str:
        """Parse the prompt response from LM Studio"""
        # Clean up the response - remove any markdown code fences
        text = response.strip()
        
        # Remove code fences if present
        if text.startswith("```"):
            m = re.search(r"```(?:.*?)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
            if m:
                text = m.group(1).strip()
        
        # Remove any leading/trailing quotes
        text = text.strip('"\'')
        
        # Return cleaned text
        return text if text else "subtle movements and expressions"
    
    def estimate_remaining_time(self, current_scene: int, total_scenes: int, scene_processing_time: float = None, scene_content: str = None) -> str:
        """Estimate remaining time based on words per minute - simple and accurate approach"""
        
        # Calculate total words in all scene entries
        total_scene_words = getattr(self, 'total_scene_words', 0)
        
        # For first scene with no data, provide a reasonable initial estimate
        if not self.processing_times and scene_processing_time is None:
            if total_scene_words > 0:
                # Initial estimate: assume 1000 words per minute for scene processing (LLM text generation)
                words_per_minute = 1000
                remaining_words = total_scene_words - (current_scene - 1) * (total_scene_words // total_scenes)
                estimated_remaining_minutes = remaining_words / words_per_minute
                estimated_remaining_seconds = estimated_remaining_minutes * 60
            else:
                # Fallback: assume 10 seconds per scene
                remaining_scenes = total_scenes - current_scene
                estimated_remaining_seconds = remaining_scenes * 10.0
            
            return self._format_time_with_confidence(estimated_remaining_seconds, confidence="low")
        
        # Calculate actual words per minute from completed scenes
        if scene_processing_time and scene_content:
            scene_words = len(scene_content.split())
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
    
    def save_prompts_to_file(self, all_prompt_entries: List[Dict[str, Any]]) -> None:
        """Save all prompt entries to motion.txt (keeping same filename for compatibility)"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for entry in all_prompt_entries:
                    f.write(f"(motion_{entry['scene_number']}) {entry['master_prompt']}\n")
            
            print(f"💾 Saved {len(all_prompt_entries)} master prompts to {self.output_file}")
            
        except Exception as e:
            raise Exception(f"Failed to save prompt file: {str(e)}")
    
    def process_story(self, story_filename="../input/1.story.txt", resumable_state: ResumableState | None = None) -> bool:
        """Main processing function - process each scene individually"""
        print("🚀 Starting Master Prompt Generation...")
        
        print(f"📁 Reading story from: {story_filename}")
        
        # Read optional data files
        characters_data = self.read_character_data()
        locations_data = self.read_location_data()
        
        # Read story content and calculate total words for ETA calculations
        content = self.read_story_content(story_filename)
        if content is None:
            return False
        
        # Parse story scenes
        scenes = self.parse_story_scenes(content)
        if not scenes:
            print("❌ No valid story scenes found")
            return False
        
        # Calculate total words for ETA calculations
        self.total_scene_words = sum(len(scene['scene_content'].split()) + len(scene.get('dialogue', '').split()) for scene in scenes)
        print(f"📊 Found {len(scenes)} story scenes")
        print(f"📊 Story contains {self.total_scene_words:,} words")
        
        # Initialize start time for time estimation
        self.start_time = time.time()
        
        # Process each scene individually
        all_prompt_entries = []
        
        print(f"\n📊 MASTER PROMPT GENERATION PROGRESS")
        print("=" * 100)
        
        for i, scene in enumerate(scenes):
            # Create unique key for this scene
            scene_key = f"scene_{scene['scene_number']}:{scene.get('dialogue', '')[:50]}"
            
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_prompt_entry_complete(scene_key):
                cached_prompt = resumable_state.get_prompt_entry(scene_key)
                if cached_prompt:
                    eta = self.estimate_remaining_time(i+1, len(scenes), scene_content=scene['scene_content'] + " " + scene.get('dialogue', ''))
                    print(f"💾 Scene {i+1}/{len(scenes)} ({scene['scene_number']}) - {scene.get('dialogue', '')[:50]} - Cached")
                    print(f"📊 Estimated time remaining: {eta}")
                    prompt_entry = {
                        'scene_number': scene['scene_number'],
                        'master_prompt': cached_prompt
                    }
                    all_prompt_entries.append(prompt_entry)
                    continue
            
            scene_start_time = time.time()
            eta = self.estimate_remaining_time(i+1, len(scenes), scene_content=scene['scene_content'] + " " + scene.get('dialogue', ''))
            print(f"🔄 Scene {i+1}/{len(scenes)} ({scene['scene_number']}) - {scene.get('dialogue', '')[:50]} - Processing...")
            print(f"📊 Estimated time remaining: {eta}")
            
            # Build raw input prompt
            raw_input_prompt = self.build_raw_input_prompt(scene, characters_data, locations_data)
            
            try:
                # Call LM Studio API with image (if enabled)
                image_path = scene.get('image_path') if USE_SCENE_IMAGE else None
                response = self.call_lm_studio_api(raw_input_prompt, image_path)
                
                # Parse prompt response
                master_prompt = self.parse_prompt_response(response)
                
                # Create output entry
                prompt_entry = {
                    'scene_number': scene['scene_number'],
                    'master_prompt': master_prompt
                }
                
                # Add to all entries
                all_prompt_entries.append(prompt_entry)
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_prompt_entry(scene_key, master_prompt)
                
                scene_processing_time = time.time() - scene_start_time
                self.processing_times.append(scene_processing_time)
                eta = self.estimate_remaining_time(i+1, len(scenes), scene_processing_time, scene['scene_content'] + " " + scene.get('dialogue', ''))
                print(f"✅ Scene {i+1}/{len(scenes)} ({scene['scene_number']}) - {scene.get('dialogue', '')[:50]} - Completed in {self.format_processing_time(scene_processing_time)}")
                print(f"📊 Estimated time remaining: {eta}")
                
                # Live preview for this scene
                print(f"🎬 Output: (motion_{scene['scene_number']}) {master_prompt[:100]}...")
                
            except Exception as e:
                scene_processing_time = time.time() - scene_start_time
                self.processing_times.append(scene_processing_time)
                eta = self.estimate_remaining_time(i+1, len(scenes), scene_processing_time, scene['scene_content'] + " " + scene.get('dialogue', ''))
                print(f"❌ Scene {i+1}/{len(scenes)} ({scene['scene_number']}) - {scene.get('dialogue', '')[:50]} - Error after {self.format_processing_time(scene_processing_time)}")
                print(f"📊 Estimated time remaining: {eta}")
                print(f"❌ Error processing scene {i+1}: {str(e)}")
                # Continue with next scene instead of failing completely
                all_prompt_entries.append({
                    'scene_number': scene['scene_number'],
                    'master_prompt': 'subtle movements and expressions'
                })
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_prompt_entry(scene_key, 'subtle movements and expressions')
            
            # Small delay between API calls
            if i < len(scenes) - 1:
                time.sleep(1)
        
        # Save all prompt entries to file
        try:
            self.save_prompts_to_file(all_prompt_entries)
            print(f"\n🎉 Master prompt generation completed successfully!")
            print(f"📄 Output saved to: {self.output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving prompt file: {str(e)}")
            return False

def main():
    """Main function"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate master prompts for story scenes")
    parser.add_argument("story_file", nargs="?", default="../input/1.story.txt",
                       help="Path to story file (default: ../input/1.story.txt)")
    parser.add_argument("--output", default="../input/2.motion.txt",
                       help="Output file path for motion prompts (default: ../input/2.motion.txt)")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")

    args = parser.parse_args()
        
    # Check if story file exists
    if not os.path.exists(args.story_file):
        print(f"❌ Story file '{args.story_file}' not found")
        print("Usage: python 2.motion.py [story_file] [--output OUTPUT_FILE] [--force-start]")
        return 1
    
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
    
    # Create generator and process
    generator = PromptGenerator(output_file=args.output)
    
    start_time = time.time()
    success = generator.process_story(args.story_file, resumable_state)
    end_time = time.time()
    
    if success:
        print(f"⏱️  Total processing time: {end_time - start_time:.2f} seconds")
        
        # Clean up checkpoint files if resumable mode was used and everything completed successfully
        if resumable_state:
            print("All operations completed successfully")
            print("Final progress:", resumable_state.get_progress_summary())
            resumable_state.cleanup()
        
        return 0
    else:
        print("❌ Processing failed")
        return 1

if __name__ == "__main__":
    exit(main())
