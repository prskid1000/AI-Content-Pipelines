import requests
import json
import time
import os
import re
import base64
from typing import List, Dict, Any
from functools import partial
import builtins as _builtins
from pathlib import Path
print = partial(_builtins.print, flush=True)

# Model constants for easy switching
MODEL_MOTION_GENERATION = "qwen_qwen3-vl-30b-a3b-instruct"  # Vision model for motion generation

# Motion description word limits
MOTION_DESCRIPTION_MIN_WORDS = 30  # Minimum words in motion description
MOTION_DESCRIPTION_MAX_WORDS = 300  # Maximum words in motion description
WORD_FACTOR = 6

# Feature flags
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them

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
            "motion_entries": {"completed": [], "results": {}},
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
    
    def is_motion_entry_complete(self, entry_key: str) -> bool:
        """Check if specific motion entry is complete."""
        return entry_key in self.state["motion_entries"]["completed"]
    
    def get_motion_entry(self, entry_key: str) -> str | None:
        """Get cached motion entry if available."""
        return self.state["motion_entries"]["results"].get(entry_key)
    
    def set_motion_entry(self, entry_key: str, motion_description: str):
        """Set motion entry and mark as complete."""
        if entry_key not in self.state["motion_entries"]["completed"]:
            self.state["motion_entries"]["completed"].append(entry_key)
        self.state["motion_entries"]["results"][entry_key] = motion_description
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
        motion_done = len(self.state["motion_entries"]["completed"])
        motion_total = len(self.state["motion_entries"]["results"]) + len([k for k in self.state["motion_entries"]["results"].keys() if k not in self.state["motion_entries"]["completed"]])
        
        return f"Progress: Motion Entries({motion_done}/{motion_total})"

class MotionGenerator:
    def __init__(self, lm_studio_url="http://localhost:1234/v1", model=MODEL_MOTION_GENERATION, use_json_schema=True):
        self.lm_studio_url = lm_studio_url
        self.output_file = "../input/2.motion.txt"
        self.model = model
        self.use_json_schema = use_json_schema
        
        # Scene image path configuration
        self.scene_image_base_path = "../../gen.image/output/scene"
        
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
    
    def parse_story_scenes(self, content: str) -> List[Dict[str, Any]]:
        """Parse story content into structured scene entries with dialogue"""
        scenes = []
        lines = content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for dialogue lines (character speaking)
            if line.startswith('[') and ']' in line:
                dialogue = line
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
                            if os.path.exists(scene_image_path):
                                scenes.append({
                                    'scene_number': scene_number,
                                    'dialogue': dialogue,
                                    'scene_content': scene_content,
                                    'image_path': scene_image_path,
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
    
    def create_prompt_for_single_scene(self, scene: Dict[str, Any]) -> str:
        """Create the prompt for a single scene with dialogue"""
        prompt = f"""DIALOGUE: {scene['dialogue']}

SCENE DESCRIPTION: {scene['scene_content']}

IMPORTANT: Only describe motions for objects, characters, and elements that are ACTUALLY VISIBLE in the provided image. Do not invent or describe motions for objects that are not present in the image."""
        return prompt

    def _build_response_format(self) -> Dict[str, Any]:
        """Build a simple JSON Schema response format for single entry output."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "motion_entry",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "motion_description": {
                            "minLength": WORD_FACTOR * MOTION_DESCRIPTION_MIN_WORDS,
                            "maxLength": WORD_FACTOR * MOTION_DESCRIPTION_MAX_WORDS,
                            "type": "string",
                            "description": f"Motion description of objects, characters, and elements that are ACTUALLY VISIBLE in the provided image."
                        }
                    },
                    "required": ["motion_description"]
                },
                "strict": True
            }
        }

    def _build_system_prompt(self) -> str:
        return f"""Generate motion descriptions for audio-visual video generation.
RULES:
- Only describe motions for elements VISIBLE in the image, not based on dialogue or scene description.
- Give the scene description in short, concise, and general terms appropriate for video generation.
- Give generic descriptions of motions, not specific to any one character or object.
- **IMPORTANT**: Use placeholders like "the character", "the person", "the figure", etc. instead of proper nouns or names when describing generic motions.

OUTPUT: A single paragraph of accurate and concise generalized motion+scene descriptions for video generation."""

    def call_lm_studio_api(self, prompt: str, image_path: str = None) -> str:
        """Call LM Studio API to generate motion for a single scene with optional image input"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Build user content with optional image
            user_content = [
                {
                    "type": "text",
                    "text": f"{prompt}\nOnly use English Language for Input, Thinking, and Output\n/no_think"
                }
            ]
            
            # Add image if provided
            if image_path and os.path.exists(image_path):
                base64_image = self.encode_image_to_base64(image_path)
                if base64_image:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    })
                    print(f"🖼️  Including image: {os.path.basename(image_path)}")
                else:
                    print(f"⚠️  Failed to encode image: {image_path}")
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": 
f"{self._build_system_prompt()}\nOnly use English Language for Input, Thinking, and Output\n/no_think"
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                "temperature": 1,
                "stream": False
            }

            # Request structured output
            payload["response_format"] = self._build_response_format()
            
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
    
    def parse_motion_response(self, response: str) -> str:
        """Parse the motion response from LM Studio for a single entry"""
        # Try JSON first
        text = response.strip()
        # Remove code fences if present
        if text.startswith("```"):
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
            if m:
                text = m.group(1).strip()
        # Fallback: extract braces region
        if not text.startswith("{"):
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                text = text[first:last+1]
        
        try:
            json_obj = json.loads(text)
            if isinstance(json_obj, dict) and "motion_description" in json_obj:
                return json_obj["motion_description"]
        except Exception:
            pass
        
        # Fallback: try to extract description from response
        for line in response.strip().split('\n'):
            line = line.strip()
            if ':' in line and '"' in line:
                try:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        content_part = parts[1].strip()
                        if content_part.startswith('"') and content_part.endswith('"'):
                            return content_part[1:-1]  # Remove quotes
                        else:
                            return content_part.strip('"')
                except Exception:
                    continue
        
        # Default fallback
        return "subtle movements and expressions"
    
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
    
    def save_motion_to_file(self, all_motion_entries: List[Dict[str, Any]]) -> None:
        """Save all motion entries to motion.txt"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for entry in all_motion_entries:
                    f.write(f"(motion_{entry['scene_number']}) {entry['motion_description']}\n")
            
            print(f"💾 Saved {len(all_motion_entries)} motion entries to {self.output_file}")
            
        except Exception as e:
            raise Exception(f"Failed to save motion file: {str(e)}")
    
    def process_story(self, story_filename="../input/1.story.txt", resumable_state: ResumableState | None = None) -> bool:
        """Main processing function - process each scene individually"""
        print("🚀 Starting Motion Generation...")
        
        print(f"📁 Reading story from: {story_filename}")
        
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
        self.total_scene_words = sum(len(scene['scene_content'].split()) + len(scene['dialogue'].split()) for scene in scenes)
        print(f"📊 Found {len(scenes)} story scenes")
        print(f"📊 Story contains {self.total_scene_words:,} words")
        
        # Initialize start time for time estimation
        self.start_time = time.time()
        
        # Process each scene individually
        all_motion_entries = []
        
        print(f"\n📊 MOTION GENERATION PROGRESS")
        print("=" * 100)
        
        for i, scene in enumerate(scenes):
            # Create unique key for this scene with scene_ prefix
            scene_key = f"scene_{scene['scene_number']}:{scene['dialogue'][:50]}"
            
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_motion_entry_complete(scene_key):
                cached_motion = resumable_state.get_motion_entry(scene_key)
                if cached_motion:
                    eta = self.estimate_remaining_time(i+1, len(scenes), scene_content=scene['scene_content'] + " " + scene['dialogue'])
                    print(f"💾 Scene {i+1}/{len(scenes)} ({scene['scene_number']}) - {scene['dialogue'][:50]} - Cached")
                    print(f"📊 Estimated time remaining: {eta}")
                    motion_entry = {
                        'scene_number': scene['scene_number'],
                        'motion_description': cached_motion
                    }
                    all_motion_entries.append(motion_entry)
                    continue
            
            scene_start_time = time.time()
            eta = self.estimate_remaining_time(i+1, len(scenes), scene_content=scene['scene_content'] + " " + scene['dialogue'])
            print(f"🔄 Scene {i+1}/{len(scenes)} ({scene['scene_number']}) - {scene['dialogue'][:50]} - Processing...")
            print(f"📊 Estimated time remaining: {eta}")
            
            # Create prompt for this single scene
            prompt = self.create_prompt_for_single_scene(scene)
            
            try:
                # Call LM Studio API with image
                response = self.call_lm_studio_api(prompt, scene['image_path'])
                
                # Parse motion response
                motion_description = self.parse_motion_response(response)
                
                # Create output entry
                motion_entry = {
                    'scene_number': scene['scene_number'],
                    'motion_description': motion_description
                }
                
                # Add to all entries
                all_motion_entries.append(motion_entry)
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_motion_entry(scene_key, motion_description)
                
                scene_processing_time = time.time() - scene_start_time
                self.processing_times.append(scene_processing_time)
                eta = self.estimate_remaining_time(i+1, len(scenes), scene_processing_time, scene['scene_content'] + " " + scene['dialogue'])
                print(f"✅ Scene {i+1}/{len(scenes)} ({scene['scene_number']}) - {scene['dialogue'][:50]} - Completed in {self.format_processing_time(scene_processing_time)}")
                print(f"📊 Estimated time remaining: {eta}")
                
                # Live preview for this scene
                print(f"🎬 Output: (motion_{scene['scene_number']}) {motion_description}")
                
            except Exception as e:
                scene_processing_time = time.time() - scene_start_time
                self.processing_times.append(scene_processing_time)
                eta = self.estimate_remaining_time(i+1, len(scenes), scene_processing_time, scene['scene_content'] + " " + scene['dialogue'])
                print(f"❌ Scene {i+1}/{len(scenes)} ({scene['scene_number']}) - {scene['dialogue'][:50]} - Error after {self.format_processing_time(scene_processing_time)}")
                print(f"📊 Estimated time remaining: {eta}")
                print(f"❌ Error processing scene {i+1}: {str(e)}")
                # Continue with next scene instead of failing completely
                all_motion_entries.append({
                    'scene_number': scene['scene_number'],
                    'motion_description': 'subtle movements and expressions'
                })
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_motion_entry(scene_key, 'subtle movements and expressions')
            
            # Small delay between API calls
            if i < len(scenes) - 1:
                time.sleep(1)
        
        # Save all motion entries to file
        try:
            self.save_motion_to_file(all_motion_entries)
            print(f"\n🎉 Motion generation completed successfully!")
            print(f"📄 Output saved to: {self.output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving motion file: {str(e)}")
            return False

def main():
    """Main function"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate motion descriptions for AV story scenes")
    parser.add_argument("story_file", nargs="?", default="../input/1.story.txt",
                       help="Path to story file (default: ../input/1.story.txt)")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")

    args = parser.parse_args()
        
    # Check if story file exists
    if not os.path.exists(args.story_file):
        print(f"❌ Story file '{args.story_file}' not found")
        print("Usage: python 2.motion.py [story_file] [--force-start]")
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
    generator = MotionGenerator()
    
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
