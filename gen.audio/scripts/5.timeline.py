import requests
import json
import time
import os
import re
from typing import List, Dict, Any
from functools import partial
import builtins as _builtins
from pathlib import Path
print = partial(_builtins.print, flush=True)

# Model constants for easy switching
MODEL_TIMELINE_GENERATION = "qwen/qwen3-14b"  # Model for timeline SFX generation

# Feature flags
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them
ENABLE_THINKING = False  # Set to True to enable thinking in LM Studio responses

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
            "sfx_entries": {"completed": [], "results": {}},
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
    
    def is_sfx_entry_complete(self, entry_key: str) -> bool:
        """Check if specific SFX entry is complete."""
        return entry_key in self.state["sfx_entries"]["completed"]
    
    def get_sfx_entry(self, entry_key: str) -> str | None:
        """Get cached SFX entry if available."""
        return self.state["sfx_entries"]["results"].get(entry_key)
    
    def set_sfx_entry(self, entry_key: str, sfx_description: str):
        """Set SFX entry and mark as complete."""
        if entry_key not in self.state["sfx_entries"]["completed"]:
            self.state["sfx_entries"]["completed"].append(entry_key)
        self.state["sfx_entries"]["results"][entry_key] = sfx_description
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
        sfx_done = len(self.state["sfx_entries"]["completed"])
        sfx_total = len(self.state["sfx_entries"]["results"]) + len([k for k in self.state["sfx_entries"]["results"].keys() if k not in self.state["sfx_entries"]["completed"]])
        
        return f"Progress: SFX Entries({sfx_done}/{sfx_total})"

class TimelineSFXGenerator:
    def __init__(self, lm_studio_url="http://localhost:1234/v1", model=MODEL_TIMELINE_GENERATION, use_json_schema=True):
        self.lm_studio_url = lm_studio_url
        self.output_file = "../input/3.timing.txt"
        self.model = model
        self.use_json_schema = use_json_schema
        
        # Time estimation tracking
        self.processing_times = []
        self.start_time = None
        
    def read_timeline_content(self, filename="../input/2.timeline.txt") -> str:
        """Read timeline content from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Timeline file '{filename}' not found.")
            return None
        except Exception as e:
            print(f"Error reading timeline file: {e}")
            return None
    
    def parse_timeline_entries(self, content: str) -> List[Dict[str, Any]]:
        """Parse timeline content into structured entries"""
        entries = []
        for line in content.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    try:
                        seconds = float(parts[0].strip())
                        description = parts[1].strip()
                        entries.append({
                            'seconds': seconds,
                            'description': description,
                            'original_line': line
                        })
                    except ValueError:
                        print(f"Warning: Invalid duration format in line: {line}")
                        continue
        
        print(f"📋 Parsed {len(entries)} timeline entries")
        return entries
    
    def create_prompt_for_single_entry(self, entry: Dict[str, Any]) -> str:
        """Create the prompt for a single timeline entry"""
        prompt = f"""CONTENT:{entry['seconds']} seconds: {entry['description']}"""
        return prompt

    def _is_dot_only_or_short(self, text: str) -> bool:
        """Return True if text is comprised only of dots or has <= 3 words."""
        if text is None:
            return True
        stripped = text.strip()
        # Treat unicode ellipsis as dots as well
        normalized = stripped.replace('…', '.')
        # Check if the sentence contains only dots
        if normalized != '' and all(ch == '.' for ch in normalized):
            return True
        # Count word-like tokens (alphanumeric/underscore sequences)
        word_tokens = re.findall(r"\w+", stripped, flags=re.UNICODE)
        return len(word_tokens) <= 3

    def _build_response_format(self) -> Dict[str, Any]:
        """Build a simple JSON Schema response format for single entry output."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "sfx_entry",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sound_or_silence_description": {"type": "string"}
                    },
                    "required": ["sound_or_silence_description"]
                },
                "strict": True
            }
        }
    
    def call_lm_studio_api(self, prompt: str) -> str:
        """Call LM Studio API to generate SFX for a single entry"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": 
"""You are an SFX(Sound or Silence) generator for Sound Generating AI Models.

RULES:
- Keep descriptions under 12 words, concrete, specific, unambiguous, descriptive(pitch, amplitude, timbre, sonance, frequency, etc.) and present tense.
- If no clear Sound related words or an important Action/Object that is producing or can produce sound is present in the transcript line, use 'Silence'; invent nothing yourself.
- No speech, lyrics, music, or vocal sounds allowed;use "Silence". May generate sounds(Diegetic/Non-diegetic) like atmosphere/ambience/background/noise/foley deduced from the transcript line.
- You must output only sound descriptions, any other sensory descriptions like visual, touch, smell, taste, etc. are not allowed;use "Silence".
- Return only JSON matching the schema.

OUTPUT: JSON with sound_or_silence_description field only."""
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}{'' if ENABLE_THINKING else '\n/no_think'}"
                    }
                ],
                "temperature": 1,
                "max_tokens": 512,
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
    
    def parse_sfx_response(self, response: str) -> str:
        """Parse the SFX response from LM Studio for a single entry"""
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
            if isinstance(json_obj, dict) and "sound_or_silence_description" in json_obj:
                return json_obj["sound_or_silence_description"]
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
        return "Silence"
    
    def estimate_remaining_time(self, current_entry: int, total_entries: int, entry_processing_time: float = None, entry_description: str = None) -> str:
        """Estimate remaining time based on processing history and content characteristics"""
        if not self.processing_times:
            return "No data available"
        
        # Calculate base average processing time per entry using ALL previous entries
        avg_time_per_entry = sum(self.processing_times) / len(self.processing_times)
        
        # If we have current entry processing time, include it in the calculation
        if entry_processing_time:
            # Use all previous entries plus current entry for more accurate estimation
            all_times = self.processing_times + [entry_processing_time]
            estimated_time_per_entry = sum(all_times) / len(all_times)
        else:
            estimated_time_per_entry = avg_time_per_entry
        
        # Apply content-based adjustments if we have entry description
        if entry_description:
            word_count = len(entry_description.split())
            char_count = len(entry_description)
            
            # Calculate complexity factor based on content characteristics
            # Longer descriptions with more complex content take longer to process
            word_factor = 1.0 + (word_count - 10) * 0.02  # Base 10 words, +2% per word over/under
            char_factor = 1.0 + (char_count - 50) * 0.002  # Base 50 chars, +0.2% per char over/under
            
            # Check for content type complexity
            content_complexity = 1.0
            if any(word in entry_description.lower() for word in ['action', 'sound', 'music', 'dialogue', 'speech']):
                content_complexity = 1.2  # 20% longer for action/sound content
            elif any(word in entry_description.lower() for word in ['silence', 'quiet', 'pause']):
                content_complexity = 0.8  # 20% shorter for silence content
            
            # Combine factors (cap at reasonable bounds)
            complexity_factor = min(2.0, max(0.5, (word_factor + char_factor) / 2 * content_complexity))
            estimated_time_per_entry *= complexity_factor
        
        remaining_entries = total_entries - current_entry
        estimated_remaining_seconds = remaining_entries * estimated_time_per_entry
        
        # Convert to human readable format
        if estimated_remaining_seconds < 60:
            return f"~{estimated_remaining_seconds:.0f}s"
        elif estimated_remaining_seconds < 3600:
            minutes = estimated_remaining_seconds / 60
            return f"~{minutes:.1f}m"
        else:
            hours = estimated_remaining_seconds / 3600
            return f"~{hours:.1f}h"
    
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
    
    def save_sfx_to_file(self, all_sfx_entries: List[Dict[str, Any]]) -> None:
        """Save all SFX entries to sfx.txt"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for entry in all_sfx_entries:
                    f.write(f"{entry['seconds']}: {entry['sound_or_silence_description']}\n")
            
            total_duration = sum(entry['seconds'] for entry in all_sfx_entries)
            print(f"💾 Saved {len(all_sfx_entries)} SFX entries to {self.output_file}")
            print(f"⏱️  Total duration: {total_duration:.3f} seconds ({total_duration/60:.2f} minutes)")
            
        except Exception as e:
            raise Exception(f"Failed to save SFX file: {str(e)}")
    
    def process_timeline(self, timeline_filename="../input/2.timeline.txt", resumable_state: ResumableState | None = None) -> bool:
        """Main processing function - process each entry individually"""
        print("🚀 Starting Timeline SFX Generation...")
        print(f"📁 Reading timeline from: {timeline_filename}")
        
        # Read timeline content
        content = self.read_timeline_content(timeline_filename)
        if content is None:
            return False
        
        # Parse timeline entries
        entries = self.parse_timeline_entries(content)
        if not entries:
            print("❌ No valid timeline entries found")
            return False
        
        # Initialize start time for time estimation
        self.start_time = time.time()
        
        # Process each entry individually
        all_sfx_entries = []
        
        print(f"\n📊 TIMELINE SFX GENERATION PROGRESS")
        print("=" * 100)
        print(f"{'Entry':<6} {'Duration':<10} {'Description':<50} {'Status':<15} {'Time':<10} {'ETA':<10}")
        print("-" * 100)
        
        for i, entry in enumerate(entries):
            # Create unique key for this entry
            entry_key = f"{entry['seconds']}:{entry['description'][:50]}"
            
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_sfx_entry_complete(entry_key):
                cached_sfx = resumable_state.get_sfx_entry(entry_key)
                if cached_sfx:
                    eta = self.estimate_remaining_time(i+1, len(entries), entry_description=entry['description'])
                    print(f"{i+1:<6} {entry['seconds']:<10.3f} {entry['description'][:50]:<50} {'CACHED':<15} {'--':<10} {eta:<10}")
                    sfx_entry = {
                        'seconds': entry['seconds'],
                        'sound_or_silence_description': cached_sfx
                    }
                    all_sfx_entries.append(sfx_entry)
                    continue
            
            entry_start_time = time.time()
            eta = self.estimate_remaining_time(i+1, len(entries), entry_description=entry['description'])
            print(f"{i+1:<6} {entry['seconds']:<10.3f} {entry['description'][:50]:<50} {'PROCESSING':<15} {'--':<10} {eta:<10}")
            
            # Create prompt for this single entry
            prompt = self.create_prompt_for_single_entry(entry)
            
            # Rule: mark as Silence without calling LM Studio if only dots or <=3 words
            if self._is_dot_only_or_short(entry['description']):
                sfx_entry = {
                    'seconds': entry['seconds'],
                    'sound_or_silence_description': 'Silence'
                }
                all_sfx_entries.append(sfx_entry)
                
                entry_processing_time = time.time() - entry_start_time
                self.processing_times.append(entry_processing_time)
                eta = self.estimate_remaining_time(i+1, len(entries), entry_processing_time, entry['description'])
                print(f"{i+1:<6} {entry['seconds']:<10.3f} {entry['description'][:50]:<50} {'SKIPPED':<15} {self.format_processing_time(entry_processing_time):<10} {eta:<10}")
                print(f"🔕 Skipped LM Studio (rule matched: dot-only or ≤3 words) → {entry['seconds']}: Silence")
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_sfx_entry(entry_key, 'Silence')
                
                if i < len(entries) - 1:
                    time.sleep(1)
                continue
            
            try:
                # Call LM Studio API
                response = self.call_lm_studio_api(prompt)
                
                # Parse SFX response
                sound_description = self.parse_sfx_response(response)
                
                # Create output entry with original duration
                sfx_entry = {
                    'seconds': entry['seconds'],
                    'sound_or_silence_description': sound_description
                }
                
                # Add to all entries
                all_sfx_entries.append(sfx_entry)
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_sfx_entry(entry_key, sound_description)
                
                entry_processing_time = time.time() - entry_start_time
                self.processing_times.append(entry_processing_time)
                eta = self.estimate_remaining_time(i+1, len(entries), entry_processing_time, entry['description'])
                print(f"{i+1:<6} {entry['seconds']:<10.3f} {entry['description'][:50]:<50} {'COMPLETED':<15} {self.format_processing_time(entry_processing_time):<10} {eta:<10}")
                
                # Live preview for this entry
                print(f"🎵 Output: {entry['seconds']}: {sound_description}")
                
            except Exception as e:
                entry_processing_time = time.time() - entry_start_time
                self.processing_times.append(entry_processing_time)
                eta = self.estimate_remaining_time(i+1, len(entries), entry_processing_time, entry['description'])
                print(f"{i+1:<6} {entry['seconds']:<10.3f} {entry['description'][:50]:<50} {'ERROR':<15} {self.format_processing_time(entry_processing_time):<10} {eta:<10}")
                print(f"❌ Error processing entry {i+1}: {str(e)}")
                # Continue with next entry instead of failing completely
                all_sfx_entries.append({
                    'seconds': entry['seconds'],
                    'sound_or_silence_description': 'Silence'
                })
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_sfx_entry(entry_key, 'Silence')
            
            # Small delay between API calls
            if i < len(entries) - 1:
                time.sleep(1)
        
        # Save all SFX entries to file
        try:
            self.save_sfx_to_file(all_sfx_entries)
            print(f"\n🎉 Timeline SFX generation completed successfully!")
            print(f"📄 Output saved to: {self.output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving SFX file: {str(e)}")
            return False

def main():
    """Main function"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate SFX for timeline entries")
    parser.add_argument("timeline_file", nargs="?", default="../input/2.timeline.txt",
                       help="Path to timeline file (default: ../input/2.timeline.txt)")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")
    parser.add_argument("--enable-thinking", action="store_true",
                       help="Enable thinking in LM Studio responses (default: disabled)")
    args = parser.parse_args()
    
    # Update ENABLE_THINKING based on CLI argument
    if args.enable_thinking:
        ENABLE_THINKING = True
        print("🧠 Thinking enabled in LM Studio responses")
    else:
        print("🚫 Thinking disabled in LM Studio responses (using /no_think)")
    
    # Check if timeline file exists
    if not os.path.exists(args.timeline_file):
        print(f"❌ Timeline file '{args.timeline_file}' not found")
        print("Usage: python 5.timeline.py [timeline_file] [--force-start] [--enable-thinking]")
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
    generator = TimelineSFXGenerator()
    
    start_time = time.time()
    success = generator.process_timeline(args.timeline_file, resumable_state)
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
