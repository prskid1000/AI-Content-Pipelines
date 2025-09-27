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
MODEL_TIMING_GENERATION = "qwen/qwen3-14b"  # Model for timing SFX generation

# Feature flags
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode

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
            "timing_entries": {"completed": [], "results": {}},
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
    
    def is_timing_entry_complete(self, entry_key: str) -> bool:
        """Check if specific timing entry is complete."""
        return entry_key in self.state["timing_entries"]["completed"]
    
    def get_timing_entry(self, entry_key: str) -> Dict[str, Any] | None:
        """Get cached timing entry if available."""
        return self.state["timing_entries"]["results"].get(entry_key)
    
    def set_timing_entry(self, entry_key: str, timing_info: Dict[str, Any]):
        """Set timing entry and mark as complete."""
        if entry_key not in self.state["timing_entries"]["completed"]:
            self.state["timing_entries"]["completed"].append(entry_key)
        self.state["timing_entries"]["results"][entry_key] = timing_info
        self._save_state()
    
    def cleanup(self):
        """Clean up checkpoint files after successful completion."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
                print("Cleaned up checkpoint files")
        except Exception as ex:
            print(f"WARNING: Failed to cleanup checkpoint files: {ex}")
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        timing_done = len(self.state["timing_entries"]["completed"])
        timing_total = len(self.state["timing_entries"]["results"]) + len([k for k in self.state["timing_entries"]["results"].keys() if k not in self.state["timing_entries"]["completed"]])
        
        return f"Progress: Timing Entries({timing_done}/{timing_total})"

class TimingSFXGenerator:
    def __init__(self, lm_studio_url="http://localhost:1234/v1", model=MODEL_TIMING_GENERATION, use_json_schema=True):
        self.lm_studio_url = lm_studio_url
        self.output_file = "../input/4.sfx.txt"
        self.model = model
        self.use_json_schema = use_json_schema
        self.timeline_file = "../input/2.timeline.txt"
        
    def read_timing_content(self, filename="../input/3.timing.txt") -> str:
        """Read timing content from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Timing file '{filename}' not found.")
            return None
        except Exception as e:
            print(f"Error reading timing file: {e}")
            return None
    
    def read_timeline_content(self, filename="../input/2.timeline.txt") -> str:
        """Read timeline content from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error reading timeline file: {e}")
            return None
    
    def parse_timing_entries(self, content: str) -> List[Dict[str, Any]]:
        """Parse timing content into structured entries"""
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
        
        print(f"📋 Parsed {len(entries)} timing entries")
        return entries
    
    def parse_timeline_entries(self, content: str) -> List[Dict[str, Any]]:
        """Parse timeline content into structured entries (same format as timing)"""
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
    
    def create_prompt_for_sound_duration(self, entry: Dict[str, Any], transcript_context: str = "") -> str:
        """Create the prompt for estimating realistic sound duration and position"""
        # Count words in transcript
        word_count = len(transcript_context.split())
        
        prompt = f"""Transcript: {transcript_context}

SFX: {entry['description']}

Duration: {entry['seconds']} seconds
Word count: {word_count} words

Consider:
- Realistic physics timing for this sound
- Proportion of words that need this sound effect
- Don't over-extend sounds for very long sentences
- Match sound duration to relevant action/description portion"""
        return prompt

    def _build_response_format(self) -> Dict[str, Any]:
        """Build JSON Schema response format for sound duration and position estimation."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "sound_timing",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "realistic_duration_seconds": {"type": "number"},
                        "position_float": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    },
                    "required": ["realistic_duration_seconds", "position_float"]
                },
                "strict": True
            }
        }
    
    def call_lm_studio_api(self, prompt: str) -> str:
        """Call LM Studio API to estimate realistic sound duration"""
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
 """You are an audio timing expert. Estimate realistic sound effect duration and optimal placement within a transcript line.

TASK: Given a sound effect description and transcript context, estimate:
1. Realistic duration in seconds (consider physics and human experience)
2. Optimal position (0.0=start, 0.5=middle, 1.0=end of transcript line)

IMPORTANT RULES:
- Sound duration should match the relevant portion of the transcript, not the entire line
- For long sentences, don't extend sounds unnecessarily (e.g., waterfall shouldn't play for entire 20-word sentence)
- Consider word count and context - match sound to action/description portion
- Be realistic about physics (footsteps = 1-2s, door knock = 0.5s, etc.)

OUTPUT: JSON with realistic_duration_seconds and position_float fields."""
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n/no_think"
                    }
                ],
                "temperature": 1,
                "max_tokens": 128,
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
    
    def parse_timing_response(self, response: str) -> Dict[str, Any]:
        """Parse the timing response from LM Studio"""
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
            if isinstance(json_obj, dict) and "realistic_duration_seconds" in json_obj:
                duration = json_obj["realistic_duration_seconds"]
                position_float = json_obj.get("position_float", 0.5)
                
                if isinstance(duration, (int, float)) and duration > 0:
                    return {
                        "duration": float(duration),
                        "position": float(position_float)
                    }
        except Exception:
            pass
        
        # Fallback: try to extract numbers from response
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            try:
                return {
                    "duration": float(numbers[0]),
                    "position": 0.5
                }
            except ValueError:
                pass
        
        # Default fallback
        return None
    
    def split_entry_into_sound_and_silence(self, entry: Dict[str, Any], timing_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a timing entry into silence + sound + silence based on position"""
        original_duration = entry['seconds']
        realistic_duration = timing_info['duration']
        position_float = timing_info['position']
        
        # Cap realistic duration at original duration
        sound_duration = min(realistic_duration, original_duration)
        
        # Calculate silence + sound + silence based on float position (0.0 to 1.0)
        before_sound = max(0, (original_duration - sound_duration) * position_float)
        after_sound = max(0, original_duration - sound_duration - before_sound)
        
        result = []
        
        # Add silence before sound (if any)
        if before_sound > 0:
            result.append({
                'seconds': before_sound,
                'description': 'Silence'
            })
        
        # Add sound entry
        if sound_duration > 0:
            result.append({
                'seconds': sound_duration,
                'description': entry['description']
            })
        
        # Add silence after sound (if any)
        if after_sound > 0:
            result.append({
                'seconds': after_sound,
                'description': 'Silence'
            })
        
        return result
    
    def post_process_entries(self, entries: List[Dict[str, Any]], original_entries: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Post-process entries: merge consecutive silence and adjust short durations"""
        if not entries:
            return entries
        
        print("🔧 Post-processing entries...")
        
        # Step 1: Merge consecutive silence entries
        merged_entries = []
        current_silence = None
        
        for entry in entries:
            if entry['description'] == 'Silence':
                if current_silence is None:
                    current_silence = entry.copy()
                else:
                    # Merge with existing silence
                    current_silence['seconds'] += entry['seconds']
            else:
                # Add accumulated silence if exists
                if current_silence is not None:
                    merged_entries.append(current_silence)
                    current_silence = None
                merged_entries.append(entry)
        
        # Add final silence if exists
        if current_silence is not None:
            merged_entries.append(current_silence)
        
        print(f"📊 Merged silence: {len(entries)} → {len(merged_entries)} entries")
        
        # Step 2: Convert short sounds to silence, then adjust short silence
        final_entries = []
        for i, entry in enumerate(merged_entries):
            # Convert short sounds to silence first - but only if we can't borrow from silence entries
            if entry['seconds'] < 1.0 and entry['description'] != 'Silence':
                # Check if original entry (before split) was greater than 1 second
                original_was_long = True
                if original_entries and i < len(original_entries):
                    original_was_long = original_entries[i]['seconds'] > 1.0
                
                # Check if we can borrow from adjacent silence entries
                # Both previous AND next must be silence for equal borrowing
                can_borrow_from_silence = False
                borrowed_duration = 0
                prev_available = 0
                next_available = 0
                
                # Check previous entry if it's silence
                if i > 0 and merged_entries[i-1]['description'] == 'Silence':
                    prev_available = merged_entries[i-1]['seconds'] - 1.0  # Keep 1s minimum
                
                # Check next entry if it's silence
                if i < len(merged_entries) - 1 and merged_entries[i+1]['description'] == 'Silence':
                    next_available = merged_entries[i+1]['seconds'] - 1.0  # Keep 1s minimum
                
                # Both must be silence and have available time
                if prev_available > 0 and next_available > 0:
                    can_borrow_from_silence = True
                    borrowed_duration = prev_available + next_available
                
                # Only convert to silence if:
                # 1. Original entry was short (≤1s), OR
                # 2. We can't borrow enough from silence entries
                should_convert_to_silence = not original_was_long or (not can_borrow_from_silence or borrowed_duration < (1.0 - entry['seconds']))
                
                if should_convert_to_silence:
                    reason = "original was short" if not original_was_long else "can't borrow enough from silence"
                    print(f"⚠️  Short sound converted to silence: {entry['description']} ({entry['seconds']:.3f}s) - {reason}")
                    entry['description'] = 'Silence'
                else:
                    # Borrow equally from both silence entries to extend this sound
                    needed = 1.0 - entry['seconds']
                    print(f"🔧 Extending short sound by borrowing equally from both silence entries: {entry['description']} ({entry['seconds']:.3f}s → 1.0s)")
                    
                    # Calculate equal borrowing from both sides
                    borrow_per_side = needed / 2.0
                    
                    # Borrow from previous silence
                    if i > 0 and merged_entries[i-1]['description'] == 'Silence':
                        prev_borrow = min(prev_available, borrow_per_side)
                        if prev_borrow > 0:
                            merged_entries[i-1]['seconds'] -= prev_borrow
                            entry['seconds'] += prev_borrow
                            print(f"   📉 Borrowed {prev_borrow:.3f}s from previous silence")
                    
                    # Borrow from next silence
                    if i < len(merged_entries) - 1 and merged_entries[i+1]['description'] == 'Silence':
                        next_borrow = min(next_available, borrow_per_side)
                        if next_borrow > 0:
                            merged_entries[i+1]['seconds'] -= next_borrow
                            entry['seconds'] += next_borrow
                            print(f"   📉 Borrowed {next_borrow:.3f}s from next silence")
            
            # Now adjust short silence (can borrow from any previous/next entry)
            if entry['seconds'] < 1.0 and entry['description'] == 'Silence':
                print(f"⚠️  Short silence detected: {entry['seconds']:.3f}s")
                
                # Calculate how much to borrow
                target_duration = 1.0
                needed_duration = target_duration - entry['seconds']
                
                # Try to borrow from previous entry (any type) - but don't make it < 1s
                prev_available = 0
                if i > 0 and merged_entries[i-1]['seconds'] > 1.0:
                    prev_available = min(merged_entries[i-1]['seconds'] * 0.1, needed_duration / 2)
                    # Ensure we don't make previous entry < 1s
                    if merged_entries[i-1]['seconds'] - prev_available < 1.0:
                        prev_available = max(0, merged_entries[i-1]['seconds'] - 1.0)
                
                # Try to borrow from next entry (any type) - but don't make it < 1s
                next_available = 0
                if i < len(merged_entries) - 1 and merged_entries[i+1]['seconds'] > 1.0:
                    next_available = min(merged_entries[i+1]['seconds'] * 0.1, needed_duration / 2)
                    # Ensure we don't make next entry < 1s
                    if merged_entries[i+1]['seconds'] - next_available < 1.0:
                        next_available = max(0, merged_entries[i+1]['seconds'] - 1.0)
                
                # If we can't borrow enough, distribute this entry among prev/next
                if prev_available + next_available < needed_duration:
                    print(f"   ⚠️  Cannot borrow enough time, distributing short entry")
                    
                    # Distribute current entry's duration among prev/next
                    distribute_amount = entry['seconds'] / 2
                    
                    if i > 0:
                        merged_entries[i-1]['seconds'] += distribute_amount
                        print(f"   📈 Distributed {distribute_amount:.3f}s to previous entry")
                    
                    if i < len(merged_entries) - 1:
                        merged_entries[i+1]['seconds'] += distribute_amount
                        print(f"   📈 Distributed {distribute_amount:.3f}s to next entry")
                    
                    # Mark this entry for removal
                    entry['seconds'] = 0
                    print(f"   🗑️  Entry distributed and marked for removal")
                    
                else:
                    # Apply normal borrowing adjustments
                    if prev_available > 0:
                        merged_entries[i-1]['seconds'] -= prev_available
                        print(f"   📉 Borrowed {prev_available:.3f}s from previous entry")
                    
                    if next_available > 0:
                        merged_entries[i+1]['seconds'] -= next_available
                        print(f"   📉 Borrowed {next_available:.3f}s from next entry")
                    
                    # Adjust current entry
                    entry['seconds'] += prev_available + next_available
                    print(f"   ✅ Adjusted to {entry['seconds']:.3f}s")
            
            final_entries.append(entry)
        
        # Filter out entries with 0 seconds (distributed entries)
        final_entries = [entry for entry in final_entries if entry['seconds'] > 0]
        
        print(f"🔧 Post-processing completed")
        return final_entries
    
    def save_sfx_to_file(self, all_sfx_entries: List[Dict[str, Any]], original_entries: List[Dict[str, Any]] = None) -> None:
        """Save all SFX entries to sfx.txt"""
        try:
            # Post-process entries before saving
            processed_entries = self.post_process_entries(all_sfx_entries, original_entries)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for entry in processed_entries:
                    f.write(f"{entry['seconds']:.3f}: {entry['description']}\n")
            
            total_duration = sum(entry['seconds'] for entry in processed_entries)
            print(f"💾 Saved {len(processed_entries)} processed SFX entries to {self.output_file}")
            print(f"⏱️  Total duration: {total_duration:.3f} seconds ({total_duration/60:.2f} minutes)")
            
        except Exception as e:
            raise Exception(f"Failed to save SFX file: {str(e)}")
    
    def process_timing(self, timing_filename="../input/3.timing.txt", resumable_state: ResumableState | None = None) -> bool:
        """Main processing function - process timing and timeline together"""
        print("🚀 Starting Timing SFX Generation...")
        print(f"📁 Reading timing from: {timing_filename}")
        print(f"📁 Reading timeline from: {self.timeline_file}")
        
        # Read both files
        timing_content = self.read_timing_content(timing_filename)
        timeline_content = self.read_timeline_content(self.timeline_file)
        
        if timing_content is None or timeline_content is None:
            print("❌ Could not read one or both files")
            return False
        
        # Parse both files
        timing_entries = self.parse_timing_entries(timing_content)
        timeline_entries = self.parse_timeline_entries(timeline_content)
        
        if not timing_entries or not timeline_entries:
            print("❌ No valid entries found in one or both files")
            return False
        
        # Check if line counts match
        if len(timing_entries) != len(timeline_entries):
            print(f"⚠️  Warning: Line count mismatch! Timing: {len(timing_entries)}, Timeline: {len(timeline_entries)}")
            print("   Processing only the minimum number of lines")
            min_lines = min(len(timing_entries), len(timeline_entries))
            timing_entries = timing_entries[:min_lines]
            timeline_entries = timeline_entries[:min_lines]
        
        print(f"📋 Processing {len(timing_entries)} line pairs")
        
        # Process each pair together
        all_sfx_entries = []
        original_entries = []  # Track original entries before splitting
        
        for i in range(len(timing_entries)):
            timing_entry = timing_entries[i]
            timeline_entry = timeline_entries[i]
            
            # Create unique key for this entry pair
            entry_key = f"{timing_entry['seconds']}:{timing_entry['description']}:{timeline_entry['description'][:50]}"
            
            # Check if resumable and already complete
            if resumable_state and resumable_state.is_timing_entry_complete(entry_key):
                cached_timing_info = resumable_state.get_timing_entry(entry_key)
                if cached_timing_info:
                    print(f"\n📝 Sound effect {i+1}/{len(timing_entries)}: using cached timing from checkpoint")
                    print(f"   🎬 Transcript: {timeline_entry['description'][:60]}...")
                    print(f"   🎵 SFX: {timing_entry['description']} ({timing_entry['seconds']}s)")
                    print(f"🎯 Original: {timing_entry['seconds']}s, Realistic: {cached_timing_info['duration']:.2f}s, Position: {cached_timing_info['position']:.2f}")
                    
                    # Split entry into silence + sound + silence based on cached position
                    split_entries = self.split_entry_into_sound_and_silence(timing_entry, cached_timing_info)
                    
                    # Add to all entries and track original entry for each split
                    for split_entry in split_entries:
                        all_sfx_entries.append(split_entry)
                        original_entries.append(timing_entry)  # Track original entry for each split
                        print(f"🎵 {split_entry['seconds']:.3f}s - {split_entry['description']}")
                    
                    continue
            
            # Skip silence entries - only process actual sound effects
            if timing_entry['description'].lower().strip() == 'silence':
                print(f"⏭️  Skipping silence entry {i+1}: {timing_entry['seconds']}s")
                all_sfx_entries.append({
                    'seconds': timing_entry['seconds'],
                    'description': 'Silence'
                })
                original_entries.append(timing_entry)  # Track original entry
                continue
            
            entry_start_time = time.time()
            print(f"\n📝 Processing sound effect {i+1}/{len(timing_entries)}:")
            print(f"   🎬 Transcript: {timeline_entry['description'][:60]}...")
            print(f"   🎵 SFX: {timing_entry['description']} ({timing_entry['seconds']}s)")
            
            try:
                # Create prompt with both transcript and SFX context
                prompt = self.create_prompt_for_sound_duration(timing_entry, timeline_entry['description'])
                
                # Call LM Studio API to get realistic duration and position
                response = self.call_lm_studio_api(prompt)
                
                # Parse timing response
                timing_info = self.parse_timing_response(response)
                
                if timing_info is None:
                    print(f"⚠️  Could not parse response for line {i+1}, using default middle position")
                    timing_info = {
                        "duration": timing_entry['seconds'] * 0.3,  # 30% of original
                        "position": 0.5
                    }
                
                print(f"🎯 Original: {timing_entry['seconds']}s, Realistic: {timing_info['duration']:.2f}s, Position: {timing_info['position']:.2f}")
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_timing_entry(entry_key, timing_info)
                
                # Split entry into silence + sound + silence based on position
                split_entries = self.split_entry_into_sound_and_silence(timing_entry, timing_info)
                
                # Add to all entries and track original entry for each split
                for split_entry in split_entries:
                    all_sfx_entries.append(split_entry)
                    original_entries.append(timing_entry)  # Track original entry for each split
                    print(f"🎵 {split_entry['seconds']:.3f}s - {split_entry['description']}")

                entry_end_time = time.time()
                entry_duration = entry_end_time - entry_start_time
                print(f"✅ Sound effect {i+1} processed successfully in {entry_duration:.2f} seconds")
                
            except Exception as e:
                entry_end_time = time.time()
                entry_duration = entry_end_time - entry_start_time
                print(f"❌ Error processing sound effect {i+1}: {str(e)} (took {entry_duration:.2f} seconds)")
                # Continue with next entry instead of failing completely
                all_sfx_entries.append({
                    'seconds': timing_entry['seconds'],
                    'description': timing_entry['description']
                })
                original_entries.append(timing_entry)  # Track original entry
            
            # Small delay between API calls
            if i < len(timing_entries) - 1:
                time.sleep(1)
        
        # Save all SFX entries to file
        try:
            self.save_sfx_to_file(all_sfx_entries, original_entries)
            print(f"\n🎉 Timing SFX generation completed successfully!")
            print(f"📄 Output saved to: {self.output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving SFX file: {str(e)}")
            return False

def main():
    """Main function"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate timing SFX for sound effects")
    parser.add_argument("timing_file", nargs="?", default="../input/3.timing.txt",
                       help="Path to timing file (default: ../input/3.timing.txt)")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")
    args = parser.parse_args()
    
    # Check if timing file exists
    if not os.path.exists(args.timing_file):
        print(f"❌ Timing file '{args.timing_file}' not found")
        print("Usage: python 6.timing.py [timing_file] [--force-start]")
        return 1
    
    # Check if timeline file exists
    if not os.path.exists("../input/2.timeline.txt"):
        print(f"❌ Timeline file 'input/2.timeline.txt' not found")
        print("Both ../input/3.timing.txt and ../input/2.timeline.txt are required")
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
    generator = TimingSFXGenerator()
    
    start_time = time.time()
    success = generator.process_timing(args.timing_file, resumable_state)
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
