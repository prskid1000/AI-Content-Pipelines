import json
import requests
import time
import os
import argparse
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import builtins as _builtins
from pathlib import Path
import shutil
print = partial(_builtins.print, flush=True)

# Feature flags
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them

# Resumable state management
class ResumableState:
    """Manages resumable state for expensive audio generation operations."""
    
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
            "audio_files": {"completed": [], "results": {}},
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
    
    def is_audio_file_complete(self, file_key: str) -> bool:
        """Check if specific audio file generation is complete."""
        return file_key in self.state["audio_files"]["completed"]
    
    def get_audio_file_info(self, file_key: str) -> dict | None:
        """Get cached audio file info if available."""
        return self.state["audio_files"]["results"].get(file_key)
    
    def set_audio_file_info(self, file_key: str, file_info: dict):
        """Set audio file info and mark as complete."""
        if file_key not in self.state["audio_files"]["completed"]:
            self.state["audio_files"]["completed"].append(file_key)
        self.state["audio_files"]["results"][file_key] = file_info
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
        """Validate that all completed audio files actually exist and clean up missing entries.
        
        Returns:
            int: Number of entries cleaned up (removed from completed list)
        """
        cleaned_count = 0
        audio_files_to_remove = []
        
        # Check each completed audio file
        for file_key in self.state["audio_files"]["completed"]:
            result = self.state["audio_files"]["results"].get(file_key, {})
            file_path = result.get('file', '')
            output_file_path = result.get('output_file', '')
            
            # Check if both the main file and output file actually exist
            main_exists = file_path and os.path.exists(file_path)
            output_exists = output_file_path and os.path.exists(output_file_path)
            
            if not main_exists or not output_exists:
                print(f"Precheck: File missing for {file_key} - marking as not completed")
                print(f"  Main file exists: {main_exists} ({file_path})")
                print(f"  Output file exists: {output_exists} ({output_file_path})")
                audio_files_to_remove.append(file_key)
                cleaned_count += 1
        
        # Remove invalid entries
        for file_key in audio_files_to_remove:
            if file_key in self.state["audio_files"]["completed"]:
                self.state["audio_files"]["completed"].remove(file_key)
            if file_key in self.state["audio_files"]["results"]:
                del self.state["audio_files"]["results"][file_key]
        
        # Save cleaned state if any changes were made
        if cleaned_count > 0:
            self._save_state()
            print(f"Precheck: Cleaned up {cleaned_count} invalid entries from checkpoint")
        
        return cleaned_count
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        audio_done = len(self.state["audio_files"]["completed"])
        audio_total = len(self.state["audio_files"]["results"]) + len([k for k in self.state["audio_files"]["results"].keys() if k not in self.state["audio_files"]["completed"]])
        
        return f"Progress: Audio Files({audio_done}/{audio_total})"

class DirectTimelineProcessor:
    def __init__(self, comfyui_url="http://127.0.0.1:8188/", max_workers=3):
        self.comfyui_url = comfyui_url
        self.output_folder = "../../ComfyUI/output/audio/sfx"
        self.final_output_folder = "../output/sfx"
        self.max_workers = max_workers
        
        # Time estimation tracking
        self.processing_times = []
        self.start_time = None
        
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.final_output_folder, exist_ok=True)
        self.clear_output_folder()
        
    def parse_timeline(self, timeline_text):
        """Parse timeline and combine silence entries"""
        timeline_entries = []
        for line in timeline_text.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    seconds = float(parts[0].strip())
                    description = parts[1].strip()
                    timeline_entries.append({'seconds': seconds, 'description': description})
        
        return self.combine_consecutive_silence(timeline_entries)
    
    def combine_consecutive_silence(self, timeline_entries):
        """Combine consecutive silence entries"""
        if not timeline_entries:
            return timeline_entries
        
        combined_entries = []
        current_silence_duration = 0
        
        for i, entry in enumerate(timeline_entries):
            if self.is_silence_entry(entry['description']):
                current_silence_duration += entry['seconds']
            else:
                if current_silence_duration > 0:
                    combined_entries.append({
                        'seconds': round(current_silence_duration, 5),
                        'description': f"Silence"
                    })
                    current_silence_duration = 0
                combined_entries.append(entry)
        
        if current_silence_duration > 0:
            combined_entries.append({
                'seconds': round(current_silence_duration, 5),
                'description': f"Silence"
            })
        
        print(f"🔇 Combined {len(timeline_entries)} entries to {len(combined_entries)} entries")
        return combined_entries
    
    def parse_timeline_preserve_order(self, timeline_text):
        """Parse timeline and preserve exact order without combining silences"""
        timeline_entries = []
        for line in timeline_text.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    seconds = round(float(parts[0].strip()), 5)
                    description = parts[1].strip()
                    timeline_entries.append({'seconds': seconds, 'description': description})
        
        print(f"📋 Parsed {len(timeline_entries)} entries preserving original order")
        return timeline_entries
    
    def save_combined_timeline(self, timeline_entries, filename="../input/4.sfx.txt"):
        """Save combined timeline back to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            for entry in timeline_entries:
                f.write(f"{round(entry['seconds'], 5)}: {entry['description']}\n")
        print(f"💾 Saved {len(timeline_entries)} entries to {filename}")
    
    def clear_output_folder(self):
        """Clear output folder"""
        if os.path.exists(self.output_folder):
            for file in os.listdir(self.output_folder):
                os.remove(os.path.join(self.output_folder, file))

        out = "../output/sfx.wav"
        if os.path.exists(out):
            os.remove(out)
            print(f"Removed existing final output: {out}")
    
    def clear_silence_files(self):
        """Clear silence files from output folder"""
        if os.path.exists(self.output_folder):
            for file in os.listdir(self.output_folder):
                if file.startswith("sfx_") and file.endswith(".flac"):
                    try:
                        os.remove(os.path.join(self.output_folder, file))
                    except Exception as e:
                        print(f"Warning: Could not remove {file}: {e}")
    
    def clear_all_sfx_files(self):
        """Clear all SFX-related files"""
        print("🧹 Clearing all SFX files...")
        
        # Clear output folder
        if os.path.exists(self.output_folder):
            for file in os.listdir(self.output_folder):
                try:
                    os.remove(os.path.join(self.output_folder, file))
                    print(f"Removed: {file}")
                except Exception as e:
                    print(f"Warning: Could not remove {file}: {e}")
        
        # Clear final output folder
        if os.path.exists(self.final_output_folder):
            for file in os.listdir(self.final_output_folder):
                try:
                    os.remove(os.path.join(self.final_output_folder, file))
                    print(f"Removed from output: {file}")
                except Exception as e:
                    print(f"Warning: Could not remove from output {file}: {e}")
                
        print("✅ All SFX files cleared!")
    
    def copy_to_output_folder(self, source_file, filename):
        """Copy generated SFX file to output/sfx folder"""
        try:
            if not os.path.exists(source_file):
                print(f"Warning: Source file does not exist: {source_file}")
                return None
            
            # Create destination path in output/sfx folder
            dest_path = os.path.join(self.final_output_folder, filename)
            
            # Copy the file
            shutil.copy2(source_file, dest_path)
            print(f"📁 Copied to output: {filename}")
            return dest_path
            
        except Exception as e:
            print(f"Error copying {source_file} to output folder: {e}")
            return None
    
    def estimate_remaining_time(self, current_file: int, total_files: int, file_processing_time: float = None, audio_duration: float = None, is_silence: bool = None, description: str = None) -> str:
        """Estimate remaining time based on processing history and content characteristics"""
        if not self.processing_times:
            return "No data available"
        
        # Calculate base average processing time per file using ALL previous entries
        avg_time_per_file = sum(self.processing_times) / len(self.processing_times)
        
        # If we have current file processing time, include it in the calculation
        if file_processing_time:
            # Use all previous entries plus current entry for more accurate estimation
            all_times = self.processing_times + [file_processing_time]
            estimated_time_per_file = sum(all_times) / len(all_times)
        else:
            estimated_time_per_file = avg_time_per_file
        
        # Apply content-based adjustments if we have audio characteristics
        if audio_duration is not None and is_silence is not None:
            # Duration factor - longer audio generally takes more time to generate
            duration_factor = 1.0 + (audio_duration - 5.0) * 0.1  # Base 5 seconds, +10% per second over/under
            
            # Generation method factor
            method_factor = 0.2 if is_silence else 1.0  # Silence is much faster (20% of normal time)
            
            # Description complexity factor
            complexity_factor = 1.0
            if description:
                word_count = len(description.split())
                char_count = len(description)
                
                # More complex descriptions take longer
                word_factor = 1.0 + (word_count - 3) * 0.1  # Base 3 words, +10% per word over/under
                char_factor = 1.0 + (char_count - 20) * 0.01  # Base 20 chars, +1% per char over/under
                
                # Check for complexity indicators
                if any(word in description.lower() for word in ['complex', 'layered', 'ambient', 'atmospheric', 'multiple']):
                    complexity_factor = 1.5  # 50% longer for complex descriptions
                elif any(word in description.lower() for word in ['simple', 'basic', 'single', 'short']):
                    complexity_factor = 0.8  # 20% shorter for simple descriptions
                
                complexity_factor = min(2.0, max(0.5, (word_factor + char_factor) / 2 * complexity_factor))
            
            # Combine factors (cap at reasonable bounds)
            total_factor = min(3.0, max(0.1, duration_factor * method_factor * complexity_factor))
            estimated_time_per_file *= total_factor
        
        remaining_files = total_files - current_file
        estimated_remaining_seconds = remaining_files * estimated_time_per_file
        
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
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            # Clean up any temporary files if needed
            pass
        except:
            pass
    
    def load_sfx_workflow(self):
        """Load SFX workflow from JSON"""
        wf = '../workflow/sfx.json'
        with open(wf, 'r') as f:
            return json.load(f)
    
    def find_node_by_type(self, workflow, node_type):
        """Find node by type"""
        for node_id, node in workflow.items():
            if node.get('class_type') == node_type:
                return node
        return None
    
    def update_workflow(self, workflow, text_prompt, duration, filename):
        """Update workflow parameters"""
        # Update text
        node = self.find_node_by_type(workflow, 'CLIPTextEncode')
        if node:
            node['inputs']['text'] = text_prompt
        
        # Update duration
        node = self.find_node_by_type(workflow, 'EmptyLatentAudio')
        if node:
            node['inputs']['seconds'] = duration
        
        # Update filename
        node = self.find_node_by_type(workflow, 'SaveAudio')
        if node:
            node['inputs']['filename_prefix'] = f"audio/sfx/{filename}"
        
        return workflow
    
    def is_silence_entry(self, description):
        """Check if an entry is a silence entry"""
        description_lower = description.lower().strip()
        return (description_lower == 'silence' or 
                description_lower.startswith('silence') or 
                description_lower.endswith('silence') or 
                'silence' in description_lower.split())
    
    def generate_silence_audio(self, duration, filename):
        """Generate silence audio directly using pydub"""
        try:
            # Create silence audio segment with standard audio format
            # 44.1kHz, 16-bit, mono to match typical ComfyUI output
            silence = AudioSegment.silent(duration=int(duration * 1000))  # pydub uses milliseconds
            silence = silence.set_frame_rate(44100).set_channels(1)
            
            # Save to the same output folder as ComfyUI files in FLAC format
            silence_path = os.path.join(self.output_folder, f"{filename}.flac")
            silence.export(silence_path, format="flac", parameters=["-ac", "1", "-ar", "44100"])
            
            print(f"🔇 Generated silence: {duration}s")
            return silence_path
            
        except Exception as e:
            print(f"Error generating silence for {duration}s: {e}")
            return None
    
    def generate_single_sfx(self, entry_data, resumable_state=None):
        """Generate single SFX audio file"""
        i, entry = entry_data
        duration = round(entry['seconds'], 5)
        if duration <= 0:
            return None
        
        file_start_time = time.time()
        
        # Create filename with order information for proper merging
        entry_type = "silence" if self.is_silence_entry(entry['description']) else "sfx"
        filename = f"sfx_{i:03d}_{entry_type}_{round(entry['seconds'], 5)}"
        
        # Create unique key for this entry
        file_key = f"{i}_{entry['description']}_{duration}"
        
        # Check if resumable and already complete
        if resumable_state and resumable_state.is_audio_file_complete(file_key):
            cached_info = resumable_state.get_audio_file_info(file_key)
            if cached_info and os.path.exists(cached_info.get('file', '')):
                print(f"Using cached audio file: {entry['description']} ({duration}s)")
                return cached_info
            elif cached_info:
                print(f"Cached file missing, regenerating: {entry['description']} ({duration}s)")
        
        # Check if this is a silence entry
        if self.is_silence_entry(entry['description']):
            silence_path = self.generate_silence_audio(duration, filename)
            if silence_path:
                # Copy to output folder
                output_filename = f"{filename}.flac"
                copied_path = self.copy_to_output_folder(silence_path, output_filename)
                
                file_processing_time = time.time() - file_start_time
                self.processing_times.append(file_processing_time)
                
                result = {
                    'file': silence_path,
                    'output_file': copied_path,  # Track the copied file location
                    'order_index': i,  # Keep track of original order
                    'duration': duration,
                    'description': entry['description']
                }
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_audio_file_info(file_key, result)
                
                return result
            return None
        
        # For non-silence entries, use ComfyUI
        try:
            print(f"Generating: {entry['description']} ({duration}s)")
            
            workflow = self.load_sfx_workflow()
            workflow = self.update_workflow(workflow, entry['description'], duration, filename)
            
            response = requests.post(f"{self.comfyui_url}prompt", json={"prompt": workflow}, timeout=30)
            if response.status_code != 200:
                raise Exception(f"Failed to send workflow: {response.text}")
            
            prompt_id = response.json()["prompt_id"]
            
            # Wait for completion
            while True:
                history_response = requests.get(f"{self.comfyui_url}history/{prompt_id}")
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    if prompt_id in history_data:
                        status = history_data[prompt_id].get('status', {})
                        if status.get('exec_info', {}).get('queue_remaining', 0) == 0:
                            outputs = history_data[prompt_id].get('outputs', {})
                            for node_id, node_output in outputs.items():
                                if 'audio' in node_output:
                                    time.sleep(3)
                                    files_in_folder = os.listdir(self.output_folder)
                                    matching_files = [f for f in files_in_folder if f.startswith(filename) and f.endswith('.flac')]
                                    
                                    if matching_files:
                                        matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_folder, x)), reverse=True)
                                        found_path = os.path.join(self.output_folder, matching_files[0])
                                        
                                        # Copy to output folder
                                        output_filename = matching_files[0]  # Use the actual generated filename
                                        copied_path = self.copy_to_output_folder(found_path, output_filename)
                                        
                                        file_processing_time = time.time() - file_start_time
                                        self.processing_times.append(file_processing_time)
                                        
                                        result = {
                                            'file': found_path,
                                            'output_file': copied_path,  # Track the copied file location
                                            'order_index': i,  # Keep track of original order
                                            'duration': duration,
                                            'description': entry['description']
                                        }
                                        
                                        # Save to checkpoint if resumable mode enabled
                                        if resumable_state:
                                            resumable_state.set_audio_file_info(file_key, result)
                                        
                                        return result
                            break
                time.sleep(5)
            
            raise Exception(f"Failed to generate audio for: {entry['description']}")
                
        except Exception as e:
            file_processing_time = time.time() - file_start_time
            self.processing_times.append(file_processing_time)
            print(f"Error generating audio for '{entry['description']}': {e}")
            return None
    
    def generate_all_sfx_batch(self, timeline_entries, resumable_state=None):
        """Generate all SFX audio files using batch processing"""
        generated_files = []
        batch_data = [(i, entry) for i, entry in enumerate(timeline_entries)]
        
        # Count silence vs non-silence entries
        silence_count = sum(1 for _, entry in batch_data if self.is_silence_entry(entry['description']))
        sfx_count = len(batch_data) - silence_count
        
        print(f"📊 Processing {len(batch_data)} entries: {silence_count} silence, {sfx_count} SFX")
        
        # Initialize start time for time estimation
        self.start_time = time.time()
        
        print(f"\n📊 SFX GENERATION PROGRESS")
        print("=" * 100)
        print(f"{'File':<6} {'Type':<10} {'Duration':<10} {'Description':<40} {'Status':<15} {'Time':<10} {'ETA':<10}")
        print("-" * 100)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_entry = {executor.submit(self.generate_single_sfx, data, resumable_state): data for data in batch_data}
            completed_count = 0
            for future in as_completed(future_to_entry):
                result = future.result()
                completed_count += 1
                if result:
                    generated_files.append(result)
                    entry_type = "Silence" if self.is_silence_entry(result['description']) else "SFX"
                    is_silence = self.is_silence_entry(result['description'])
                    eta = self.estimate_remaining_time(completed_count, len(batch_data), audio_duration=result['duration'], is_silence=is_silence, description=result['description'])
                    print(f"{completed_count:<6} {entry_type:<10} {result['duration']:<10.3f} {result['description'][:40]:<40} {'COMPLETED':<15} {'--':<10} {eta:<10}")
                else:
                    entry_data = future_to_entry[future]
                    entry_type = "Silence" if self.is_silence_entry(entry_data[1]['description']) else "SFX"
                    is_silence = self.is_silence_entry(entry_data[1]['description'])
                    eta = self.estimate_remaining_time(completed_count, len(batch_data), audio_duration=entry_data[1]['seconds'], is_silence=is_silence, description=entry_data[1]['description'])
                    print(f"{completed_count:<6} {entry_type:<10} {entry_data[1]['seconds']:<10.3f} {entry_data[1]['description'][:40]:<40} {'FAILED':<15} {'--':<10} {eta:<10}")
        
        return generated_files
    
    def concatenate_audio_files(self, generated_files):
        """Concatenate all generated audio files into final audio"""
        print("🔗 Concatenating audio files in order...")
        # Sort by order_index to maintain the exact order from sfx.txt
        generated_files.sort(key=lambda x: x['order_index'])
        final_audio = AudioSegment.empty()
        
        current_time = 0.0
        
        for file_info in generated_files:
            try:
                # All files are now in FLAC format
                audio_segment = AudioSegment.from_file(file_info['file'], format="flac")
                final_audio = final_audio + audio_segment
                
                print(f"➕ [{file_info['order_index']:2d}] {current_time:6.3f}s - {current_time + file_info['duration']:6.3f}s: {file_info['description']} ({file_info['duration']:.3f}s)")
                current_time += file_info['duration']
                
            except Exception as e:
                print(f"❌ Error loading {file_info['file']}: {e}")
                continue
        
        # Always save as sfx.wav in output folder
        out = "../output/sfx.wav"
        final_audio.export(out, format="wav")
        print(f"🎵 Final audio saved as: {out}")
        print(f"📊 Total duration: {current_time:.3f} seconds ({current_time/60:.2f} minutes)")
        
        return out
    

    
    def calculate_total_duration(self, timeline_entries):
        """Calculate total duration from timeline entries"""
        total_duration = sum(entry['seconds'] for entry in timeline_entries)
        return round(total_duration, 3)
    
    def display_timeline_summary(self, timeline_entries):
        """Display timeline summary with duration breakdown"""
        print("\n" + "="*60)
        print("📋 TIMELINE SUMMARY")
        print("="*60)
        
        total_duration = self.calculate_total_duration(timeline_entries)
        silence_count = sum(1 for entry in timeline_entries if self.is_silence_entry(entry['description']))
        sfx_count = len(timeline_entries) - silence_count
        
        print(f"📊 Total Entries: {len(timeline_entries)}")
        print(f"🎵 SFX Segments: {sfx_count}")
        print(f"🔇 Silence Segments: {silence_count}")
        print(f"⏱️  Total Duration: {total_duration:.3f} seconds ({total_duration/60:.2f} minutes)")
        
        print("\n📝 Timeline Breakdown:")
        print("-" * 60)
        current_time = 0.0
        for i, entry in enumerate(timeline_entries):
            entry_type = "🔇 SILENCE" if self.is_silence_entry(entry['description']) else "🎵 SFX"
            print(f"[{i+1:2d}] {current_time:6.3f}s - {current_time + entry['seconds']:6.3f}s | {entry_type} | {entry['description']}")
            current_time += entry['seconds']
        
        print("="*60)
        return total_duration
    
    def get_user_confirmation(self, total_duration):
        """Get user confirmation to proceed with processing"""
        print(f"\n⚠️  This will generate audio with a total duration of {total_duration:.3f} seconds ({total_duration/60:.2f} minutes)")
        print("Do you want to proceed? (y/n): ", end="")
        
        while True:
            try:
                auto = (AUTO_SFX_CONFIRM or "").strip().lower()
                if auto in ["y", "yes", "n", "no"]:
                    print(f"[AUTO] Using --auto-confirm='{auto}'")
                    response = auto
                else:
                    response = input().strip().lower()
                if response in ['y', 'yes', '']:
                    return True
                elif response in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no: ", end="")
            except KeyboardInterrupt:
                print("\n❌ Processing cancelled by user")
                return False
    
    def process_timeline(self, timeline_text, resumable_state=None):
        """Main processing function"""
        if not timeline_text or not timeline_text.strip():
            raise Exception("Empty or invalid timeline text provided")
        
        # Run precheck to validate file existence and clean up invalid entries
        if resumable_state:
            cleaned_count = resumable_state.validate_and_cleanup_results()
            if cleaned_count > 0:
                print(f"Precheck completed: {cleaned_count} invalid entries removed from checkpoint")
        
        print("🔄 Step 1: Merging consecutive silences and updating sfx.txt...")
        
        # First, combine consecutive silences and save to file
        timeline_entries = self.parse_timeline(timeline_text)
        if not timeline_entries:
            raise Exception("No valid timeline entries found")
        
        self.save_combined_timeline(timeline_entries)
        
        print("📋 Step 2: Reloading updated timeline for processing...")
        # Reload the updated file
        updated_timeline = read_timeline_from_file("../input/4.sfx.txt")
        if updated_timeline is None:
            raise Exception("Failed to reload updated timeline file")
        
        # Parse the updated timeline
        updated_entries = self.parse_timeline_preserve_order(updated_timeline)
        if not updated_entries:
            raise Exception("No entries found in updated timeline")
        
        # Display timeline summary and get user confirmation
        total_duration = self.display_timeline_summary(updated_entries)
        
        if not self.get_user_confirmation(total_duration):
            print("❌ Processing cancelled by user")
            return None
        
        print(f"\n📊 Processing {len(updated_entries)} entries from updated timeline")
        
        print("🎵 Step 3: Generating audio files...")
        # Generate all SFX files using batch processing
        generated_files = self.generate_all_sfx_batch(updated_entries, resumable_state)
        if not generated_files:
            raise Exception("No audio files were generated successfully")
        
        print(f"✅ Generated {len(generated_files)} audio files")
        
        print("🔗 Step 4: Combining files in order...")
        # Concatenate into final audio
        final_audio = self.concatenate_audio_files(generated_files)
        print("🎉 Processing complete!")
        return final_audio

def read_timeline_from_file(filename="../input/4.sfx.txt"):
    """Read timeline data from a text file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Timeline file '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading timeline file: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description="Generate SFX audio files from timeline")
    parser.add_argument("--clear", action="store_true",
                       help="Clear all SFX files and exit")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")
    parser.add_argument("--auto-confirm", default="y", choices=["y", "yes", "n", "no"],
                       help="Auto-confirm processing (default: y)")
    args = parser.parse_args()
    
    # Set global auto-confirm variable
    AUTO_SFX_CONFIRM = args.auto_confirm

    # Check if user wants to clear files
    if args.clear:
        processor = DirectTimelineProcessor(max_workers=3)
        processor.clear_all_sfx_files()
        exit(0)
    
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
    
    timeline_text = read_timeline_from_file()
    if timeline_text is None:
        print("Exiting due to timeline file error.")
        exit(1)
    
    print("🚀 Starting SFX generation with optimized silence handling...")
    processor = DirectTimelineProcessor(max_workers=3)
    
    try:
        final_audio = processor.process_timeline(timeline_text, resumable_state)
        if final_audio:
            print(f"✅ Final audio file: {final_audio}")
            print(f"⏱️  Total execution time: {time.time() - start_time:.3f} seconds")
            
            # Clean up checkpoint files if resumable mode was used and everything completed successfully
            if resumable_state:
                print("All operations completed successfully")
                print("Final progress:", resumable_state.get_progress_summary())
                resumable_state.cleanup()
        else:
            print("❌ Processing was cancelled by user")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("💡 Make sure ComfyUI is running at http://127.0.0.1:8188/ for SFX generation")
        print("🔇 Silence segments will still be generated locally")
