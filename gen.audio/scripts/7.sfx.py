import json
import requests
import time
import os
import argparse
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

class DirectTimelineProcessor:
    def __init__(self, comfyui_url="http://127.0.0.1:8188/", max_workers=3):
        self.comfyui_url = comfyui_url
        self.output_folder = "../../ComfyUI/output/audio/sfx"
        self.max_workers = max_workers
        os.makedirs(self.output_folder, exist_ok=True)
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
                
        print("✅ All SFX files cleared!")
    
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
    
    def generate_single_sfx(self, entry_data):
        """Generate single SFX audio file"""
        i, entry = entry_data
        duration = round(entry['seconds'], 5)
        if duration <= 0:
            return None
        
        # Create filename with order information for proper merging
        entry_type = "silence" if self.is_silence_entry(entry['description']) else "sfx"
        filename = f"sfx_{i:03d}_{entry_type}_{round(entry['seconds'], 5)}"
        
        # Check if this is a silence entry
        if self.is_silence_entry(entry['description']):
            silence_path = self.generate_silence_audio(duration, filename)
            if silence_path:
                return {
                    'file': silence_path,
                    'order_index': i,  # Keep track of original order
                    'duration': duration,
                    'description': entry['description']
                }
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
                                        return {
                                            'file': found_path,
                                            'order_index': i,  # Keep track of original order
                                            'duration': duration,
                                            'description': entry['description']
                                        }
                            break
                time.sleep(5)
            
            raise Exception(f"Failed to generate audio for: {entry['description']}")
                
        except Exception as e:
            print(f"Error generating audio for '{entry['description']}': {e}")
            return None
    
    def generate_all_sfx_batch(self, timeline_entries):
        """Generate all SFX audio files using batch processing"""
        generated_files = []
        batch_data = [(i, entry) for i, entry in enumerate(timeline_entries)]
        
        # Count silence vs non-silence entries
        silence_count = sum(1 for _, entry in batch_data if self.is_silence_entry(entry['description']))
        sfx_count = len(batch_data) - silence_count
        
        print(f"📊 Processing {len(batch_data)} entries: {silence_count} silence, {sfx_count} SFX")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_entry = {executor.submit(self.generate_single_sfx, data): data for data in batch_data}
            for future in as_completed(future_to_entry):
                result = future.result()
                if result:
                    generated_files.append(result)
        
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
    
    def process_timeline(self, timeline_text):
        """Main processing function"""
        if not timeline_text or not timeline_text.strip():
            raise Exception("Empty or invalid timeline text provided")
        
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
        generated_files = self.generate_all_sfx_batch(updated_entries)
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
    # Hardcoded default for non-interactive auto-confirm
    AUTO_SFX_CONFIRM = "y"

    # Check if user wants to clear files
    if len(sys.argv) > 1 and sys.argv[1] == "--clear":
        processor = DirectTimelineProcessor(max_workers=3)
        processor.clear_all_sfx_files()
        exit(0)
    
    start_time = time.time()
    
    timeline_text = read_timeline_from_file()
    if timeline_text is None:
        print("Exiting due to timeline file error.")
        exit(1)
    
    print("🚀 Starting SFX generation with optimized silence handling...")
    processor = DirectTimelineProcessor(max_workers=3)
    
    try:
        final_audio = processor.process_timeline(timeline_text)
        if final_audio:
            print(f"✅ Final audio file: {final_audio}")
            print(f"⏱️  Total execution time: {time.time() - start_time:.3f} seconds")
        else:
            print("❌ Processing was cancelled by user")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("💡 Make sure ComfyUI is running at http://127.0.0.1:8188/ for SFX generation")
        print("🔇 Silence segments will still be generated locally")
