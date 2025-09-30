import json
import requests
import time
import os
import shutil
import re
import argparse
from pydub import AudioSegment
from functools import partial
import builtins as _builtins
from pathlib import Path
print = partial(_builtins.print, flush=True)

# Configuration constants
CHUNK_SIZE = 5  # Number of dialogues/lines per chunk
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion

# Resumable state management
class ResumableState:
    """Manages resumable state for story processing."""
    
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
            "chunks": {"completed": [], "results": {}},
            "final_concatenation": {"completed": False, "result": None},
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
    
    def is_chunk_complete(self, chunk_number: int) -> bool:
        """Check if specific chunk is complete."""
        return chunk_number in self.state["chunks"]["completed"]
    
    def get_chunk_result(self, chunk_number: int) -> dict | None:
        """Get cached chunk result if available."""
        return self.state["chunks"]["results"].get(str(chunk_number))
    
    def set_chunk_result(self, chunk_number: int, chunk_data: dict):
        """Set chunk result and mark as complete."""
        if chunk_number not in self.state["chunks"]["completed"]:
            self.state["chunks"]["completed"].append(chunk_number)
        self.state["chunks"]["results"][str(chunk_number)] = chunk_data
        self._save_state()
    
    def is_final_concatenation_complete(self) -> bool:
        """Check if final concatenation is complete."""
        return self.state["final_concatenation"]["completed"]
    
    def get_final_result(self) -> str | None:
        """Get cached final result if available."""
        return self.state["final_concatenation"]["result"]
    
    def set_final_result(self, final_path: str):
        """Set final result and mark as complete."""
        self.state["final_concatenation"]["completed"] = True
        self.state["final_concatenation"]["result"] = final_path
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
        chunks_done = len(self.state["chunks"]["completed"])
        chunks_total = len(self.state["chunks"]["results"]) + len([k for k in self.state["chunks"]["results"].keys() if int(k) not in self.state["chunks"]["completed"]])
        final_done = "✓" if self.is_final_concatenation_complete() else "✗"
        
        return (
            f"Progress: Chunks({chunks_done}/{chunks_total}) Final({final_done})"
        )

class StoryProcessor:
    def __init__(self, comfyui_url="http://127.0.0.1:8188/"):
        self.comfyui_url = comfyui_url
        self.output_folder = "../../ComfyUI/output/audio"
        self.final_output = "../output/story.wav"
        self.chunk_output_dir = "../output/story"
        
        # Create chunk output directory
        os.makedirs(self.chunk_output_dir, exist_ok=True)
        
        # Clear the final output file if it exists
        if os.path.exists(self.final_output):
            os.remove(self.final_output)
            print(f"Removed existing final output: {self.final_output}")
    
    def split_story_into_chunks(self, story_text: str, chunk_size: int = CHUNK_SIZE) -> list:
        """Split story into chunks of specified line count"""
        lines = story_text.strip().split('\n')
        chunks = []
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            
            chunks.append({
                'chunk_number': (i // chunk_size) + 1,
                'start_line': i + 1,
                'end_line': min(i + chunk_size, len(lines)),
                'text': chunk_text,
                'line_count': len(chunk_lines)
            })
        
        return chunks
    
    def load_story_workflow(self):
        """Load the story workflow from JSON"""
        wf = '../workflow/story.json'
        with open(wf, 'r') as f:
            return json.load(f)
    
    def find_node_by_type(self, workflow, node_type):
        """Find a node by its type"""
        for node_id, node in workflow.items():
            if node.get('class_type') == node_type:
                return node
        return None
    
    def update_workflow_text(self, workflow, story_text):
        """Update the text prompt in the workflow"""
        # Find the PrimitiveStringMultiline node and update its text
        node = self.find_node_by_type(workflow, 'PrimitiveStringMultiline')
        if node:
            node['inputs']['value'] = story_text
        return workflow
    
    def update_workflow_filename(self, workflow, filename):
        """Update the filename in the SaveAudioMP3 node"""
        # Find the SaveAudioMP3 node and update its filename
        node = self.find_node_by_type(workflow, 'SaveAudioMP3')
        if node:
            node['inputs']['filename_prefix'] = f"audio/{filename}"
        return workflow
    
    def generate_chunk_audio(self, chunk_text, chunk_number, start_line, end_line):
        """Generate audio for a single chunk and save to output/story/start_line_end_line.wav"""
        try:
            print(f"Generating audio for chunk {chunk_number} (lines {start_line}-{end_line})...")
            print(f"Chunk length: {len(chunk_text)} characters")
            
            # Load and update workflow
            workflow = self.load_story_workflow()
            workflow = self.update_workflow_text(workflow, chunk_text)
            workflow = self.update_workflow_filename(workflow, f"chunk_{chunk_number}")
            
            # Send workflow to ComfyUI
            print(f"Sending workflow to ComfyUI...")
            try:
                response = requests.post(f"{self.comfyui_url}prompt", json={"prompt": workflow}, timeout=60)
                print(f"Response status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"ComfyUI response: {response.text}")
                    raise Exception(f"Failed to send workflow: {response.text}")
                
                response_json = response.json()
                print(f"Response JSON: {response_json}")
                prompt_id = response_json["prompt_id"]
                print(f"Workflow sent successfully, prompt_id: {prompt_id}")
                
            except requests.exceptions.Timeout:
                print("Timeout while sending workflow to ComfyUI")
                raise Exception("Timeout while sending workflow to ComfyUI")
            except Exception as e:
                print(f"Error sending workflow: {e}")
                raise
            
            # Wait for completion
            print("Waiting for audio generation to complete...")
            while True:
                history_response = requests.get(f"{self.comfyui_url}history/{prompt_id}")
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    if prompt_id in history_data:
                        status = history_data[prompt_id].get('status', {})
                        if status.get('exec_info', {}).get('queue_remaining', 0) == 0:
                            # Check if there are outputs
                            outputs = history_data[prompt_id].get('outputs', {})
                            for node_id, node_output in outputs.items():
                                if 'audio' in node_output:
                                    for audio_file in node_output['audio']:
                                        # The file is already saved in the ComfyUI output folder
                                        # Wait a bit longer for the file to be written
                                        time.sleep(3)
                                        
                                        # Look for the generated file in the output folder
                                        print(f"Looking for generated audio file in {self.output_folder}")
                                        files_in_folder = os.listdir(self.output_folder)
                                        print(f"Files in folder: {files_in_folder}")
                                        
                                        # Look for files that start with "chunk_" and end with .mp3
                                        matching_files = []
                                        for file in files_in_folder:
                                            if file.startswith(f"chunk_{chunk_number}") and file.endswith('.mp3'):
                                                matching_files.append(file)
                                        
                                        if matching_files:
                                            # Sort by modification time and get the most recent
                                            matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_folder, x)), reverse=True)
                                            most_recent_file = matching_files[0]
                                            source_path = os.path.join(self.output_folder, most_recent_file)
                                            print(f"Found generated file: {source_path}")
                                            
                                            # Convert MP3 to WAV and save to chunk directory
                                            chunk_filename = f"{start_line}_{end_line}.wav"
                                            chunk_output_path = os.path.join(self.chunk_output_dir, chunk_filename)
                                            print(f"Converting {source_path} to WAV format...")
                                            audio = AudioSegment.from_mp3(source_path)
                                            audio.export(chunk_output_path, format="wav")
                                            print(f"Converted and saved as {chunk_output_path}")
                                            return chunk_output_path
                                        
                                        print(f"No files found starting with 'chunk_{chunk_number}'")
                            break
                time.sleep(5)
            
            raise Exception("Failed to generate chunk audio")
                
        except Exception as e:
            print(f"Error generating chunk audio: {e}")
            return None
    
    def concatenate_chunks(self, chunk_files):
        """Concatenate all chunk audio files into final story.wav"""
        try:
            print("Concatenating all chunks into final story...")
            
            if not chunk_files:
                print("No chunk files to concatenate")
                return None
            
            # Load all audio segments
            audio_segments = []
            for chunk_file in chunk_files:
                if os.path.exists(chunk_file):
                    print(f"Loading chunk: {chunk_file}")
                    audio = AudioSegment.from_wav(chunk_file)
                    audio_segments.append(audio)
                else:
                    print(f"Warning: Chunk file not found: {chunk_file}")
            
            if not audio_segments:
                print("No valid audio segments to concatenate")
                return None
            
            # Concatenate all segments
            print(f"Concatenating {len(audio_segments)} audio segments...")
            final_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                final_audio += segment
            
            # Export final audio
            final_audio.export(self.final_output, format="wav")
            print(f"Final story saved as: {self.final_output}")
            
            # Display final statistics
            duration_seconds = len(final_audio) / 1000.0
            duration_minutes = duration_seconds / 60.0
            print(f"Final audio duration: {duration_seconds:.1f} seconds ({duration_minutes:.1f} minutes)")
            
            return self.final_output
            
        except Exception as e:
            print(f"Error concatenating chunks: {e}")
            return None
    
    def process_story_chunks(self, story_text, resumable_state=None):
        """Process story in chunks with resumable functionality"""
        print("Processing story in chunks...")
        
        if not story_text or story_text.strip() == "":
            print("Error: Story text is empty")
            return None
        
        # Split story into chunks
        chunks = self.split_story_into_chunks(story_text)
        total_chunks = len(chunks)
        
        print(f"Story split into {total_chunks} chunks of {CHUNK_SIZE} lines each")
        
        chunk_files = []
        successful_chunks = 0
        failed_chunks = 0
        skipped_chunks = 0
        
        # Calculate total lines for percentage calculation
        total_lines = len(story_text.strip().split('\n'))
        
        print(f"\n📊 STORY PROCESSING PROGRESS")
        print("=" * 80)
        print(f"{'Chunk':<6} {'Lines':<10} {'Progress':<12} {'Status':<15} {'Output':<30}")
        print("-" * 80)
        
        for chunk in chunks:
            chunk_number = chunk['chunk_number']
            start_line = chunk['start_line']
            end_line = chunk['end_line']
            chunk_text = chunk['text']
            
            # Calculate progress percentage
            lines_processed = end_line
            progress_percent = (lines_processed / total_lines) * 100
            
            # Check if resumable and chunk already complete
            if resumable_state and resumable_state.is_chunk_complete(chunk_number):
                cached_chunk = resumable_state.get_chunk_result(chunk_number)
                if cached_chunk and cached_chunk.get('output_file'):
                    print(f"{chunk_number:<6} {start_line}-{end_line:<8} {progress_percent:6.1f}%     {'CACHED':<15} {os.path.basename(cached_chunk['output_file']):<30}")
                    chunk_files.append(cached_chunk['output_file'])
                    successful_chunks += 1
                    skipped_chunks += 1
                    continue
            
            # Show progress before starting
            print(f"{chunk_number:<6} {start_line}-{end_line:<8} {progress_percent:6.1f}%     {'PROCESSING':<15} {'Generating audio...':<30}")
            
            try:
                # Generate audio for this chunk
                chunk_output = self.generate_chunk_audio(chunk_text, chunk_number, start_line, end_line)
                
                if chunk_output:
                    chunk_files.append(chunk_output)
                    successful_chunks += 1
                    print(f"{chunk_number:<6} {start_line}-{end_line:<8} {progress_percent:6.1f}%     {'✅ COMPLETED':<15} {os.path.basename(chunk_output):<30}")
                    
                    # Save to checkpoint if resumable mode enabled
                    if resumable_state:
                        chunk_data = {
                            'chunk_number': chunk_number,
                            'start_line': start_line,
                            'end_line': end_line,
                            'output_file': chunk_output,
                            'success': True
                        }
                        resumable_state.set_chunk_result(chunk_number, chunk_data)
                else:
                    failed_chunks += 1
                    print(f"{chunk_number:<6} {start_line}-{end_line:<8} {progress_percent:6.1f}%     {'❌ FAILED':<15} {'Audio generation failed':<30}")
                    
                    # Save error state to checkpoint if resumable mode enabled
                    if resumable_state:
                        chunk_data = {
                            'chunk_number': chunk_number,
                            'start_line': start_line,
                            'end_line': end_line,
                            'output_file': None,
                            'success': False,
                            'error': 'Audio generation failed'
                        }
                        resumable_state.set_chunk_result(chunk_number, chunk_data)
                
                # Small delay between chunks
                if chunk_number < total_chunks:
                    time.sleep(2)
                    
            except Exception as e:
                failed_chunks += 1
                print(f"{chunk_number:<6} {start_line}-{end_line:<8} {progress_percent:6.1f}%     {'❌ ERROR':<15} {str(e)[:30]:<30}")
                
                # Save error state to checkpoint if resumable mode enabled
                if resumable_state:
                    chunk_data = {
                        'chunk_number': chunk_number,
                        'start_line': start_line,
                        'end_line': end_line,
                        'output_file': None,
                        'success': False,
                        'error': str(e)
                    }
                    resumable_state.set_chunk_result(chunk_number, chunk_data)
        
        print("-" * 80)
        print(f"📊 FINAL SUMMARY:")
        print(f"   ✅ Successful: {successful_chunks}")
        print(f"   📁 Cached: {skipped_chunks}")
        print(f"   ❌ Failed: {failed_chunks}")
        print(f"   📁 Total chunk files: {len(chunk_files)}")
        print("=" * 80)
        
        return chunk_files
    
    def process_story(self, story_text, resumable_state=None):
        """Main processing function with resumable support"""
        print("Processing story...")
        
        if not story_text or story_text.strip() == "":
            print("Error: Story text is empty")
            return None

        # Check if resumable and final concatenation already complete
        if resumable_state and resumable_state.is_final_concatenation_complete():
            cached_final = resumable_state.get_final_result()
            if cached_final and os.path.exists(cached_final):
                print("Using cached final result from checkpoint")
                return cached_final

        # Process story in chunks
        chunk_files = self.process_story_chunks(story_text, resumable_state)
        
        if not chunk_files:
            print("No chunk files generated - story processing failed!")
            return None
        
        # Concatenate all chunks into final story
        print("\n=== FINAL CONCATENATION PHASE ===")
        final_audio = self.concatenate_chunks(chunk_files)
        
        if final_audio:
            # Save final result to checkpoint if resumable mode enabled
            if resumable_state:
                resumable_state.set_final_result(final_audio)
            
            print("Story processing complete!")
            return final_audio
        else:
            print("Story processing failed!")
            return None

def read_story_from_file(filename="../input/1.story.txt"):
    """Read story data from a text file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Story file '{filename}' not found.")
        print("Please create a ../input/1.story.txt file with your story text.")
        return None
    except Exception as e:
        print(f"Error reading story file: {e}")
        return None

if __name__ == "__main__":
    # Parse CLI arguments for resumable mode control
    parser = argparse.ArgumentParser(description="Resumable story audio generation")
    parser.add_argument("--force-start", action="store_true", help="Force start from beginning, ignoring any existing checkpoint files")
    parser.add_argument("--disable-resumable", action="store_true", help="Disable resumable mode (default: enabled)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help=f"Number of lines per chunk (default: {CHUNK_SIZE})")
    args = parser.parse_args()
    
    # Update configuration based on CLI arguments
    if args.disable_resumable:
        ENABLE_RESUMABLE_MODE = False
        print("🚫 Resumable mode disabled via --disable-resumable")
    else:
        print("✅ Resumable mode enabled (use --disable-resumable to disable)")
    
    # Update chunk size if provided
    if args.chunk_size != CHUNK_SIZE:
        CHUNK_SIZE = args.chunk_size
        print(f"📏 Chunk size set to: {CHUNK_SIZE} lines per chunk")
    
    start_time = time.time()
    
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
    
    # Read story data from file
    file_read_start = time.time()
    story_text = read_story_from_file()
    file_read_time = time.time() - file_read_start
    
    if story_text is None:
        print("Exiting due to story file error.")
        exit(1)
    
    # Create processor and run
    processor = StoryProcessor()
    
    # Time the story processing
    processing_start = time.time()
    final_audio = processor.process_story(story_text, resumable_state)
    processing_time = time.time() - processing_start
    
    if final_audio:
        print(f"Final audio file: {final_audio}")
    else:
        print("Failed to generate story audio")
        exit(1)
    
    # Clean up checkpoint files if resumable mode was used and everything completed successfully
    if resumable_state:
        print("All operations completed successfully")
        print("Final progress:", resumable_state.get_progress_summary())
        resumable_state.cleanup()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print detailed timing information
    print("\n" + "=" * 50)
    print("⏱️  TIMING SUMMARY")
    print("=" * 50)
    print(f"📖 File reading time: {file_read_time:.3f} seconds")
    print(f"🎵 Story processing time: {processing_time:.3f} seconds")
    print(f"⏱️  Total execution time: {total_time:.3f} seconds ({total_time/60:.3f} minutes)")
    print("=" * 50)
