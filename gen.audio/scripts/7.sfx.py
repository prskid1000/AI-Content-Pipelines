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
WORKFLOW_SUMMARY_ENABLED = False  # Set to True to enable workflow summary printing

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
    
    def validate_and_cleanup_results(self, output_sfx_dir: str = None) -> int:
        """Validate that all completed audio files actually exist and clean up missing entries.
        
        Args:
            output_sfx_dir: Path to the output/sfx directory to check for actual files
        
        Returns:
            int: Number of entries cleaned up (removed from completed list)
        """
        cleaned_count = 0
        audio_files_to_remove = []
        
        print(f"Validating {len(self.state['audio_files']['completed'])} completed audio files against output/sfx directory...")
        
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
            elif output_sfx_dir:
                # Additional check: verify file exists in output/sfx directory
                expected_sfx_file = os.path.join(output_sfx_dir, os.path.basename(output_file_path))
                if not os.path.exists(expected_sfx_file):
                    print(f"Precheck: SFX file missing in output/sfx directory for {file_key} - marking as not completed")
                    print(f"  Expected: {expected_sfx_file}")
                    audio_files_to_remove.append(file_key)
                    cleaned_count += 1
                else:
                    print(f"Precheck: ✓ {file_key} validated in output/sfx directory")
        
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
    
    def sync_with_output_directory(self, output_sfx_dir: str) -> int:
        """Sync resumable state with actual files in output directory.
        
        This method finds files that exist in the output directory but aren't tracked
        in the resumable state, and adds them to the completed list.
        
        Args:
            output_sfx_dir: Path to the output/sfx directory to check for actual files
        
        Returns:
            int: Number of files found and added to completed list
        """
        if not os.path.exists(output_sfx_dir):
            print(f"Output/sfx directory does not exist: {output_sfx_dir}")
            return 0
            
        added_count = 0
        tracked_files = set(self.state["audio_files"]["completed"])
        
        print(f"Scanning output/sfx directory for untracked files: {output_sfx_dir}")
        
        # Find all audio files in the output directory
        for filename in os.listdir(output_sfx_dir):
            if filename.endswith(('.flac', '.wav', '.mp3')):
                # Create a file key based on filename
                file_key = f"auto_detected_{filename}"
                
                # If this file isn't tracked, add it to completed
                if file_key not in tracked_files:
                    file_path = os.path.join(output_sfx_dir, filename)
                    result = {
                        'file': file_path,
                        'output_file': file_path,
                        'auto_detected': True
                    }
                    self.state["audio_files"]["results"][file_key] = result
                    self.state["audio_files"]["completed"].append(file_key)
                    added_count += 1
                    print(f"Auto-detected completed SFX file: {file_key} -> {file_path}")
                else:
                    print(f"SFX file already tracked: {file_key}")
        
        # Save state if any files were added
        if added_count > 0:
            self._save_state()
            print(f"Auto-detection: Added {added_count} files from output/sfx directory")
        else:
            print("No untracked SFX files found in output/sfx directory")
        
        return added_count
    
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
        """Estimate remaining time based on audio duration - simple and accurate approach"""
        
        # Calculate total audio duration if we have timeline data
        total_audio_duration = getattr(self, 'total_audio_duration', 0)
        
        # For first file with no data, provide a reasonable initial estimate
        if not self.processing_times and file_processing_time is None:
            if total_audio_duration > 0:
                # Initial estimate: assume 10x realtime for SFX generation (10 seconds of audio takes 100 seconds to generate)
                estimated_remaining_seconds = total_audio_duration * 10
            else:
                # Fallback: assume 20 seconds per file
                remaining_files = total_files - current_file
                estimated_remaining_seconds = remaining_files * 20.0
            
            return self._format_time_with_confidence(estimated_remaining_seconds, confidence="low")
        
        # Calculate actual generation speed from completed files
        if file_processing_time and audio_duration is not None:
            # Store audio processing data for better estimation
            if not hasattr(self, 'audio_processing_data'):
                self.audio_processing_data = []
            self.audio_processing_data.append({
                'audio_duration': audio_duration, 
                'processing_time': file_processing_time, 
                'is_silence': is_silence,
                'speed_ratio': file_processing_time / audio_duration if audio_duration > 0 else 1.0
            })
        
        # Use duration-based estimation if we have audio processing data
        if hasattr(self, 'audio_processing_data') and self.audio_processing_data:
            # Calculate average generation speed ratio (processing_time / audio_duration)
            total_audio_duration_processed = sum(data['audio_duration'] for data in self.audio_processing_data)
            total_processing_time = sum(data['processing_time'] for data in self.audio_processing_data)
            
            if total_audio_duration_processed > 0:
                actual_speed_ratio = total_processing_time / total_audio_duration_processed
                
                # Estimate remaining audio duration
                audio_duration_processed_so_far = sum(data['audio_duration'] for data in self.audio_processing_data)
                remaining_audio_duration = total_audio_duration - audio_duration_processed_so_far
                
                # Calculate remaining time based on actual speed ratio
                estimated_remaining_seconds = remaining_audio_duration * actual_speed_ratio
                
                # Determine confidence based on data points
                confidence = "low"
                if len(self.audio_processing_data) >= 5:
                    confidence = "high"
                elif len(self.audio_processing_data) >= 3:
                    confidence = "medium"
                
                return self._format_time_with_confidence(estimated_remaining_seconds, confidence)
        
        # Fallback to file-based estimation if no audio duration data available
        if file_processing_time:
            all_times = self.processing_times + [file_processing_time]
        else:
            all_times = self.processing_times
        
        # Simple average of recent processing times
        estimated_time_per_file = sum(all_times) / len(all_times)
        remaining_files = total_files - current_file
        estimated_remaining_seconds = remaining_files * estimated_time_per_file
        
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
    
    def _print_workflow_summary(self, workflow: dict, title: str) -> None:
        """Print a comprehensive workflow summary showing the flow to sampler inputs."""
        if not WORKFLOW_SUMMARY_ENABLED:
            return
        print(f"\n🔗 WORKFLOW SUMMARY: {title}")
        print("   📊 KSampler(3) - Core Parameters:")
        
        # Find KSampler node
        sampler_node = None
        for node_id, node in workflow.items():
            if node.get("class_type") == "KSampler":
                sampler_node = node
                break
        
        if sampler_node:
            inputs = sampler_node.get("inputs", {})
            print(f"      🎲 Seed: {inputs.get('seed', 'N/A')}")
            print(f"      📈 Steps: {inputs.get('steps', 'N/A')}")
            print(f"      🎯 CFG: {inputs.get('cfg', 'N/A')}")
            print(f"      🎲 Sampler: {inputs.get('sampler_name', 'N/A')}")
            print(f"      📊 Scheduler: {inputs.get('scheduler', 'N/A')}")
            print(f"      🔄 Denoise: {inputs.get('denoise', 'N/A')}")
            
            # Trace input flows
            self._trace_input_flow(workflow, "model", inputs.get("model", [None, 0])[0], inputs.get("model", [None, 0])[1], "3")
            self._trace_input_flow(workflow, "positive", inputs.get("positive", [None, 0])[0], inputs.get("positive", [None, 0])[1], "3")
            self._trace_input_flow(workflow, "negative", inputs.get("negative", [None, 0])[0], inputs.get("negative", [None, 0])[1], "3")
            self._trace_input_flow(workflow, "latent_image", inputs.get("latent_image", [None, 0])[0], inputs.get("latent_image", [None, 0])[1], "3")
        
        print("   " + "="*50)
    
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
            
        elif node_type == "EmptyLatentAudio":
            print(f"{indent}🎵 Seconds: {node_inputs.get('seconds', 'N/A')}")
            print(f"{indent}📦 Batch: {node_inputs.get('batch_size', 'N/A')}")
            
        elif node_type == "CheckpointLoaderSimple":
            print(f"{indent}📦 Checkpoint: {node_inputs.get('ckpt_name', 'N/A')}")
            
        elif node_type == "CLIPLoader":
            print(f"{indent}📖 Type: CLIPLoader")
            print(f"{indent}📁 Clip: {node_inputs.get('clip_name', 'N/A')}")
            print(f"{indent}⚙️ type: {node_inputs.get('type', 'N/A')}")
            print(f"{indent}⚙️ device: {node_inputs.get('device', 'N/A')}")
            
        elif node_type == "VAEDecodeAudio":
            print(f"{indent}🔄 VAE: {node_inputs.get('vae', 'N/A')}")
            print(f"{indent}📦 Samples: {node_inputs.get('samples', 'N/A')}")
            
        elif node_type == "SaveAudio":
            print(f"{indent}💾 Prefix: {node_inputs.get('filename_prefix', 'N/A')}")
            
        elif node_type == "SaveAudioMP3":
            print(f"{indent}💾 Prefix: {node_inputs.get('filename_prefix', 'N/A')}")
            print(f"{indent}🎵 Quality: {node_inputs.get('quality', 'N/A')}")
            
        # Show any other relevant parameters
        for key, value in node_inputs.items():
            if key not in ['model', 'clip', 'vae', 'pixels', 'samples', 'image', 'text', 'lora_name', 'strength_model', 'strength_clip', 'model_name', 'device', 'seconds', 'batch_size', 'filename_prefix', 'quality', 'clip_name', 'type', 'ckpt_name']:
                if isinstance(value, (str, int, float, bool)) and len(str(value)) < 50:
                    print(f"{indent}⚙️ {key}: {value}")
    
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
        
        # Check if file already exists in output folder (skip regeneration)
        output_filename = f"{filename}.flac"
        output_file_path = os.path.join(self.final_output_folder, output_filename)
        
        # Check for exact match first
        if os.path.exists(output_file_path):
            # File already exists, check if we can use it
            comfyui_file_path = os.path.join(self.output_folder, output_filename)
            # Prefer ComfyUI output folder file if it exists, otherwise use the output folder file
            if os.path.exists(comfyui_file_path):
                found_path = comfyui_file_path
            else:
                found_path = output_file_path
            
            result = {
                'file': found_path,
                'output_file': output_file_path,
                'order_index': i,
                'duration': duration,
                'description': entry['description']
            }
            
            # Update resumable state if enabled
            if resumable_state:
                resumable_state.set_audio_file_info(file_key, result)
            
            entry_type = "Silence" if self.is_silence_entry(entry['description']) else "SFX"
            print(f"⏭️  Skipping {entry_type} ({duration}s) - File already exists: {output_filename}")
            return result
        
        # Check for files that start with the filename pattern (ComfyUI may add suffixes)
        if os.path.exists(self.final_output_folder):
            matching_files = [f for f in os.listdir(self.final_output_folder) 
                            if f.startswith(filename) and f.endswith('.flac')]
            if matching_files:
                # Use the most recently modified file
                matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.final_output_folder, x)), reverse=True)
                found_output_file = os.path.join(self.final_output_folder, matching_files[0])
                
                # Check ComfyUI folder for matching file
                comfyui_matching = None
                if os.path.exists(self.output_folder):
                    comfyui_matching = [f for f in os.listdir(self.output_folder) 
                                      if f.startswith(filename) and f.endswith('.flac')]
                    if comfyui_matching:
                        comfyui_matching.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_folder, x)), reverse=True)
                        found_path = os.path.join(self.output_folder, comfyui_matching[0])
                    else:
                        found_path = found_output_file
                else:
                    found_path = found_output_file
                
                result = {
                    'file': found_path,
                    'output_file': found_output_file,
                    'order_index': i,
                    'duration': duration,
                    'description': entry['description']
                }
                
                # Update resumable state if enabled
                if resumable_state:
                    resumable_state.set_audio_file_info(file_key, result)
                
                entry_type = "Silence" if self.is_silence_entry(entry['description']) else "SFX"
                print(f"⏭️  Skipping {entry_type} ({duration}s) - File already exists: {matching_files[0]}")
                return result
        
        # Check if resumable and already complete
        if resumable_state and resumable_state.is_audio_file_complete(file_key):
            cached_info = resumable_state.get_audio_file_info(file_key)
            # Check both the main file and output file paths
            main_file_exists = cached_info and os.path.exists(cached_info.get('file', ''))
            output_file_exists = cached_info and os.path.exists(cached_info.get('output_file', ''))
            
            if main_file_exists or output_file_exists:
                # If output file exists, use it; otherwise use main file
                if output_file_exists:
                    cached_info['file'] = cached_info.get('output_file')
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
            
            # Print workflow summary
            self._print_workflow_summary(workflow, f"SFX: {entry['description']}")
            
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
        
        # Calculate total audio duration for ETA calculations
        self.total_audio_duration = sum(entry['seconds'] for entry in timeline_entries)
        
        # Count silence vs non-silence entries
        silence_count = sum(1 for _, entry in batch_data if self.is_silence_entry(entry['description']))
        sfx_count = len(batch_data) - silence_count
        
        print(f"📊 Processing {len(batch_data)} entries: {silence_count} silence, {sfx_count} SFX")
        print(f"📊 Total audio duration: {self.total_audio_duration:.1f} seconds ({self.total_audio_duration/60:.1f} minutes)")
        
        # Initialize start time for time estimation
        self.start_time = time.time()
        
        print(f"\n📊 SFX GENERATION PROGRESS")
        print("=" * 100)
        
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
                    print(f"✅ File {completed_count}/{len(batch_data)} - {entry_type} ({result['duration']:.3f}s) - {result['description'][:40]} - Completed")
                    print(f"📊 Estimated time remaining: {eta}")
                else:
                    entry_data = future_to_entry[future]
                    entry_type = "Silence" if self.is_silence_entry(entry_data[1]['description']) else "SFX"
                    is_silence = self.is_silence_entry(entry_data[1]['description'])
                    eta = self.estimate_remaining_time(completed_count, len(batch_data), audio_duration=entry_data[1]['seconds'], is_silence=is_silence, description=entry_data[1]['description'])
                    print(f"❌ File {completed_count}/{len(batch_data)} - {entry_type} ({entry_data[1]['seconds']:.3f}s) - {entry_data[1]['description'][:40]} - Failed")
                    print(f"📊 Estimated time remaining: {eta}")
        
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
