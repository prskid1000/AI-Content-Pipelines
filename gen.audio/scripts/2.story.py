import json
import requests
import time
import os
import shutil
import re
from pydub import AudioSegment
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

 

class StoryProcessor:
    def __init__(self, comfyui_url="http://127.0.0.1:8188/"):
        self.comfyui_url = comfyui_url
        self.output_folder = "../../ComfyUI/output/audio"
        self.final_output = "../output/story.wav"
        
        # Clear the final output file if it exists
        if os.path.exists(self.final_output):
            os.remove(self.final_output)
            print(f"Removed existing final output: {self.final_output}")
    
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
    
    def generate_story_audio(self, story_text):
        """Generate story audio from the provided text"""
        try:
            print(f"Generating story audio...")
            print(f"Story length: {len(story_text)} characters")
            
            # Load and update workflow
            workflow = self.load_story_workflow()
            workflow = self.update_workflow_text(workflow, story_text)
            workflow = self.update_workflow_filename(workflow, "story")
            
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
                                        
                                        # Look for files that start with "story" and end with .mp3
                                        matching_files = []
                                        for file in files_in_folder:
                                            if file.startswith("story") and file.endswith('.mp3'):
                                                matching_files.append(file)
                                        
                                        if matching_files:
                                            # Sort by modification time and get the most recent
                                            matching_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.output_folder, x)), reverse=True)
                                            most_recent_file = matching_files[0]
                                            source_path = os.path.join(self.output_folder, most_recent_file)
                                            print(f"Found generated file: {source_path}")
                                            
                                            # Convert MP3 to WAV and save to current directory
                                            print(f"Converting {source_path} to WAV format...")
                                            audio = AudioSegment.from_mp3(source_path)
                                            audio.export(self.final_output, format="wav")
                                            print(f"Converted and saved as {self.final_output}")
                                            return self.final_output
                                        
                                        print(f"No files found starting with 'story'")
                            break
                time.sleep(5)
            
            raise Exception("Failed to generate story audio")
                
        except Exception as e:
            print(f"Error generating story audio: {e}")
            return None
    
    def process_story(self, story_text):
        """Main processing function"""
        print("Processing story...")
        
        if not story_text or story_text.strip() == "":
            print("Error: Story text is empty")
            return None

        # Generate story audio
        final_audio = self.generate_story_audio(story_text)
        
        if final_audio:
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
    start_time = time.time()
    
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
    final_audio = processor.process_story(story_text)
    processing_time = time.time() - processing_start
    
    if final_audio:
        print(f"Final audio file: {final_audio}")
    else:
        print("Failed to generate story audio")
        exit(1)
    
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
