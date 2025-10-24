#!/usr/bin/env python3
"""
Script to process video story input and generate image story outputs.
Takes gen.video/input/1.story.txt as input and outputs:
- gen.image/input/1.story.txt (scene descriptions for image generation)
"""

import re
import os
from pathlib import Path

def parse_video_story(input_file):
    """Parse the video story file to extract scene information."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    scenes = []
    dialogue = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract scene descriptions (scene_X.X format)
        scene_match = re.match(r'\(scene_(\d+\.\d+)\)\s*(.+)', line)
        if scene_match:
            scene_id = scene_match.group(1)
            scene_content = scene_match.group(2)
            scenes.append(f"(scene_{scene_id}) {scene_content}")
            continue
            
        # Extract dialogue and narration
        if line.startswith('[') and ']' in line:
            dialogue.append(line)
        elif line.startswith('(') and 'narration' in line:
            dialogue.append(line)
    
    return scenes, dialogue

def write_image_story_file(scenes, dialogue, output_file, input_file):
    """Write scene descriptions and dialogue to the image story output file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write dialogue and scenes in the order they appear
        with open(input_file, 'r', encoding='utf-8') as input_f:
            content = input_f.read()
        
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                f.write('\n')
                continue
                
            # Write dialogue lines
            if line.startswith('[') and ']' in line:
                f.write(line + '\n')
            # Write scene descriptions
            elif line.startswith('(scene_'):
                f.write(line + '\n')
            # Write narration
            elif line.startswith('(') and 'narration' in line:
                f.write(line + '\n')

def main():
    """Main function to process the video story and generate outputs."""
    # Get base directory and normalize paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input file
    input_file = os.path.normpath(os.path.join(base_dir, "../input/1.story.txt"))
    
    # Output files
    image_story_output = os.path.normpath(os.path.join(base_dir, "../../gen.image/input/1.story.txt"))
    
    print(f"Processing video story from: {input_file}")
    
    # Parse the input file
    scenes, dialogue = parse_video_story(input_file)
    
    print(f"Found {len(scenes)} scene descriptions")
    print(f"Found {len(dialogue)} dialogue/narration lines")
    
    # Write image story file
    write_image_story_file(scenes, dialogue, image_story_output, input_file)
    print(f"Image story written to: {image_story_output}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
