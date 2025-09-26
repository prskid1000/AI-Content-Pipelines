#!/usr/bin/env python3
"""
Script to process video story input and generate motion and image story outputs.
Takes gen.video/input/1.story.txt as input and outputs:
- gen.video/input/2.motion.txt (motion descriptions)
- gen.image/input/1.story.txt (scene descriptions for image generation)
"""

import re
import os
from pathlib import Path

def parse_video_story(input_file):
    """Parse the video story file to extract motion and scene information."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    motions = []
    scenes = []
    dialogue = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract motion descriptions (motion_X.X format)
        motion_match = re.match(r'\(motion_(\d+\.\d+)\)\s*(.+)', line)
        if motion_match:
            motion_id = motion_match.group(1)
            motion_content = motion_match.group(2)
            motions.append(f"(motion_{motion_id}) {motion_content}")
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
    
    return motions, scenes, dialogue

def write_motion_file(motions, output_file):
    """Write motion descriptions to the motion output file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for motion in motions:
            f.write(motion + '\n')

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
            # Skip motion descriptions for image story

def main():
    """Main function to process the video story and generate outputs."""
    # Get base directory and normalize paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input file
    input_file = os.path.normpath(os.path.join(base_dir, "../input/1.story.txt"))
    
    # Output files
    motion_output = os.path.normpath(os.path.join(base_dir, "../input/2.motion.txt"))
    image_story_output = os.path.normpath(os.path.join(base_dir, "../../gen.image/input/1.story.txt"))
    
    print(f"Processing video story from: {input_file}")
    
    # Parse the input file
    motions, scenes, dialogue = parse_video_story(input_file)
    
    print(f"Found {len(motions)} motion descriptions")
    print(f"Found {len(scenes)} scene descriptions")
    print(f"Found {len(dialogue)} dialogue/narration lines")
    
    # Write motion file
    write_motion_file(motions, motion_output)
    print(f"Motion descriptions written to: {motion_output}")
    
    # Write image story file
    write_image_story_file(scenes, dialogue, image_story_output, input_file)
    print(f"Image story written to: {image_story_output}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
