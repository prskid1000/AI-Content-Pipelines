#!/usr/bin/env python3
"""
Video Combiner Script

This script combines all videos from the animation output folder into a single video,
then merges it with the final audio to produce the complete video with sound.

Features:
- Combines all MP4 videos from ../output/animation directory
- Sorts videos by scene ID for proper chronological order
- Merges the combined video with audio from gen.audio/output/final.wav
- Outputs final video to ../output/final.mp4
- Uses FFmpeg for video processing
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Optional


class VideoCombiner:
    def __init__(self):
        # Directory paths relative to script location
        self.animation_dir = Path("../output/animation")
        self.output_dir = Path("../output")
        self.audio_file = Path("../../gen.audio/output/final.wav")
        
        # Output files
        self.combined_video = self.output_dir / "combined.mp4"
        self.final_video = self.output_dir / "final.mp4"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available in the system."""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _get_video_files(self) -> List[Path]:
        """Get all MP4 video files from the animation directory, sorted by scene ID."""
        if not self.animation_dir.exists():
            print(f"ERROR: Animation directory not found: {self.animation_dir}")
            return []
        
        video_files = []
        for file in self.animation_dir.glob("*.mp4"):
            video_files.append(file)
        
        if not video_files:
            print(f"ERROR: No MP4 files found in {self.animation_dir}")
            return []
        
        # Sort by scene ID (extract numeric parts for proper ordering)
        def extract_scene_key(filepath: Path) -> tuple:
            """Extract sorting key from scene filename like 'scene_1.1.mp4'."""
            name = filepath.stem
            # Find all numeric parts in the filename
            numbers = re.findall(r'\d+', name)
            # Convert to integers for proper numeric sorting
            return tuple(int(n) for n in numbers) if numbers else (0,)
        
        video_files.sort(key=extract_scene_key)
        print(f"Found {len(video_files)} video files to combine:")
        for i, video in enumerate(video_files, 1):
            print(f"  {i}. {video.name}")
        
        return video_files

    def _create_file_list(self, video_files: List[Path]) -> Path:
        """Create a temporary file list for FFmpeg concat."""
        file_list_path = self.output_dir / "video_list.txt"
        
        try:
            with open(file_list_path, 'w', encoding='utf-8') as f:
                for video_file in video_files:
                    # Use absolute paths to avoid issues
                    abs_path = video_file.resolve()
                    f.write(f"file '{abs_path}'\n")
            
            print(f"Created file list: {file_list_path}")
            return file_list_path
            
        except Exception as e:
            print(f"ERROR: Failed to create file list: {e}")
            raise

    def combine_videos(self) -> bool:
        """Combine all animation videos into a single video."""
        
        print("=" * 60)
        print("STEP 1: Combining animation videos")
        print("=" * 60)
        
        # Check FFmpeg
        if not self._check_ffmpeg():
            print("ERROR: FFmpeg not found. Please install FFmpeg and add it to your PATH.")
            return False
        
        # Get video files
        video_files = self._get_video_files()
        if not video_files:
            return False
        
        # Create file list for concat
        try:
            file_list_path = self._create_file_list(video_files)
        except Exception:
            return False
        
        # Combine videos using FFmpeg concat
        print(f"\nCombining {len(video_files)} videos...")
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(file_list_path),
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-y',  # Overwrite output file
            str(self.combined_video)
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✓ Successfully combined videos: {self.combined_video}")
                # Clean up temporary file list
                file_list_path.unlink(missing_ok=True)
                return True
            else:
                print(f"ERROR: FFmpeg failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("ERROR: FFmpeg timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"ERROR: Failed to run FFmpeg: {e}")
            return False

    def merge_with_audio(self) -> bool:
        """Merge combined video with audio to create final video."""
        if not self.combined_video.exists():
            print(f"ERROR: Combined video not found: {self.combined_video}")
            return False
        
        if not self.audio_file.exists():
            print(f"ERROR: Audio file not found: {self.audio_file}")
            return False
        
        print("=" * 60)
        print("STEP 2: Merging video with audio")
        print("=" * 60)
        
        print(f"Video: {self.combined_video}")
        print(f"Audio: {self.audio_file}")
        print(f"Output: {self.final_video}")
        
        # Merge video and audio
        cmd = [
            'ffmpeg',
            '-i', str(self.combined_video),
            '-i', str(self.audio_file),
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'aac',   # Encode audio as AAC
            '-map', '0:v:0', # Use video from first input
            '-map', '1:a:0', # Use audio from second input
            '-shortest',     # End when shortest stream ends
            '-y',            # Overwrite output file
            str(self.final_video)
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"✓ Successfully created final video: {self.final_video}")
                
                # Show file info
                if self.final_video.exists():
                    size_mb = self.final_video.stat().st_size / (1024 * 1024)
                    print(f"  File size: {size_mb:.1f} MB")
                
                return True
            else:
                print(f"ERROR: FFmpeg failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("ERROR: FFmpeg timed out after 10 minutes")
            return False
        except Exception as e:
            print(f"ERROR: Failed to merge video and audio: {e}")
            return False

    def process_all(self) -> bool:
        """Run the complete video processing pipeline."""
        print("🎬 Video Processing Pipeline")
        print("=" * 60)
        
        # Step 1: Combine videos
        if not self.combine_videos():
            print("❌ Failed to combine videos")
            return False
        
        print()
        
        # Step 2: Merge with audio
        if not self.merge_with_audio():
            print("❌ Failed to merge with audio")
            return False
        
        print()
        print("🎉 Video processing completed successfully!")
        print(f"📁 Final video: {self.final_video.resolve()}")
        return True

    def cleanup(self) -> None:
        """Clean up temporary files."""
        temp_files = [
            self.output_dir / "video_list.txt"
        ]
        
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)


def main() -> int:
    """Main entry point."""
    combiner = VideoCombiner()
    
    try:
        success = combiner.process_all()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1
    finally:
        combiner.cleanup()


if __name__ == "__main__":
    sys.exit(main())
