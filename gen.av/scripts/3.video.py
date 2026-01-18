#!/usr/bin/env python3
"""
AV Video Combiner Script

This script combines all videos from the AV output folder into a single final video.
Since AV videos already have built-in audio, no audio merging is needed.

Features:
- Combines all MP4 videos from ../output directory (scene_*.mp4 files)
- Sorts videos by scene ID for proper chronological order
- Optionally adds thumbnail intro if available
- Outputs final video to ../output/final.mp4
- Uses FFmpeg for video processing
"""

import os
import sys
import subprocess
import re
import shutil
from pathlib import Path
from typing import List, Optional


def find_thumbnail(output_dir: str) -> Optional[str]:
    """Find a thumbnail image in the output directory.
    
    Prefers PNG but falls back to common formats.
    """
    candidates = [
        os.path.join(output_dir, "thumbnail.png"),
        os.path.join(output_dir, "thumbnail.jpg"),
        os.path.join(output_dir, "thumbnail.jpeg"),
        os.path.join(output_dir, "thumbnail.webp"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class AVVideoCombiner:
    def __init__(self):
        # Directory paths relative to script location
        self.output_dir = Path("../output")
        self.thumbnail_dir = Path("../../gen.audio/output")  # Check audio folder for thumbnail
        
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

    def _create_thumbnail_video(self, image_path: str, duration: float, output_path: Path) -> bool:
        """Create a video from a still image using ffmpeg."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            print("❌ ffmpeg is not available in PATH. Please install ffmpeg.")
            return False

        cmd = [
            ffmpeg_path,
            "-y",
            "-loop", "1",
            "-i", image_path,
            "-t", f"{duration:.3f}",
            "-vf", "format=yuv420p",
            "-c:v", "libx264",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ ffmpeg failed for thumbnail: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("❌ ffmpeg timed out while creating thumbnail video")
            return False

    def _get_video_files(self) -> List[Path]:
        """Get all MP4 video files from the output directory, sorted by scene ID."""
        if not self.output_dir.exists():
            print(f"ERROR: Output directory not found: {self.output_dir}")
            return []
        
        video_files = []
        # Look for scene_*.mp4 files
        for file in self.output_dir.glob("scene_*.mp4"):
            video_files.append(file)
        
        if not video_files:
            print(f"ERROR: No scene_*.mp4 files found in {self.output_dir}")
            return []
        
        # Sort by scene ID (extract numeric parts for proper ordering)
        def extract_scene_key(filepath: Path) -> tuple:
            """Extract sorting key from scene filename like 'scene_1.1.mp4'."""
            name = filepath.stem
            # Remove "scene_" prefix
            if name.startswith("scene_"):
                name = name[6:]  # Remove "scene_" (6 chars)
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
                    # Replace backslashes with forward slashes for Windows compatibility
                    abs_path_str = str(abs_path).replace('\\', '/')
                    f.write(f"file '{abs_path_str}'\n")
            
            print(f"Created file list: {file_list_path}")
            return file_list_path
            
        except Exception as e:
            print(f"ERROR: Failed to create file list: {e}")
            raise

    def combine_videos(self) -> bool:
        """Combine all AV videos into a single video."""
        
        print("=" * 60)
        print("STEP 1: Combining AV videos")
        print("=" * 60)
        
        # Check FFmpeg
        if not self._check_ffmpeg():
            print("ERROR: FFmpeg not found. Please install FFmpeg and add it to your PATH.")
            return False
        
        # Get video files
        video_files = self._get_video_files()
        if not video_files:
            return False
        
        # Check for thumbnail and prepend if found
        thumbnail_path = find_thumbnail(str(self.thumbnail_dir))
        if thumbnail_path:
            print(f"\n🖼️  Found thumbnail: {os.path.basename(thumbnail_path)}")
            thumbnail_video_path = self.output_dir / "thumbnail_intro.mp4"
            thumbnail_duration = 1.0  # 1 second thumbnail intro
            
            print(f"🎬 Creating {thumbnail_duration}s thumbnail intro video...")
            thumbnail_ok = self._create_thumbnail_video(
                image_path=thumbnail_path,
                duration=thumbnail_duration,
                output_path=thumbnail_video_path
            )
            
            if thumbnail_ok:
                # Prepend thumbnail video to the beginning
                video_files.insert(0, thumbnail_video_path)
                print(f"✅ Thumbnail intro added to sequence")
            else:
                print(f"⚠️  Failed to create thumbnail video, continuing without it")
        else:
            print(f"\n⚠️  No thumbnail found in {self.thumbnail_dir}")
        
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
            '-c', 'copy',  # Copy streams without re-encoding for speed (preserves audio)
            '-y',  # Overwrite output file
            str(self.final_video)
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✓ Successfully combined videos: {self.final_video}")
                # Clean up temporary file list
                file_list_path.unlink(missing_ok=True)
                
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
            print("ERROR: FFmpeg timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"ERROR: Failed to run FFmpeg: {e}")
            return False

    def process_all(self) -> bool:
        """Run the complete video processing pipeline."""
        print("🎬 AV Video Processing Pipeline")
        print("=" * 60)
        
        # Combine videos (audio is already included in AV videos)
        if not self.combine_videos():
            print("❌ Failed to combine videos")
            return False
        
        print()
        print("🎉 Video processing completed successfully!")
        print(f"📁 Final video: {self.final_video.resolve()}")
        return True

    def cleanup(self) -> None:
        """Clean up temporary files."""
        temp_files = [
            self.output_dir / "video_list.txt",
            self.output_dir / "thumbnail_intro.mp4",
            self.output_dir / "combined.mp4"  # Remove intermediate combined file if exists
        ]
        
        for temp_file in temp_files:
            temp_file.unlink(missing_ok=True)


def main() -> int:
    """Main entry point."""
    combiner = AVVideoCombiner()
    
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
