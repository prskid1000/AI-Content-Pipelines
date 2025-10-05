#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
import json
import re
import time
from datetime import datetime

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
except ImportError as e:
    print(f"Error: Missing required packages. Please install:")
    print(f"pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    print(f"Original error: {e}")
    sys.exit(1)

# YouTube API scopes
SCOPES = ['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube.readonly']

# API service name and version
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def get_authenticated_service():
    """Authenticate and return a YouTube service object."""
    credentials = None
    
    # Check if token.json exists
    if os.path.exists('../token.json'):
        credentials = Credentials.from_authorized_user_file('../token.json', SCOPES)
    
    # If there are no valid credentials, get new ones
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not os.path.exists('../client_secrets.json'):
                print("Error: client_secrets.json not found!")
                print("Please download your OAuth 2.0 credentials from Google Cloud Console")
                print("and save them as 'client_secrets.json' in the same directory as this script.")
                sys.exit(1)
            
            flow = InstalledAppFlow.from_client_secrets_file('../client_secrets.json', SCOPES)
            credentials = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open('../token.json', 'w') as token:
            token.write(credentials.to_json())
    
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=credentials)


def read_description_file():
    """Read description.txt and extract title and description."""
    description_path = Path(__file__).parent.parent / "output" / "description.txt"
    
    if not description_path.exists():
        print(f"Error: {description_path} not found!")
        sys.exit(1)
    
    with open(description_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    if not lines:
        print("Error: description.txt is empty!")
        sys.exit(1)
    
    # First line is the title (preserve emojis and clean up only problematic characters)
    title = lines[0].strip()
    # Remove only control characters and other problematic characters, but keep emojis
    # This preserves Unicode characters including emojis while removing only truly problematic ones
    title = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', title).strip()
    
    # Rest of the content is the description
    description = '\n'.join(lines[1:]).strip()
    
    return title, description

def read_tags_file():
    """Read tags.txt and return as a list."""
    tags_path = Path(__file__).parent.parent / "output" / "tags.txt"
    
    if not tags_path.exists():
        print(f"Warning: {tags_path} not found! No tags will be added.")
        return []
    
    with open(tags_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Split by comma and clean up
    tags = [tag.strip() for tag in content.split(',') if tag.strip()]
    
    # YouTube allows max 15 tags, so limit to first 15
    return tags[:15]

def upload_video(youtube_service, video_file_path, title, description, tags):
    """Upload video to YouTube."""
    # Get today's date in ISO format
    today = datetime.now().strftime('%Y-%m-%d')
    
    body = {
        'snippet': {
            'title': title,
            'description': description,
            'tags': tags,
            'categoryId': '24',  # Entertainment category
            'defaultLanguage': 'en',
            'defaultAudioLanguage': 'en'
        },
        'status': {
            'privacyStatus': 'private',  # Start as private, can be changed later
            'embeddable': True,
            'license': 'creativeCommon',  # Allow remixing with Creative Commons license
            'publicStatsViewable': True,
            'selfDeclaredMadeForKids': False  # Set made for kids to false
        },
        'recordingDetails': {
            'recordingDate': today + 'T00:00:00Z',  # Set recording date to today
        }
    }
    
    # Create media upload object with smaller chunk size for better progress tracking
    # Use 10MB chunks instead of uploading the entire file at once
    media = MediaFileUpload(video_file_path, chunksize=10*1024*1024, resumable=True)
    
    # Call the API's videos.insert method to create and upload the video
    insert_request = youtube_service.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media
    )
    
    print(f"Uploading video: {video_file_path}")
    print(f"Title: {title}")
    print(f"Tags: {', '.join(tags) if tags else 'None'}")
    print("Description preview:", description[:100] + "..." if len(description) > 100 else description)
    print("\nStarting upload...")
    
    try:
        response = None
        last_progress_time = time.time()
        start_time = time.time()
        
        while response is None:
            try:
                status, response = insert_request.next_chunk()
                if status:
                    current_time = time.time()
                    # Show progress every 5 seconds for better feedback
                    if current_time - last_progress_time >= 5:
                        elapsed_time = current_time - start_time
                        progress_percent = int(status.progress() * 100)
                        print(f"Upload progress: {progress_percent}% (elapsed: {elapsed_time:.0f}s)")
                        last_progress_time = current_time
            except Exception as chunk_error:
                print(f"Error during chunk upload: {chunk_error}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
                continue
        
        if 'id' in response:
            video_id = response['id']
            print(f"\n✅ Upload successful!")
            print(f"Video ID: {video_id}")
            print(f"Video URL: https://www.youtube.com/watch?v={video_id}")
            
            return video_id
        else:
            print("❌ Upload failed - no video ID in response")
            return None
            
    except HttpError as e:
        print(f"❌ HTTP Error during upload: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error during upload: {e}")
        return None

def find_shorts_videos(shorts_dir: str) -> list[str]:
    """Find all shorts video files in the specified directory."""
    shorts_videos = []
    for i in range(1, 6):  # shorts.v1.mp4 through shorts.v5.mp4
        video_path = os.path.join(shorts_dir, f"shorts.v{i}.mp4")
        if os.path.exists(video_path):
            shorts_videos.append(video_path)
    return shorts_videos

def create_shorts_title(title: str, shorts_num: int) -> str:
    """Create a title for shorts video."""
    # Add shorts indicator and variation number
    shorts_title = f"{title} #Shorts (Part {shorts_num})"
    return shorts_title

def create_shorts_description(description: str, shorts_num: int) -> str:
    """Create a description for shorts video."""
    shorts_desc = f"🎬 Shorts Version {shorts_num} - {description}\n\n#Shorts #YouTubeShorts"
    return shorts_desc

def upload_shorts_videos(youtube_service, shorts_dir: str, base_title: str, base_description: str, tags: list) -> list[str]:
    """Upload all shorts videos to YouTube."""
    shorts_videos = find_shorts_videos(shorts_dir)
    
    if not shorts_videos:
        print("No shorts videos found to upload.")
        return []
    
    print(f"\nFound {len(shorts_videos)} shorts videos to upload:")
    for video in shorts_videos:
        print(f"  - {os.path.basename(video)}")
    
    uploaded_video_ids = []
    
    for i, video_path in enumerate(shorts_videos, 1):
        print(f"\n{'='*60}")
        print(f"Uploading Shorts Video {i}/{len(shorts_videos)}: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # Create shorts-specific title and description
        shorts_title = create_shorts_title(base_title, i)
        shorts_description = create_shorts_description(base_description, i)
        
        # Add shorts-specific tags
        shorts_tags = tags + ["#Shorts", "#YouTubeShorts", f"Part{i}"]
        
        # Upload the shorts video
        video_id = upload_video(youtube_service, video_path, shorts_title, shorts_description, shorts_tags)
        
        if video_id:
            uploaded_video_ids.append(video_id)
            print(f"✅ Shorts video {i} uploaded successfully!")
        else:
            print(f"❌ Shorts video {i} upload failed!")
    
    return uploaded_video_ids

def main():
    parser = argparse.ArgumentParser(description='Upload video to YouTube')
    parser.add_argument('--video-file', required=True, help='Path to the video file to upload')
    parser.add_argument('--shorts-dir', help='Directory containing shorts videos (default: ../output)')
    parser.add_argument('--upload-shorts', action='store_true', help='Also upload shorts videos')
    parser.add_argument('--shorts-only', action='store_true', help='Upload only shorts videos (skip main video)')
    
    args = parser.parse_args()
    
    # Validate video file exists (unless shorts-only mode)
    if not args.shorts_only:
        video_path = Path(args.video_file)
        if not video_path.exists():
            print(f"Error: Video file '{video_path}' not found!")
            sys.exit(1)
        
        if not video_path.is_file():
            print(f"Error: '{video_path}' is not a file!")
            sys.exit(1)
        
        # Check if it's a video file (basic check)
        video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
        if video_path.suffix.lower() not in video_extensions:
            print(f"Warning: '{video_path.suffix}' might not be a supported video format.")
            print("Supported formats: " + ', '.join(video_extensions))
            
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    try:
        # Read metadata
        print("Reading video metadata...")
        title, description = read_description_file()
        tags = read_tags_file()
        
        print(f"Title: {title}")
        print(f"Description length: {len(description)} characters")
        print(f"Number of tags: {len(tags)}")
        
        # Authenticate
        print("\nAuthenticating with YouTube API...")
        youtube_service = get_authenticated_service()
        
        uploaded_videos = []
        
        # Upload main video (unless shorts-only mode)
        if not args.shorts_only:
            print(f"\n{'='*60}")
            print("Uploading Main Video")
            print(f"{'='*60}")
            
            video_id = upload_video(youtube_service, str(video_path), title, description, tags)
            
            if video_id:
                uploaded_videos.append(video_id)
                print(f"\n🎉 Main video uploaded successfully!")
                print(f"Remember: The video is set to 'private' by default.")
                print(f"You can change the privacy settings in YouTube Studio.")
            else:
                print("\n❌ Main video upload failed!")
                if not args.upload_shorts:
                    sys.exit(1)
        
        # Upload shorts videos (if requested or shorts-only mode)
        if args.upload_shorts or args.shorts_only:
            # Use provided shorts directory or default to ../output
            if args.shorts_dir:
                shorts_dir = args.shorts_dir
            else:
                shorts_dir = str(Path(__file__).parent.parent / "output")
            
            # Validate shorts directory exists
            if not os.path.exists(shorts_dir):
                print(f"Error: Shorts directory '{shorts_dir}' not found!")
                sys.exit(1)
            
            if not os.path.isdir(shorts_dir):
                print(f"Error: '{shorts_dir}' is not a directory!")
                sys.exit(1)
            
            shorts_video_ids = upload_shorts_videos(youtube_service, shorts_dir, title, description, tags)
            uploaded_videos.extend(shorts_video_ids)
            
            if shorts_video_ids:
                print(f"\n🎉 {len(shorts_video_ids)} shorts videos uploaded successfully!")
            else:
                print("\n❌ No shorts videos were uploaded!")
        
        # Summary
        if uploaded_videos:
            print(f"\n{'='*60}")
            print("UPLOAD SUMMARY")
            print(f"{'='*60}")
            print(f"Total videos uploaded: {len(uploaded_videos)}")
            for i, video_id in enumerate(uploaded_videos, 1):
                print(f"  {i}. https://www.youtube.com/watch?v={video_id}")
            print(f"\nAll videos are set to 'private' by default.")
            print(f"You can change privacy settings in YouTube Studio.")
        else:
            print("\n❌ No videos were uploaded successfully!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nUpload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
