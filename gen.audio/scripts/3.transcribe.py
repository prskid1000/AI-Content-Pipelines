import os
import re
import whisper
import time
import math
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

 

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def get_silence_text(duration):
    """
    Generate silence text with dots based on duration.
    Add 1 dot for every 1 second of silence for clean readability.
    """
    # Calculate number of dots: 1 dot per second, minimum 1 dot
    num_dots = max(1, math.ceil(duration))
    
    # Build the silence text using a for loop
    silence_text = ""
    for i in range(num_dots):
        silence_text += "."
    
    return silence_text


def post_process_segments(segments, audio_file_path):
    """
    Post-process segments to make timeline continuous by adding silent segments.
    Creates a continuous timeline while preserving original segment timing.
    Only silence gaps are inserted, segments keep their original start/end times.
    """
    if not segments:
        return segments
    
    # Always get actual audio file duration
    import wave
    with wave.open(audio_file_path, 'rb') as w:
        frames = w.getnframes()
        rate = w.getframerate()
        target_duration = frames / float(rate)
    
    print(f"Using actual audio file duration as target: {target_duration:.6f}s")
    
    # Calculate current total duration from segments
    current_duration = segments[-1]["end"]
    missing_duration = target_duration - current_duration
    
    print(f"Current duration: {current_duration:.6f}s")
    print(f"Target duration: {target_duration:.6f}s")
    print(f"Missing duration: {missing_duration:.6f}s")
    
    # Create new continuous timeline
    continuous_segments = []
    
    # First, handle initial gap if first segment doesn't start at 0
    if segments[0]["start"] > 0.000001:  # Microsecond precision (1 μs)
        initial_gap = segments[0]["start"]
        silence_text = get_silence_text(initial_gap)
        continuous_segments.append({
            "start": 0.0,
            "end": initial_gap,
            "text": silence_text
        })
        print(f"Added initial silence: {initial_gap:.6f}s ({silence_text})")
    
    # Process all segments and fill gaps
    for i, segment in enumerate(segments):
        # Handle gap between this segment and previous
        if i > 0:
            gap = segment["start"] - segments[i-1]["end"]
            if gap > 0.000001:  # Microsecond precision (1 μs)
                if gap <= 1.0:  # Small gap: extend adjacent segments
                    extension = gap / 2.0
                    continuous_segments[-1]["end"] += extension
                    segment["start"] -= extension
                    print(f"Extended segments to fill {gap:.6f}s gap")
                else:  # Large gap: add silence segment
                    silence_text = get_silence_text(gap)
                    continuous_segments.append({
                        "start": segments[i-1]["end"],  # Original end time of previous segment
                        "end": segment["start"],        # Original start time of current segment
                        "text": silence_text
                    })
                    print(f"Added silence gap: {gap:.6f}s ({silence_text})")
        
        # Add the actual segment with ORIGINAL timing (preserves audio sync)
        continuous_segments.append({
            "start": segment["start"],  # Keep original start time
            "end": segment["end"],      # Keep original end time
            "text": segment["text"]
        })
    
    # Add final silence to reach target duration
    final_silence = target_duration - segments[-1]["end"]
    if final_silence > 0.000001:  # Microsecond precision (1 μs)
        silence_text = get_silence_text(final_silence)
        continuous_segments.append({
            "start": segments[-1]["end"],  # Original end time of last segment
            "end": target_duration,
            "text": silence_text
        })
        print(f"Added final silence: {final_silence:.6f}s ({silence_text})")
    
    # Verify final duration
    final_duration = continuous_segments[-1]["end"]
    print(f"Final continuous duration: {final_duration:.6f}s")
    print(f"Total segments (including silence): {len(continuous_segments)}")
    
    return continuous_segments

def generate_files(segments, srt_file, text_file, timeline_file):
    """Generate SRT, text, and timeline files from segments"""
    
    # Generate SRT content
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = re.sub(r'\s+', ' ', segment["text"].strip())
        srt_content += f"{start_time} --> {end_time}\n{text}\n\n"
    
    # Generate text content (excluding silence markers)
    text_content = ""
    for segment in segments:
        text = re.sub(r'\s+', ' ', segment["text"].strip())
        # Skip segments that are just dots (silence markers)
        if not re.match(r'^\.+$', text):  # Only dots from start to end
            text_content += text + " "
    text_content = text_content.strip()
    
    # Generate timeline content
    timeline_content = ""
    total_duration = 0
    for segment in segments:
        duration = segment["end"] - segment["start"]
        total_duration += duration
        text = re.sub(r'\s+', ' ', segment["text"].strip())
        timeline_content += f"{duration:.6f}: {text}\n"
    
    # Write files
    with open(srt_file, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text_content)
    
    with open(timeline_file, 'w', encoding='utf-8') as f:
        f.write(timeline_content)
    
    print(f"SRT file saved to: {srt_file}")
    print(f"Text file saved to: {text_file}")
    print(f"Timeline file saved to: {timeline_file}")
    
    return total_duration

def transcribe_audio(audio_path, srt_file, text_file, timeline_file, model_name="large"):
    """Transcribe audio and generate all output files"""
    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        
        print(f"Transcribing audio file: {audio_path}")
        result = model.transcribe(
            audio_path, 
            verbose=True,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        
        segment_count = len(result['segments'])
        print(f"Original segments: {segment_count}")
        
        # Post-process segments to make timeline continuous
        print("\nPost-processing segments...")
        processed_segments = post_process_segments(result["segments"], audio_file_path=audio_path)
        
        # Generate all files
        total_duration = generate_files(processed_segments, srt_file, text_file, timeline_file)
        return True, total_duration, segment_count
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return False, 0, 0



def main():
    """Main function to transcribe story.wav and generate three output files"""
    start_time = time.time()
    
    audio_file = "../output/story.wav"
    srt_file = "../input/2.story.srt"
    text_file = "../input/2.story.str.txt"
    timeline_file = "../input/2.timeline.txt"
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found!")
        return
    
    print("Starting audio transcription with OpenAI Whisper...")
    print("=" * 50)
    
    # Time the transcription process
    transcription_start = time.time()
    result = transcribe_audio(audio_file, srt_file, text_file, timeline_file)
    transcription_time = time.time() - transcription_start
    
    if result[0]:  # success
        total_duration = result[1]
        segment_count = result[2]
        print("\nTranscription completed successfully!")
        print(f"\n✅ Process completed! Files generated:")
        print(f"  • {srt_file} (Original SRT format)")
        print(f"  • {text_file} (Plain text format)")
        print(f"  • {timeline_file} (Duration computed format)")
        
        # Display total duration
        minutes = int(total_duration // 60)
        seconds = total_duration % 60
        print(f"\n📊 TRANSCRIPTION SUMMARY:")
        print(f"  • Total audio duration: {total_duration:.3f} seconds ({minutes}m {seconds:.3f}s)")
        print(f"  • Number of segments: {segment_count}")
        
        print(f"\n💡 To analyze transcription quality, run: python quality.py")
    else:
        print("\nTranscription failed!")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print detailed timing information
    print("\n" + "=" * 50)
    print("⏱️  TIMING SUMMARY")
    print("=" * 50)
    print(f"📝 Transcription time: {transcription_time:.3f} seconds")
    print(f"⏱️  Total execution time: {total_time:.3f} seconds ({total_time/60:.3f} minutes)")
    print("=" * 50)

if __name__ == "__main__":
    main()
