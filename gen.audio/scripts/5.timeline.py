import requests
import json
import time
import os
import re
from typing import List, Dict, Any
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

# Model constants for easy switching
MODEL_TIMELINE_GENERATION = "qwen/qwen3-14b"  # Model for timeline SFX generation

class TimelineSFXGenerator:
    def __init__(self, lm_studio_url="http://localhost:1234/v1", model=MODEL_TIMELINE_GENERATION, use_json_schema=True):
        self.lm_studio_url = lm_studio_url
        self.output_file = "../input/3.timing.txt"
        self.model = model
        self.use_json_schema = use_json_schema
        
    def read_timeline_content(self, filename="../input/2.timeline.txt") -> str:
        """Read timeline content from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Timeline file '{filename}' not found.")
            return None
        except Exception as e:
            print(f"Error reading timeline file: {e}")
            return None
    
    def parse_timeline_entries(self, content: str) -> List[Dict[str, Any]]:
        """Parse timeline content into structured entries"""
        entries = []
        for line in content.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    try:
                        seconds = float(parts[0].strip())
                        description = parts[1].strip()
                        entries.append({
                            'seconds': seconds,
                            'description': description,
                            'original_line': line
                        })
                    except ValueError:
                        print(f"Warning: Invalid duration format in line: {line}")
                        continue
        
        print(f"📋 Parsed {len(entries)} timeline entries")
        return entries
    
    def create_prompt_for_single_entry(self, entry: Dict[str, Any]) -> str:
        """Create the prompt for a single timeline entry"""
        prompt = f"""CONTENT:{entry['seconds']} seconds: {entry['description']}"""
        return prompt

    def _is_dot_only_or_short(self, text: str) -> bool:
        """Return True if text is comprised only of dots or has <= 3 words."""
        if text is None:
            return True
        stripped = text.strip()
        # Treat unicode ellipsis as dots as well
        normalized = stripped.replace('…', '.')
        # Check if the sentence contains only dots
        if normalized != '' and all(ch == '.' for ch in normalized):
            return True
        # Count word-like tokens (alphanumeric/underscore sequences)
        word_tokens = re.findall(r"\w+", stripped, flags=re.UNICODE)
        return len(word_tokens) <= 3

    def _build_response_format(self) -> Dict[str, Any]:
        """Build a simple JSON Schema response format for single entry output."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "sfx_entry",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sound_or_silence_description": {"type": "string"}
                    },
                    "required": ["sound_or_silence_description"]
                },
                "strict": True
            }
        }
    
    def call_lm_studio_api(self, prompt: str) -> str:
        """Call LM Studio API to generate SFX for a single entry"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": 
"""You are an SFX(Sound or Silence) generator for Sound Generating AI Models.

RULES:
- Keep descriptions under 12 words, concrete, specific, unambiguous, descriptive(pitch, amplitude, timbre, sonance, frequency, etc.) and present tense.
- If no clear Sound related words or an important Action/Object that is producing or can produce sound is present in the transcript line, use 'Silence'; invent nothing yourself.
- No speech, lyrics, music, or vocal sounds allowed;use "Silence". May generate sounds(Diegetic/Non-diegetic) like atmosphere/ambience/background/noise/foley deduced from the transcript line.
- You must output only sound descriptions, any other sensory descriptions like visual, touch, smell, taste, etc. are not allowed;use "Silence".
- Return only JSON matching the schema.

OUTPUT: JSON with sound_or_silence_description field only."""
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n/no_think"
                    }
                ],
                "temperature": 1,
                "max_tokens": 512,
                "stream": False
            }

            # Request structured output
            payload["response_format"] = self._build_response_format()
            
            response = requests.post(
                f"{self.lm_studio_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    return content
                else:
                    raise Exception("No content in API response")
            else:
                raise Exception(f"API call failed with status {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to LM Studio API. Make sure LM Studio is running on localhost:1234")
        except requests.exceptions.Timeout:
            raise Exception("API call timed out")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    def parse_sfx_response(self, response: str) -> str:
        """Parse the SFX response from LM Studio for a single entry"""
        # Try JSON first
        text = response.strip()
        # Remove code fences if present
        if text.startswith("```"):
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
            if m:
                text = m.group(1).strip()
        # Fallback: extract braces region
        if not text.startswith("{"):
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                text = text[first:last+1]
        
        try:
            json_obj = json.loads(text)
            if isinstance(json_obj, dict) and "sound_or_silence_description" in json_obj:
                return json_obj["sound_or_silence_description"]
        except Exception:
            pass
        
        # Fallback: try to extract description from response
        for line in response.strip().split('\n'):
            line = line.strip()
            if ':' in line and '"' in line:
                try:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        content_part = parts[1].strip()
                        if content_part.startswith('"') and content_part.endswith('"'):
                            return content_part[1:-1]  # Remove quotes
                        else:
                            return content_part.strip('"')
                except Exception:
                    continue
        
        # Default fallback
        return "Silence"
    
    def save_sfx_to_file(self, all_sfx_entries: List[Dict[str, Any]]) -> None:
        """Save all SFX entries to sfx.txt"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for entry in all_sfx_entries:
                    f.write(f"{entry['seconds']}: {entry['sound_or_silence_description']}\n")
            
            total_duration = sum(entry['seconds'] for entry in all_sfx_entries)
            print(f"💾 Saved {len(all_sfx_entries)} SFX entries to {self.output_file}")
            print(f"⏱️  Total duration: {total_duration:.3f} seconds ({total_duration/60:.2f} minutes)")
            
        except Exception as e:
            raise Exception(f"Failed to save SFX file: {str(e)}")
    
    def process_timeline(self, timeline_filename="../input/2.timeline.txt") -> bool:
        
        
        """Main processing function - process each entry individually"""
        print("🚀 Starting Timeline SFX Generation...")
        print(f"📁 Reading timeline from: {timeline_filename}")
        
        # Read timeline content
        content = self.read_timeline_content(timeline_filename)
        if content is None:
            return False
        
        # Parse timeline entries
        entries = self.parse_timeline_entries(content)
        if not entries:
            print("❌ No valid timeline entries found")
            return False
        
        # Process each entry individually
        all_sfx_entries = []
        
        for i, entry in enumerate(entries):
            entry_start_time = time.time()
            print(f"\n📝 Processing entry {i+1}/{len(entries)}: {entry['seconds']}s - {entry['description'][:50]}...")
            
            # Create prompt for this single entry
            prompt = self.create_prompt_for_single_entry(entry)
            
            # Rule: mark as Silence without calling LM Studio if only dots or <=3 words
            if self._is_dot_only_or_short(entry['description']):
                sfx_entry = {
                    'seconds': entry['seconds'],
                    'sound_or_silence_description': 'Silence'
                }
                all_sfx_entries.append(sfx_entry)
                print(f"🔕 Skipped LM Studio (rule matched: dot-only or ≤3 words) → {entry['seconds']}: Silence")
                entry_end_time = time.time()
                entry_duration = entry_end_time - entry_start_time
                print(f"✅ Entry {i+1} processed successfully in {entry_duration:.2f} seconds")
                if i < len(entries) - 1:
                    time.sleep(1)
                continue
            
            try:
                # Call LM Studio API
                response = self.call_lm_studio_api(prompt)
                
                # Parse SFX response
                sound_description = self.parse_sfx_response(response)
                
                # Create output entry with original duration
                sfx_entry = {
                    'seconds': entry['seconds'],
                    'sound_or_silence_description': sound_description
                }
                
                # Add to all entries
                all_sfx_entries.append(sfx_entry)
                
                # Live preview for this entry
                print(f"🎵 Output: {entry['seconds']}: {sound_description}")

                entry_end_time = time.time()
                entry_duration = entry_end_time - entry_start_time
                print(f"✅ Entry {i+1} processed successfully in {entry_duration:.2f} seconds")
                
            except Exception as e:
                entry_end_time = time.time()
                entry_duration = entry_end_time - entry_start_time
                print(f"❌ Error processing entry {i+1}: {str(e)} (took {entry_duration:.2f} seconds)")
                # Continue with next entry instead of failing completely
                all_sfx_entries.append({
                    'seconds': entry['seconds'],
                    'sound_or_silence_description': 'Silence'
                })
            
            # Small delay between API calls
            if i < len(entries) - 1:
                time.sleep(1)
        
        # Save all SFX entries to file
        try:
            self.save_sfx_to_file(all_sfx_entries)
            print(f"\n🎉 Timeline SFX generation completed successfully!")
            print(f"📄 Output saved to: {self.output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving SFX file: {str(e)}")
            return False

def main():
    """Main function"""
    import sys
    
    # Check command line arguments
    timeline_file = "../input/2.timeline.txt"
    if len(sys.argv) > 1:
        timeline_file = sys.argv[1]
    
    # Check if timeline file exists
    if not os.path.exists(timeline_file):
        print(f"❌ Timeline file '{timeline_file}' not found")
        print("Usage: python 5.timeline.py [timeline_file]")
        return 1
    
    # Create generator and process
    generator = TimelineSFXGenerator()
    
    start_time = time.time()
    success = generator.process_timeline(timeline_file)
    end_time = time.time()
    
    if success:
        print(f"⏱️  Total processing time: {end_time - start_time:.2f} seconds")
        return 0
    else:
        print("❌ Processing failed")
        return 1

if __name__ == "__main__":
    exit(main())
