import os
import shutil
import re
import glob
import time
import argparse
import requests
import json
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

LANGUAGE = "en"
REGION = "in"

# Model constants for easy switching
MODEL_CHARACTER_CHAPTER_SUMMARY = "qwen/qwen3-14b"  # Model for chapter summarization
MODEL_CHARACTER_TITLE_GENERATION = "qwen/qwen3-14b"  # Model for story title generation
MODEL_CHARACTER_META_SUMMARY = "qwen/qwen3-14b"  # Model for meta-summary generation

# Story processing configuration
CHUNK_SIZE = 50  # Number of lines per chapter chunk for summarization
GENERATE_TITLE = True  # Set to False to disable automatic title generation
ENABLE_THINKING = False  # Set to True to enable thinking in LM Studio responses

# Non-interactive defaults (can be overridden by CLI flags in __main__)
AUTO_GENDER = "m"
AUTO_CONFIRM = "y"
AUTO_CHANGE_SETTINGS = "n"
AUTO_REGION = ""
AUTO_LANGUAGE = ""

def load_available_voices(language=LANGUAGE, region=REGION):
    """
    Automatically load available voices from the voices folder structure
    based on the specified language and region.
    
    Folder structure: voices/{gender}/{region}/{language}/
    Example: voices/male/in/en/ for male English voices from India
    """
    male_voices = []
    female_voices = []
    
    # Define the base path and patterns
    base_path = "../voices"
    
    # Pattern for male voices
    male_pattern = os.path.join(base_path, "male", region, language, "*.wav")
    female_pattern = os.path.join(base_path, "female", region, language, "*.wav")
    
    # Load male voices
    male_files = glob.glob(male_pattern)
    for file_path in male_files:
        # Extract voice name from filename (e.g., "alok_en.wav" -> "alok_en")
        voice_name = os.path.splitext(os.path.basename(file_path))[0]
        male_voices.append(voice_name)
    
    # Load female voices
    female_files = glob.glob(female_pattern)
    for file_path in female_files:
        # Extract voice name from filename (e.g., "aisha_en.wav" -> "aisha_en")
        voice_name = os.path.splitext(os.path.basename(file_path))[0]
        female_voices.append(voice_name)
    
    # Sort voices alphabetically for consistency
    male_voices.sort()
    female_voices.sort()
    
    return male_voices, female_voices

# Load available voices based on current language setting
male_voices, female_voices = load_available_voices(LANGUAGE, REGION)

# Default character voice assignments
character_voices = {
    "male_doctor_watson": "alok_en",
    "male_detective_holmes": "ramesh_en"
}

class CharacterManager:
    def __init__(self, language=LANGUAGE, region=REGION, lm_studio_url="http://localhost:1234/v1", model=MODEL_CHARACTER_CHAPTER_SUMMARY):
        self.language = language
        self.region = region
        self.character_voices = character_voices.copy()
        # Reload voices for the specified language and region
        self.male_voices, self.female_voices = load_available_voices(language, region)
        # LM Studio integration for story summarization
        self.lm_studio_url = lm_studio_url
        self.model = model
    
    def set_language(self, language):
        """Change the language and reload available voices"""
        self.language = language
        self.male_voices, self.female_voices = load_available_voices(language, self.region)
        print(f"Language changed to: {language}")
        print(f"Available male voices: {', '.join(self.male_voices)}")
        print(f"Available female voices: {', '.join(self.female_voices)}")
    
    def set_region(self, region):
        """Change the region and reload available voices"""
        self.region = region
        self.male_voices, self.female_voices = load_available_voices(self.language, region)
        print(f"Region changed to: {region}")
        print(f"Available male voices: {', '.join(self.male_voices)}")
        print(f"Available female voices: {', '.join(self.female_voices)}")
    
    def set_language_and_region(self, language, region):
        """Change both language and region and reload available voices"""
        self.language = language
        self.region = region
        self.male_voices, self.female_voices = load_available_voices(language, region)
        print(f"Language changed to: {language}, Region changed to: {region}")
        print(f"Available male voices: {', '.join(self.male_voices)}")
        print(f"Available female voices: {', '.join(self.female_voices)}")
    
    def get_available_languages(self, region=None):
        """Get list of available languages from the voices folder for a specific region"""
        if region is None:
            region = self.region
            
        languages = set()
        base_path = "../voices"
        
        # Check both male and female folders
        for gender in ["male", "female"]:
            gender_path = os.path.join(base_path, gender, region)
            if os.path.exists(gender_path):
                for item in os.listdir(gender_path):
                    item_path = os.path.join(gender_path, item)
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        languages.add(item)
        
        return sorted(list(languages))
    
    def get_available_regions(self):
        """Get list of available regions from the voices folder"""
        regions = set()
        base_path = "../voices"
        
        # Check both male and female folders
        for gender in ["male", "female"]:
            gender_path = os.path.join(base_path, gender)
            if os.path.exists(gender_path):
                for item in os.listdir(gender_path):
                    item_path = os.path.join(gender_path, item)
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        regions.add(item)
        
        return sorted(list(regions))
    
    def extract_characters_from_story(self, story_text):
        """Extract all unique characters from the story text"""
        # Find all text in square brackets
        character_pattern = r'\[([^\]]+)\]'
        characters = re.findall(character_pattern, story_text)
        
        # Remove duplicates and return unique characters
        unique_characters = list(set(characters))
        return unique_characters
    
    def assign_voices_to_characters(self, characters):
        """Assign voices to characters through user interaction"""
        updated_character_voices = self.character_voices.copy()
        
        print("\n=== CHARACTER VOICE ASSIGNMENT ===")
        print("I found the following characters in your story:")
        
        for char in characters:
            print(f"- {char}")
        
        print(f"\nCurrently assigned voices:")
        for char, voice in updated_character_voices.items():
            print(f"- {char}: {voice}")
        
        # Process characters not already assigned
        unassigned_chars = [char for char in characters if char not in updated_character_voices]
        
        if unassigned_chars:
            print(f"\nNeed to assign voices for: {', '.join(unassigned_chars)}")
            
            for char in unassigned_chars:
                # Check if character name has gender prefix
                if char.lower().startswith('male_'):
                    # Auto-assign male voice - avoid reusing already assigned voices
                    used_male_voices = [v for v in updated_character_voices.values() if v in self.male_voices]
                    available_male_voices = [v for v in self.male_voices if v not in used_male_voices]
                    
                    if available_male_voices:
                        voice = available_male_voices[0]  # Use first available voice
                        updated_character_voices[char] = voice
                        print(f"Auto-assigned male voice '{voice}' to '{char}' (male_ prefix detected)")
                    else:
                        print(f"Warning: No available male voices left for '{char}'. All male voices are already assigned.")
                        # Fallback to cycling through voices
                        male_voice_index = len(used_male_voices) % len(self.male_voices)
                        voice = self.male_voices[male_voice_index]
                        updated_character_voices[char] = voice
                        print(f"Fallback: Reused male voice '{voice}' for '{char}'")
                        
                elif char.lower().startswith('female_'):
                    # Auto-assign female voice - avoid reusing already assigned voices
                    used_female_voices = [v for v in updated_character_voices.values() if v in self.female_voices]
                    available_female_voices = [v for v in self.female_voices if v not in used_female_voices]
                    
                    if available_female_voices:
                        voice = available_female_voices[0]  # Use first available voice
                        updated_character_voices[char] = voice
                        print(f"Auto-assigned female voice '{voice}' to '{char}' (female_ prefix detected)")
                    else:
                        print(f"Warning: No available female voices left for '{char}'. All female voices are already assigned.")
                        # Fallback to cycling through voices
                        female_voice_index = len(used_female_voices) % len(self.female_voices)
                        voice = self.female_voices[female_voice_index]
                        updated_character_voices[char] = voice
                        print(f"Fallback: Reused female voice '{voice}' for '{char}'")
                        
                else:
                    # Ask for gender if no prefix found (supports non-interactive via CLI)
                    while True:
                        auto_gender = (AUTO_GENDER or "").lower().strip()
                        if auto_gender in ["m", "male", "f", "female"]:
                            gender = auto_gender
                            print(f"[AUTO] Using --auto-gender='{auto_gender}' for '{char}'")
                        else:
                            gender = input(f"\nIs '{char}' male or female? (m/f): ").lower().strip()
                        if gender in ['m', 'male']:
                            # Assign a male voice - avoid reusing already assigned voices
                            used_male_voices = [v for v in updated_character_voices.values() if v in self.male_voices]
                            available_male_voices = [v for v in self.male_voices if v not in used_male_voices]
                            
                            if available_male_voices:
                                voice = available_male_voices[0]  # Use first available voice
                                updated_character_voices[char] = voice
                                print(f"Assigned male voice '{voice}' to '{char}'")
                            else:
                                print(f"Warning: No available male voices left for '{char}'. All male voices are already assigned.")
                                # Fallback to cycling through voices
                                male_voice_index = len(used_male_voices) % len(self.male_voices)
                                voice = self.male_voices[male_voice_index]
                                updated_character_voices[char] = voice
                                print(f"Fallback: Reused male voice '{voice}' for '{char}'")
                            break
                        elif gender in ['f', 'female']:
                            # Assign a female voice - avoid reusing already assigned voices
                            used_female_voices = [v for v in updated_character_voices.values() if v in self.female_voices]
                            available_female_voices = [v for v in self.female_voices if v not in used_female_voices]
                            
                            if available_female_voices:
                                voice = available_female_voices[0]  # Use first available voice
                                updated_character_voices[char] = voice
                                print(f"Assigned female voice '{voice}' to '{char}'")
                            else:
                                print(f"Warning: No available female voices left for '{char}'. All female voices are already assigned.")
                                # Fallback to cycling through voices
                                female_voice_index = len(used_female_voices) % len(self.female_voices)
                                voice = self.female_voices[female_voice_index]
                                updated_character_voices[char] = voice
                                print(f"Fallback: Reused female voice '{voice}' for '{char}'")
                            break
                        else:
                            print("Please enter 'm' for male or 'f' for female.")
        else:
            print("\nAll characters already have voice assignments!")
        
        # Show final character->voice mapping
        print(f"\n=== FINAL CHARACTER-VOICE MAPPING ===")
        for char in characters:
            voice = updated_character_voices.get(char, "UNASSIGNED")
            print(f"- {char}: {voice}")
        
        # Ask for confirmation (supports non-interactive via CLI)
        while True:
            auto_confirm = (AUTO_CONFIRM or "").lower().strip()
            if auto_confirm in ["y", "yes", "n", "no"]:
                confirm = auto_confirm
                print(f"[AUTO] Using --auto-confirm='{auto_confirm}'")
            else:
                confirm = input(f"\nDo you accept this voice assignment? (y/n): ").lower().strip()
            if confirm in ['y', 'yes']:
                self.character_voices.update(updated_character_voices)
                print("Voice assignment confirmed!")
                
                # Update the character alias map file
                self.update_character_alias_map_file(updated_character_voices)
                
                return updated_character_voices
            elif confirm in ['n', 'no']:
                print("Exiting program as requested.")
                exit(0)
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def update_character_alias_map_file(self, character_voices_dict):
        """Update the character alias map file with current character-voice mappings"""
        alias_map_path = "../../ComfyUI/custom_nodes/tts_audio_suite/voices_examples/#character_alias_map.txt"
        
        try:
            # Create a backup of the original file
            backup_path = alias_map_path + ".backup"
            if os.path.exists(alias_map_path):
                shutil.copy2(alias_map_path, backup_path)
                print(f"Created backup: {backup_path}")
            
            # Write the updated character-voice mappings
            with open(alias_map_path, 'w') as f:
                f.write("# Character Voice Mapping\n")
                f.write("# Format: character=voice\n\n")
                
                for character, voice in character_voices_dict.items():
                    f.write(f"{character}={voice}\n")
            
            print(f"Updated character alias map file: {alias_map_path}")
            print("Format: character=voice")
            
        except Exception as e:
            print(f"Error updating character alias map file: {e}")
            # Restore backup if update failed
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, alias_map_path)
                print("Restored original file from backup")
    
    def get_character_voices(self):
        """Get the current character voice assignments"""
        return self.character_voices.copy()
    
    def set_character_voices(self, voices_dict):
        """Set character voice assignments"""
        self.character_voices.update(voices_dict)
    
    def _build_chapter_response_format(self) -> dict:
        """Build JSON Schema response format for chapter title and summary generation"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "chapter_analysis",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "short_summary": {"type": "string"}
                    },
                    "required": ["title", "summary", "short_summary"]
                },
                "strict": True
            }
        }

    def _build_story_title_response_format(self) -> dict:
        """Build JSON Schema response format for story title generation"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "story_title",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"}
                    },
                    "required": ["title"]
                },
                "strict": True
            }
        }

    def _build_meta_summary_response_format(self) -> dict:
        """Build JSON Schema response format for meta-summary generation"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "meta_summary",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {"type": "string"}
                    },
                    "required": ["summary"]
                },
                "strict": True
            }
        }

    def _call_lm_studio(self, system_prompt: str, use_structured_output: bool = False, is_title_generation: bool = False, is_meta_summary: bool = False) -> str:
        """Call LM Studio API for text generation with optional structured output"""
        headers = {"Content-Type": "application/json"}
        
        # Select model based on prompt type
        if is_title_generation:
            model = MODEL_CHARACTER_TITLE_GENERATION
        elif is_meta_summary:
            model = MODEL_CHARACTER_META_SUMMARY
        else:
            model = MODEL_CHARACTER_CHAPTER_SUMMARY
            
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
            ],
            "temperature": 1,
            "max_tokens": 2048,
            "stream": False,
        }

        # Add structured output if requested
        if use_structured_output:
            if is_title_generation:
                payload["response_format"] = self._build_story_title_response_format()
                payload["max_tokens"] = 512  # Reduce tokens for title generation
            elif is_meta_summary:
                payload["response_format"] = self._build_meta_summary_response_format()
                payload["max_tokens"] = 2048  # More tokens for comprehensive meta-summary
            else:
                payload["response_format"] = self._build_chapter_response_format()
                payload["max_tokens"] = 1024  # Reduce tokens for structured output

        try:
            resp = requests.post(f"{self.lm_studio_url}/chat/completions", headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"LM Studio API error: {resp.status_code} {resp.text}")
            data = resp.json()
            if not data.get("choices"):
                raise RuntimeError("LM Studio returned no choices")
            content = data["choices"][0]["message"]["content"]
            return content
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Could not connect to LM Studio API. Make sure LM Studio is running on localhost:1234")
        except Exception as e:
            raise RuntimeError(f"LM Studio API call failed: {str(e)}")

    def _sanitize_single_paragraph(self, text: str) -> str:
        """Convert text to a single paragraph format"""
        if not text:
            return ""
        # Strip code fences if any
        if text.startswith("```"):
            m = re.search(r"```(?:[a-zA-Z]+)?\s*([\s\S]*?)\s*```", text)
            if m:
                text = m.group(1)
        # Collapse newlines/tabs and excessive whitespace
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _count_story_characters_excluding_names(self, text: str) -> int:
        """Count characters in story text excluding character names in square brackets"""
        if not text:
            return 0
        # Remove all character names in square brackets [character_name]
        text_without_names = re.sub(r'\[([^\]]+)\]', '', text)
        return len(text_without_names)

    def _calculate_story_statistics(self, story_text: str) -> dict:
        """Calculate total character count and story statistics"""
        total_chars = self._count_story_characters_excluding_names(story_text)
        total_lines = len(story_text.strip().split('\n'))
        total_words = len(story_text.split())
        
        return {
            'total_characters': total_chars,
            'total_lines': total_lines,
            'total_words': total_words
        }

    def _split_story_into_chunks(self, story_text: str, chunk_size: int = 50, total_story_chars: int = None) -> list:
        """Split story into chunks of specified line count with character statistics"""
        lines = story_text.strip().split('\n')
        chunks = []
        
        # Calculate total characters if not provided
        if total_story_chars is None:
            total_story_chars = self._count_story_characters_excluding_names(story_text)
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            chunk_chars = self._count_story_characters_excluding_names(chunk_text)
            chunk_percentage = (chunk_chars / total_story_chars * 100) if total_story_chars > 0 else 0
            
            chunks.append({
                'part_number': (i // chunk_size) + 1,
                'start_line': i + 1,
                'end_line': min(i + chunk_size, len(lines)),
                'text': chunk_text,
                'line_count': len(chunk_lines),
                'character_count': chunk_chars,
                'percentage': chunk_percentage
            })
        
        return chunks

    def _build_chapter_summary_prompt(self, chunk_text: str, part_number: int, total_parts: int, percentage: float) -> str:
        """Build system prompt for chapter title and summary generation with structured JSON output"""
        thinking_suffix = "" if ENABLE_THINKING else " /no_think"
        return f"""You are a literary analyst creating chapter titles and summaries. This is Part {part_number} of {total_parts} ({percentage:.1f}% of the total story).

Analyze the story section and generate:
1. A compelling chapter title (2-4 words, captures the essence/theme of this section)
2. A comprehensive summary (exactly 500 words, single paragraph)
3. A short summary (exactly 20 words, captures the key essence, should start with an emoji, engaging and spoiler-light)

REQUIREMENTS:
- Chapter title should reflect the main theme, conflict, or development in this section
- Summary: exactly 500 words (±10 words acceptable), single paragraph format
- Short summary: exactly 20 words (±2 words acceptable), concise key points only, should start with an emoji, engaging and spoiler-light
- Capture key plot points, character interactions, and important dialogue
- Maintain narrative flow and story continuity
- Include character names and their actions
- Preserve the story's tone and style

STORY PART {part_number}/{total_parts} ({percentage:.1f}%):
{chunk_text}

Generate a JSON response with "title", "summary", and "short_summary" fields. Both summaries should be single continuous paragraphs.{thinking_suffix}"""

    def _build_meta_summary_prompt(self, summary_parts: list) -> str:
        """Build system prompt for meta-summarization of all parts"""
        combined_summaries = "\n\n".join([f"PART {i+1}: {summary}" for i, summary in enumerate(summary_parts)])
        thinking_suffix = "" if ENABLE_THINKING else " /no_think"
        
        return f"""You are a master literary summarizer creating a comprehensive story overview. You have been given {len(summary_parts)} individual part summaries of a complete story. Your task is to synthesize these into one cohesive, comprehensive summary.

REQUIREMENTS:
- Exactly 1000-1200 words
- Single continuous paragraph (no line breaks)
- Synthesize all parts into a flowing narrative
- Maintain chronological story progression
- Include all major characters and their development
- Capture key plot points, conflicts, and resolutions
- Preserve the story's tone, style, and themes
- Eliminate redundancy between parts while maintaining completeness
- Create smooth transitions between story segments

INDIVIDUAL PART SUMMARIES:
{combined_summaries}

Create a masterful synthesis that reads as a single, comprehensive story summary rather than separate parts stitched together. Focus on narrative flow and character arcs across the entire story. Generate a JSON response with a "summary" field containing your comprehensive synthesis.{thinking_suffix}"""

    def _generate_meta_summary(self, summary_parts: list) -> str:
        """Generate a meta-summary by re-summarizing all part summaries through LM Studio"""
        print(f"🔄 Re-summarizing {len(summary_parts)} part summaries into comprehensive overview...")
        
        # Build the meta-summary prompt
        system_prompt = self._build_meta_summary_prompt(summary_parts)
        
        # Generate meta-summary with structured output
        start_time = time.time()
        raw_meta_summary = self._call_lm_studio(system_prompt, use_structured_output=True, is_meta_summary=True)
        generation_time = time.time() - start_time
        
        # Parse structured JSON response
        meta_summary = self._parse_meta_summary_response(raw_meta_summary)
        
        # Validate length requirements
        word_count = len(meta_summary.split())
        char_count = len(meta_summary)
        
        print(f"📝 Meta-summary statistics:")
        print(f"   🎯 Words: {word_count} (target: 1000-1200)")
        print(f"   📏 Characters: {char_count}")
        print(f"   ⏱️  Generation time: {generation_time:.2f}s")
        
        # Check if within target range
        if 950 <= word_count <= 1250:
            print("✅ Meta-summary meets word count target")
        else:
            print(f"⚠️  Meta-summary word count outside target range (1000-1200)")
        
        return meta_summary

    def _build_story_title_prompt(self, story_summary: str) -> str:
        """Build system prompt for story title generation"""
        thinking_suffix = "" if ENABLE_THINKING else " /no_think"
        return f"""You are a creative title generator specializing in compelling story titles. Based on the comprehensive story summary provided, generate a captivating and memorable title that captures the essence, theme, and intrigue of the story.

REQUIREMENTS:
- Create a title that is engaging and memorable
- Should be 3-8 words long
- Capture the main theme, genre, or central conflict
- Avoid spoilers while being intriguing
- Consider the story's tone, setting, and main characters
- Make it suitable for audiobook/story content
- Should work well for YouTube/social media sharing
- Consider classic literature style if applicable

EXAMPLES OF GOOD TITLES:
- "The Adventure of the Copper Beeches"
- "Murder on the Orient Express" 
- "The Case of the Missing Heir"
- "Shadows in Baker Street"
- "The Mystery of Thornfield Manor"

STORY SUMMARY:
{story_summary}

Generate a JSON response with a "title" field containing your suggested story title. Focus on creating something that would intrigue potential listeners and capture the story's essence without giving away the plot.{thinking_suffix}"""

    def generate_story_title(self, story_summary: str, output_dir: str = "../input") -> str:
        """Generate a story title based on the comprehensive summary and save to 10.title.txt"""
        if not GENERATE_TITLE:
            print("📝 Story title generation disabled (GENERATE_TITLE = False)")
            return None
        
        print("\n=== GENERATING STORY TITLE ===")
        
        try:
            # Build title generation prompt
            system_prompt = self._build_story_title_prompt(story_summary)
            
            # Generate title with structured output
            start_time = time.time()
            raw_response = self._call_lm_studio(system_prompt, use_structured_output=True, is_title_generation=True)
            generation_time = time.time() - start_time
            
            # Parse structured JSON response
            story_title = self._parse_story_title_response(raw_response)
            
            print(f"✅ Story title generated:")
            print(f"   📖 Title: {story_title}")
            print(f"   ⏱️  Generation time: {generation_time:.2f}s")
            
            # Save title to 10.title.txt
            self._save_story_title(story_title, output_dir)
            
            return story_title
            
        except Exception as e:
            print(f"❌ Error generating story title: {e}")
            return None

    def _parse_story_title_response(self, raw_response: str) -> str:
        """Parse the structured JSON title response to extract the title"""
        try:
            # Clean up response text (remove code fences if present)
            text = raw_response.strip()
            if text.startswith("```"):
                m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
                if m:
                    text = m.group(1).strip()
            
            # Fallback: extract braces region if not starting with {
            if not text.startswith("{"):
                first = text.find("{")
                last = text.rfind("}")
                if first != -1 and last != -1 and last > first:
                    text = text[first:last+1]
            
            # Parse JSON
            json_obj = json.loads(text)
            
            if isinstance(json_obj, dict):
                title = json_obj.get("title", "Untitled Story").strip()
                
                # Clean up title (remove quotes, extra whitespace)
                title = re.sub(r'^["\']|["\']$', '', title).strip()
                
                return title
            else:
                raise ValueError("JSON response is not a dictionary")
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error for title: {e}")
            print(f"Raw response: {raw_response[:200]}...")
            # Fallback to extracting any title-like text
            return self._parse_fallback_title_response(raw_response)
        except Exception as e:
            print(f"⚠️  Error parsing story title response: {e}")
            # Fallback to extracting any title-like text
            return self._parse_fallback_title_response(raw_response)

    def _parse_fallback_title_response(self, raw_response: str) -> str:
        """Fallback parser for non-JSON title responses"""
        try:
            # Look for TITLE: pattern or just use the response as title
            title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', raw_response, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                # Use the first line or entire response if short
                lines = raw_response.strip().split('\n')
                title = lines[0].strip() if lines else "Untitled Story"
            
            # Clean up title (remove quotes, extra whitespace)
            title = re.sub(r'^["\']|["\']$', '', title).strip()
            
            # Limit title length
            if len(title) > 100:
                title = title[:97] + "..."
            
            return title if title else "Untitled Story"
            
        except Exception as e:
            print(f"⚠️  Error in fallback title parsing: {e}")
            return "Untitled Story"

    def _parse_meta_summary_response(self, raw_response: str) -> str:
        """Parse the structured JSON meta-summary response to extract the summary"""
        try:
            # Clean up response text (remove code fences if present)
            text = raw_response.strip()
            if text.startswith("```"):
                m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
                if m:
                    text = m.group(1).strip()
            
            # Fallback: extract braces region if not starting with {
            if not text.startswith("{"):
                first = text.find("{")
                last = text.rfind("}")
                if first != -1 and last != -1 and last > first:
                    text = text[first:last+1]
            
            # Parse JSON
            json_obj = json.loads(text)
            
            if isinstance(json_obj, dict):
                summary = json_obj.get("summary", "").strip()
                
                # Sanitize to single paragraph
                summary = self._sanitize_single_paragraph(summary)
                
                return summary
            else:
                raise ValueError("JSON response is not a dictionary")
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error for meta-summary: {e}")
            print(f"Raw response: {raw_response[:200]}...")
            # Fallback to treating response as summary
            return self._sanitize_single_paragraph(raw_response)
        except Exception as e:
            print(f"⚠️  Error parsing meta-summary response: {e}")
            # Fallback to treating response as summary
            return self._sanitize_single_paragraph(raw_response)

    def _save_story_title(self, title: str, output_dir: str):
        """Save the generated story title to 10.title.txt"""
        output_path = os.path.join(output_dir, "10.title.txt")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(title)
            
            print(f"📄 Story title saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving story title file: {e}")

    def generate_chapter_summaries(self, story_text: str, output_dir: str = "../input") -> list:
        """Generate chapter titles and summaries for each chunk of the story (configurable via CHUNK_SIZE)"""
        print("\n=== CHAPTER ANALYSIS AND SUMMARIZATION ===")
        
        # Calculate story statistics
        story_stats = self._calculate_story_statistics(story_text)
        print(f"📊 Story Statistics:")
        print(f"   📏 Total characters (excluding [names]): {story_stats['total_characters']:,}")
        print(f"   📝 Total words: {story_stats['total_words']:,}")
        print(f"   📄 Total lines: {story_stats['total_lines']:,}")
        
        # Split story into chunks with character statistics
        chunks = self._split_story_into_chunks(story_text, CHUNK_SIZE, story_stats['total_characters'])
        total_parts = len(chunks)
        
        print(f"\nStory split into {total_parts} chapters of ~{CHUNK_SIZE} lines each")
        
        chapters = []
        chapter_data = []
        
        for i, chunk in enumerate(chunks):
            print(f"\nProcessing Chapter {chunk['part_number']}/{total_parts} ({chunk['percentage']:.1f}% of story)...")
            print(f"   Lines {chunk['start_line']}-{chunk['end_line']} ({chunk['character_count']:,} chars)")
            
            try:
                # Build chapter prompt
                system_prompt = self._build_chapter_summary_prompt(
                    chunk['text'], 
                    chunk['part_number'], 
                    total_parts, 
                    chunk['percentage']
                )
                
                # Generate chapter title and summary with structured output
                start_time = time.time()
                raw_response = self._call_lm_studio(system_prompt, use_structured_output=True)
                generation_time = time.time() - start_time
                
                # Parse structured JSON response
                chapter_title, chapter_summary, short_summary = self._parse_structured_chapter_response(raw_response)
                
                # Validate length requirements
                word_count = len(chapter_summary.split()) if chapter_summary else 0
                char_count = len(chapter_summary) if chapter_summary else 0
                short_word_count = len(short_summary.split()) if short_summary else 0
                
                print(f"✅ Chapter {chunk['part_number']} generated:")
                print(f"   📖 Title: {chapter_title}")
                print(f"   📝 Summary words: {word_count} (target: 500)")
                print(f"   📏 Summary characters: {char_count}")
                print(f"   🔸 Short summary words: {short_word_count} (target: 20)")
                print(f"   ⏱️  Generation time: {generation_time:.2f}s")
                
                # Store chapter info
                chapter_info = {
                    'part_number': chunk['part_number'],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line'],
                    'percentage': chunk['percentage'],
                    'character_count': chunk['character_count'],
                    'title': chapter_title,
                    'summary': chapter_summary,
                    'short_summary': short_summary,
                    'word_count': word_count,
                    'char_count': char_count,
                    'short_word_count': short_word_count,
                    'generation_time': generation_time
                }
                chapters.append(chapter_info)
                chapter_data.append(chapter_summary)  # For meta-summary
                
                # Small delay between API calls
                if i < len(chunks) - 1:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Error generating chapter {chunk['part_number']}: {e}")
                # Continue with next part instead of failing completely
                chapters.append({
                    'part_number': chunk['part_number'],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line'],
                    'percentage': chunk['percentage'],
                    'character_count': chunk['character_count'],
                    'error': str(e),
                    'title': None,
                    'summary': None,
                    'short_summary': None
                })
        
        # Save chapters to files
        self._save_chapters_file(chapters, output_dir)
        meta_summary = self._save_merged_summary(chapter_data, output_dir, chapters)
        
        # Generate story title from meta-summary if enabled
        if meta_summary and GENERATE_TITLE:
            try:
                self.generate_story_title(meta_summary, output_dir)
            except Exception as e:
                print(f"⚠️  Title generation failed: {e}")
        
        print(f"\n🎉 Chapter analysis completed!")
        print(f"📊 Successfully generated {len([c for c in chapters if c.get('summary')])} chapters")
        print(f"❌ Failed: {len([c for c in chapters if c.get('error')])} chapters")
        
        return chapters

    def _parse_structured_chapter_response(self, raw_response: str) -> tuple:
        """Parse the structured JSON chapter response to extract title, summary, and short_summary"""
        try:
            # Clean up response text (remove code fences if present)
            text = raw_response.strip()
            if text.startswith("```"):
                m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
                if m:
                    text = m.group(1).strip()
            
            # Fallback: extract braces region if not starting with {
            if not text.startswith("{"):
                first = text.find("{")
                last = text.rfind("}")
                if first != -1 and last != -1 and last > first:
                    text = text[first:last+1]
            
            # Parse JSON
            json_obj = json.loads(text)
            
            if isinstance(json_obj, dict):
                title = json_obj.get("title", "Untitled Chapter").strip()
                summary = json_obj.get("summary", "").strip()
                short_summary = json_obj.get("short_summary", "").strip()
                
                # Clean up title (remove quotes, extra whitespace)
                title = re.sub(r'^["\']|["\']$', '', title).strip()
                
                # Sanitize summaries to single paragraph
                summary = self._sanitize_single_paragraph(summary)
                short_summary = self._sanitize_single_paragraph(short_summary)
                
                return title, summary, short_summary
            else:
                raise ValueError("JSON response is not a dictionary")
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {e}")
            print(f"Raw response: {raw_response[:200]}...")
            # Fallback to old text parsing method
            return self._parse_fallback_chapter_response(raw_response)
        except Exception as e:
            print(f"⚠️  Error parsing structured chapter response: {e}")
            # Fallback to old text parsing method
            return self._parse_fallback_chapter_response(raw_response)

    def _parse_fallback_chapter_response(self, raw_response: str) -> tuple:
        """Fallback parser for non-JSON responses"""
        try:
            # Look for TITLE:, SUMMARY:, and SHORT_SUMMARY: patterns
            title_match = re.search(r'TITLE:\s*(.+?)(?:\n|SUMMARY:|SHORT_SUMMARY:|$)', raw_response, re.IGNORECASE | re.DOTALL)
            summary_match = re.search(r'SUMMARY:\s*(.+?)(?:\n|SHORT_SUMMARY:|$)', raw_response, re.IGNORECASE | re.DOTALL)
            short_summary_match = re.search(r'SHORT_SUMMARY:\s*(.+?)$', raw_response, re.IGNORECASE | re.DOTALL)
            
            title = title_match.group(1).strip() if title_match else "Untitled Chapter"
            summary = summary_match.group(1).strip() if summary_match else ""
            short_summary = short_summary_match.group(1).strip() if short_summary_match else ""
            
            # Clean up title (remove quotes, extra whitespace)
            title = re.sub(r'^["\']|["\']$', '', title).strip()
            
            # Sanitize summaries to single paragraph
            summary = self._sanitize_single_paragraph(summary)
            short_summary = self._sanitize_single_paragraph(short_summary)
            
            if not summary:
                # Last resort: treat entire response as summary
                summary = self._sanitize_single_paragraph(raw_response)
            
            if not short_summary:
                # Generate a basic short summary from the long summary or response
                words = (summary or raw_response).split()[:20]
                short_summary = " ".join(words)
            
            return title, summary, short_summary
            
        except Exception as e:
            print(f"⚠️  Error in fallback parsing: {e}")
            # Ultimate fallback
            summary = self._sanitize_single_paragraph(raw_response)
            words = summary.split()[:20]
            short_summary = " ".join(words)
            return "Untitled Chapter", summary, short_summary

    def _save_chapters_file(self, chapters: list, output_dir: str):
        """Save chapters with percentages and titles to 12.chapters.txt"""
        output_path = os.path.join(output_dir, "12.chapters.txt")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for chapter in chapters:
                    if chapter.get('title') and chapter.get('summary'):
                        # Write the percentage and title line
                        f.write(f"{chapter['percentage']:.0f}%: {chapter['title']}\n")
                        # Write the short summary (20 words)
                        if chapter.get('short_summary'):
                            f.write(f"SHORT_SUMMARY : {chapter['short_summary']}\n")
                    elif chapter.get('error'):
                        # Write error placeholder
                        f.write(f"{chapter['percentage']:.0f}%: [Chapter Generation Failed]\n")
                        f.write(f"Error: {chapter['error']}\n\n")
            
            print(f"📄 Chapters saved to: {output_path}")
            
            # Display chapter breakdown
            self._display_chapter_breakdown(chapters)
            
        except Exception as e:
            print(f"❌ Error saving chapters file: {e}")

    def _display_chapter_breakdown(self, chapters: list):
        """Display a formatted breakdown of chapters with percentages"""
        print("\n📚 CHAPTER BREAKDOWN")
        print("=" * 90)
        print(f"{'%':<4} {'Chapter Title':<30} {'Words':<8} {'Short':<8} {'Status':<8}")
        print("-" * 90)
        
        total_percentage = 0
        successful_chapters = 0
        
        for chapter in chapters:
            percentage = chapter['percentage']
            title = chapter.get('title', '[Failed]')[:28] + ('...' if len(chapter.get('title', '')) > 28 else '')
            word_count = chapter.get('word_count', 0)
            short_word_count = chapter.get('short_word_count', 0)
            status = "✅ OK" if chapter.get('summary') else "❌ FAIL"
            
            total_percentage += percentage
            if chapter.get('summary'):
                successful_chapters += 1
            
            print(f"{percentage:3.0f}% {title:<30} {word_count:<8} {short_word_count:<8} {status:<8}")
        
        print("-" * 90)
        print(f"{'TOT':<4} {f'{successful_chapters}/{len(chapters)} chapters generated':<30} {'-':<8} {'-':<8} {'DONE':<8}")
        print("=" * 90)

    def _save_merged_summary(self, summary_parts: list, output_dir: str, summaries: list):
        """Save all summaries merged into 9.description.txt and display statistics table"""
        if not summary_parts:
            print("⚠️  No summaries to merge - skipping 9.description.txt creation")
            return None
        
        output_path = os.path.join(output_dir, "9.description.txt")
        
        # Generate meta-summary through LM Studio
        print("\n=== GENERATING META-SUMMARY ===")
        try:
            meta_summary = self._generate_meta_summary(summary_parts)
            final_content = meta_summary
            print("✅ Meta-summary generated successfully")
        except Exception as e:
            print(f"❌ Meta-summary generation failed: {e}")
            print("📄 Falling back to concatenated summaries")
            # Fallback to simple concatenation
            merged_content = " ".join(summary_parts)
            final_content = self._sanitize_single_paragraph(merged_content)
        
        # Write to 9.description.txt
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        # Calculate final statistics
        total_words = len(final_content.split())
        total_chars = len(final_content)
        
        print(f"📄 Final summary saved to: {output_path}")
        
        # Display statistics table
        self._display_summary_table(summaries, total_words, total_chars)
        
        # Return the final content for title generation
        return final_content

    def _display_summary_table(self, summaries: list, total_output_words: int, total_output_chars: int):
        """Display a formatted table of input/output statistics"""
        print("\n📊 SUMMARY STATISTICS TABLE")
        print("=" * 90)
        print(f"{'Part':<4} {'Input Lines':<12} {'Input Words':<12} {'Input Chars':<12} {'Output Words':<13} {'Output Chars':<13} {'Status':<8}")
        print("-" * 90)
        
        successful_summaries = [s for s in summaries if s.get('summary')]
        total_input_lines = 0
        total_input_words = 0
        total_input_chars = 0
        
        for summary in summaries:
            part_num = summary['part_number']
            input_lines = summary['end_line'] - summary['start_line'] + 1
            input_words = summary.get('input_words', 0)
            input_chars = summary.get('input_chars', 0)
            
            if summary.get('summary'):
                output_words = summary['word_count']
                output_chars = summary['char_count']
                status = "✅ OK"
            else:
                output_words = 0
                output_chars = 0
                status = "❌ FAIL"
            
            total_input_lines += input_lines
            total_input_words += input_words
            total_input_chars += input_chars
            
            print(f"{part_num:<4} {input_lines:<12} {input_words:<12} {input_chars:<12} {output_words:<13} {output_chars:<13} {status:<8}")
        
        print("-" * 90)
        print(f"{'PARTS':<4} {total_input_lines:<12} {total_input_words:<12} {total_input_chars:<12} {len([s for s in summaries if s.get('summary')]) * 500:<13} {'-':<13} {'PARTS':<8}")
        print(f"{'FINAL':<4} {'-':<12} {'-':<12} {'-':<12} {total_output_words:<13} {total_output_chars:<13} {'META':<8}")
        print("=" * 90)
        
        # Calculate compression ratios
        parts_total_words = len([s for s in summaries if s.get('summary')]) * 500  # Approximate
        
        if total_input_words > 0:
            parts_compression = (parts_total_words / total_input_words) * 100 if parts_total_words > 0 else 0
            final_compression = (total_output_words / total_input_words) * 100
            print(f"📈 Parts Compression: {parts_compression:.1f}% ({parts_total_words}/{total_input_words} words)")
            print(f"🎯 Final Compression: {final_compression:.1f}% ({total_output_words}/{total_input_words} words)")
        
        if total_input_chars > 0:
            char_compression = (total_output_chars / total_input_chars) * 100
            print(f"📏 Character Compression: {char_compression:.1f}% ({total_output_chars}/{total_input_chars} chars)")
        
        print(f"📑 Parts processed: {len(summaries)}")
        print(f"✅ Successful: {len(successful_summaries)}")
        print(f"❌ Failed: {len(summaries) - len(successful_summaries)}")
        
        # Meta-summary info
        if total_output_words >= 1000:
            print(f"🔄 Meta-summary: {total_output_words} words (target: 1000-1200)")

    def preprocess_story(self, story_text):
        """Preprocess the story to identify and assign voices to characters"""
        print("=== STORY PREPROCESSING ===")
        
        # Extract characters from story
        characters = self.extract_characters_from_story(story_text)
        print(f"Found {len(characters)} unique characters: {', '.join(characters)}")
        
        # Show available voices
        print(f"\nAvailable male voices: {', '.join(self.male_voices)}")
        print(f"Available female voices: {', '.join(self.female_voices)}")
        
        # Assign voices to characters
        character_assignments = self.assign_voices_to_characters(characters)
        
        # Generate chapter summaries with titles
        try:
            chapters = self.generate_chapter_summaries(story_text)
            print(f"\n📚 Generated {len([c for c in chapters if c.get('summary')])} chapter summaries")
        except Exception as e:
            print(f"⚠️  Chapter summarization failed: {e}")
            print("Continuing with character voice assignment only...")
        
        return character_assignments

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
    # Parse CLI arguments for non-interactive behavior
    parser = argparse.ArgumentParser(description="Character voice assignment")
    parser.add_argument("--auto-gender", choices=["m", "f", "male", "female"], help="Default gender for characters without prefix")
    parser.add_argument("--auto-confirm", choices=["y", "n", "yes", "no"], help="Auto-accept final voice assignment")
    parser.add_argument("--change-settings", choices=["y", "n", "yes", "no"], help="Whether to change region/language")
    parser.add_argument("--region", help="Region code to use when changing settings")
    parser.add_argument("--language", help="Language code to use when changing settings")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking in LM Studio responses (default: disabled)")
    args = parser.parse_args()

    # Expose as module-level vars for use inside functions
    AUTO_GENDER = (args.auto_gender or None)
    AUTO_CONFIRM = (args.auto_confirm or None)
    AUTO_CHANGE_SETTINGS = (args.change_settings or None)
    AUTO_REGION = (args.region or None)
    AUTO_LANGUAGE = (args.language or None)
    
    # Update ENABLE_THINKING based on CLI argument
    if args.enable_thinking:
        ENABLE_THINKING = True
        print("🧠 Thinking enabled in LM Studio responses")
    else:
        print("🚫 Thinking disabled in LM Studio responses (using /no_think)")

    start_time = time.time()
    
    # Show available regions and languages
    setup_start = time.time()
    character_manager = CharacterManager()
    available_regions = character_manager.get_available_regions()
    available_languages = character_manager.get_available_languages()
    setup_time = time.time() - setup_start
    
    print("=== VOICE REGION AND LANGUAGE SELECTION ===")
    print(f"Available regions: {', '.join(available_regions)}")
    print(f"Available languages: {', '.join(available_languages)}")
    print(f"Current region: {REGION}, Current language: {LANGUAGE}")
    
    # Allow user to change region and language (supports non-interactive via CLI)
    if len(available_regions) > 1 or len(available_languages) > 1:
        while True:
            auto_change = (AUTO_CHANGE_SETTINGS or "").lower().strip()
            if auto_change in ["y", "yes", "n", "no"]:
                change_settings = auto_change
                print(f"[AUTO] Using --change-settings='{auto_change}'")
            else:
                change_settings = input(f"\nDo you want to change region/language settings? (y/n): ").lower().strip()
            if change_settings in ['y', 'yes']:
                # Region selection
                if len(available_regions) > 1:
                    print(f"Available regions: {', '.join(available_regions)}")
                    auto_region = (AUTO_REGION or "").strip()
                    if auto_region:
                        new_region = auto_region
                        print(f"[AUTO] Using --region='{auto_region}'")
                    else:
                        new_region = input(f"Enter region code (e.g., in): ").strip()
                    if new_region in available_regions:
                        character_manager.set_region(new_region)
                        # Update available languages for the new region
                        available_languages = character_manager.get_available_languages(new_region)
                    else:
                        print(f"Invalid region. Please choose from: {', '.join(available_regions)}")
                        continue
                
                # Language selection
                if len(available_languages) > 1:
                    print(f"Available languages for region '{character_manager.region}': {', '.join(available_languages)}")
                    auto_lang = (AUTO_LANGUAGE or "").strip()
                    if auto_lang:
                        new_lang = auto_lang
                        print(f"[AUTO] Using --language='{auto_lang}'")
                    else:
                        new_lang = input(f"Enter language code (e.g., en, hi, ba): ").strip()
                    if new_lang in available_languages:
                        character_manager.set_language(new_lang)
                    else:
                        print(f"Invalid language. Please choose from: {', '.join(available_languages)}")
                        continue
                
                print(f"\nFinal settings - Region: {character_manager.region}, Language: {character_manager.language}")
                break
            elif change_settings in ['n', 'no']:
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    # Read and process story
    story_read_start = time.time()
    story_text = read_story_from_file()
    story_read_time = time.time() - story_read_start
    
    if story_text is None:
        print("Exiting due to story file error.")
        exit(1)
    
    # Time the character preprocessing
    preprocessing_start = time.time()
    character_manager.preprocess_story(story_text)
    preprocessing_time = time.time() - preprocessing_start
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print detailed timing information
    print("\n" + "=" * 50)
    print("⏱️  TIMING SUMMARY")
    print("=" * 50)
    print(f"⚙️  Setup time: {setup_time:.3f} seconds")
    print(f"📖 Story reading time: {story_read_time:.3f} seconds")
    print(f"👥 Character preprocessing time: {preprocessing_time:.3f} seconds")
    print(f"⏱️  Total execution time: {total_time:.3f} seconds ({total_time/60:.3f} minutes)")
    print("=" * 50)
