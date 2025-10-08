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
from pathlib import Path
print = partial(_builtins.print, flush=True)

LANGUAGE = "en"
REGION = "in"

# Model constants for easy switching
MODEL_CHARACTER_CHAPTER_SUMMARY = "magistral-small-2509"  # Model for chapter summarization
MODEL_CHARACTER_TITLE_GENERATION = "magistral-small-2509"  # Model for story title generation
MODEL_CHARACTER_META_SUMMARY = "magistral-small-2509"  # Model for meta-summary generation
MODEL_DESCRIPTION_GENERATION = "magistral-small-2509"  # Model for description generation

STORY_DESCRIPTION_CHARACTER_MIN = 7200
STORY_DESCRIPTION_CHARACTER_MAX = 9600
STORY_DESCRIPTION_WORD_MIN = 1200
STORY_DESCRIPTION_WORD_MAX = 1600
STORY_DESCRIPTION_PARTS = 5

# Story processing configuration
GENERATE_TITLE = True  # Set to False to disable automatic title generation

# Feature flags for resumable mode
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them

# Non-interactive defaults (can be overridden by CLI flags in __main__)
AUTO_GENDER = "m"
AUTO_CONFIRM = "y"
AUTO_CHANGE_SETTINGS = "n"
AUTO_REGION = ""
AUTO_LANGUAGE = ""

# Resumable state management
class ResumableState:
    """Manages resumable state for expensive LLM operations.
    
    Tracks completion of:
    - Character voice assignments
    - Meta-summary generation (contains all chapter data)
    - Story title generation
    """
    
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
            "character_voices": {"completed": False, "result": None},
            "meta_summary": {"completed": False, "result": None},
            "story_title": {"completed": False, "result": None},
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
    
    def is_character_voices_complete(self) -> bool:
        """Check if character voice assignment is complete."""
        return self.state["character_voices"]["completed"]
    
    def get_character_voices(self) -> dict | None:
        """Get cached character voice assignments if available."""
        return self.state["character_voices"]["result"]
    
    def set_character_voices(self, character_voices: dict):
        """Set character voice assignments and mark as complete."""
        self.state["character_voices"]["completed"] = True
        self.state["character_voices"]["result"] = character_voices
        self._save_state()
    
    def is_meta_summary_complete(self) -> bool:
        """Check if meta-summary generation is complete."""
        return self.state["meta_summary"]["completed"]
    
    def get_meta_summary(self) -> dict | None:
        """Get cached meta-summary (dict with parts) if available."""
        return self.state["meta_summary"]["result"]
    
    def set_meta_summary(self, meta_summary: dict):
        """Set meta-summary (dict with parts) and mark as complete."""
        self.state["meta_summary"]["completed"] = True
        self.state["meta_summary"]["result"] = meta_summary
        self._save_state()
    
    def is_story_title_complete(self) -> bool:
        """Check if story title generation is complete."""
        return self.state["story_title"]["completed"]
    
    def get_story_title(self) -> str | None:
        """Get cached story title if available."""
        return self.state["story_title"]["result"]
    
    def set_story_title(self, story_title: str):
        """Set story title and mark as complete."""
        self.state["story_title"]["completed"] = True
        self.state["story_title"]["result"] = story_title
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
    
    def get_progress_summary(self) -> str:
        """Get a summary of current progress."""
        voices_done = "✓" if self.is_character_voices_complete() else "✗"
        meta_done = "✓" if self.is_meta_summary_complete() else "✗"
        title_done = "✓" if self.is_story_title_complete() else "✗"
        
        return (
            f"Progress: Voices({voices_done}) Meta-Summary({meta_done}) Title({title_done})"
        )

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
        """JSON schema for story description with 5 parts."""
        # Calculate character limits per part (divide total by 5)
        part_min = STORY_DESCRIPTION_CHARACTER_MIN // STORY_DESCRIPTION_PARTS
        part_max = STORY_DESCRIPTION_CHARACTER_MAX // STORY_DESCRIPTION_PARTS
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "story_summary",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "parts": {
                            "type": "array",
                            "minItems": STORY_DESCRIPTION_PARTS,
                            "maxItems": STORY_DESCRIPTION_PARTS,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "title": {
                                        "type": "string",
                                        "minLength": 10,
                                        "maxLength": 100,
                                        "description": "Brief descriptive title for this part of the story"
                                    },
                                    "short_summary": {
                                        "type": "string",
                                        "minLength": 20,
                                        "maxLength": 200,
                                        "description": "Short summary of this part of the story (20-200 characters)"
                                    },
                                    "plot_summary": {
                                        "type": "string",
                                        "minLength": part_min,
                                        "maxLength": part_max,
                                        "description": f"Detailed summary of this part of the story ({part_min}-{part_max} characters)"
                                    }
                                },
                                "required": ["title", "short_summary", "plot_summary"]
                            }
                        }
                    },
                    "required": ["parts"]
                },
                "strict": True
            }
        }

    def _call_lm_studio(self, system_prompt: str, user_prompt: str, use_structured_output: bool = False, is_title_generation: bool = False, is_meta_summary: bool = False) -> str:
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
                {"role": "system", "content": system_prompt + "\nOnly use English Language for Input, Thinking, and Output\n/no_think"},
                {"role": "user", "content": user_prompt + "\nOnly use English Language for Input, Thinking, and Output\n/no_think"},
            ],
            "temperature": 1,
            "stream": False,
        }

        # Add structured output if requested
        if use_structured_output:
            if is_title_generation:
                payload["response_format"] = self._build_story_title_response_format()
            elif is_meta_summary:
                payload["response_format"] = self._build_meta_summary_response_format()

        try:
            resp = requests.post(f"{self.lm_studio_url}/chat/completions", headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"LM Studio API error: {resp.status_code} {resp.text}")
            data = resp.json()
            if not data.get("choices"):
                raise RuntimeError("LM Studio returned no choices")
            content = data["choices"][0]["message"]["content"]
            return content
        except requests.exceptions.Timeout:
            raise RuntimeError("LM Studio API request timed out after 5 minutes")
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


    def _build_meta_summary_system_prompt(self) -> str:
        char_min = STORY_DESCRIPTION_CHARACTER_MIN // STORY_DESCRIPTION_PARTS
        char_max = STORY_DESCRIPTION_CHARACTER_MAX // STORY_DESCRIPTION_PARTS
        word_min = STORY_DESCRIPTION_WORD_MIN // STORY_DESCRIPTION_PARTS
        word_max = STORY_DESCRIPTION_WORD_MAX // STORY_DESCRIPTION_PARTS
        return (
            f"You are a Professional Visual Director and Story Creator and Story Designer and Story Writer and Story Illustrator. Your Job is to Transform the story into 5 distinct parts, each with a title, a short summary, and a detailed summary.\n"
            f"Each part/sub-plot should be {char_min}-{char_max} characters (approximately {word_min}-{word_max} words).\n"
            f"Divide the story chronologically into {STORY_DESCRIPTION_PARTS} meaningful parts/sub-plots. Total across all parts/sub-plots: {STORY_DESCRIPTION_CHARACTER_MIN}-{STORY_DESCRIPTION_CHARACTER_MAX} characters.\n"
            f"Each part/sub-plot should summarize in third person perspective thats includes all characters, locations, and events in details for that section/sub-plot of story.\n"
        )

    def _build_meta_summary_user_prompt(self, story_content: str) -> str:
        """Extract only dialogue lines from story content using existing regex"""
        lines = story_content.split('\n')
        dialogue_lines = []
        
        for line in lines:
            # Remove brackets, braces, and parentheses from the line
            cleaned_line = line.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('(', '').replace(')', '')
            dialogue_lines.append(cleaned_line)
        
        dialogue_content = '\n'.join(dialogue_lines)
        return (
            f"Story content: {dialogue_content}"
        )

    def _generate_meta_summary(self, story_content: str) -> dict:
        """Generate a meta-summary with parts containing title, short_summary, and summary through LM Studio"""
        print(f"🔄 Summarizing story content into {STORY_DESCRIPTION_PARTS} parts...")
        
        # Build the meta-summary prompts
        system_prompt = self._build_meta_summary_system_prompt()
        user_prompt = self._build_meta_summary_user_prompt(story_content)
        
        # Generate meta-summary with structured output
        start_time = time.time()
        raw_meta_summary = self._call_lm_studio(system_prompt, user_prompt, use_structured_output=True, is_meta_summary=True)
        generation_time = time.time() - start_time
        
        # Parse structured JSON response
        meta_summary_data = self._parse_meta_summary_response(raw_meta_summary)
        parts = meta_summary_data.get("parts", [])
        
        # Calculate total statistics
        total_words = 0
        total_chars = 0
        for part in parts:
            summary_text = part.get("plot_summary", "")
            total_words += len(summary_text.split())
            total_chars += len(summary_text)
        
        print(f"📝 Meta-summary statistics:")
        print(f"   🎯 Total Words: {total_words} (target: {STORY_DESCRIPTION_WORD_MIN}-{STORY_DESCRIPTION_WORD_MAX})")
        print(f"   📏 Total Characters: {total_chars} (target: {STORY_DESCRIPTION_CHARACTER_MIN}-{STORY_DESCRIPTION_CHARACTER_MAX})")
        print(f"   📦 Parts: {len(parts)} (expected: {STORY_DESCRIPTION_PARTS})")
        print(f"   ⏱️  Generation time: {generation_time:.2f}s")
        
        # Check if within target range
        if STORY_DESCRIPTION_WORD_MIN <= total_words <= STORY_DESCRIPTION_WORD_MAX:
            print("✅ Meta-summary meets word count target")
        else:
            print(f"⚠️  Meta-summary word count outside target range ({STORY_DESCRIPTION_WORD_MIN}-{STORY_DESCRIPTION_WORD_MAX})")
        
        if len(parts) != STORY_DESCRIPTION_PARTS:
            print(f"⚠️  Expected {STORY_DESCRIPTION_PARTS} parts but got {len(parts)}")
        
        return meta_summary_data

    def _build_story_title_system_prompt(self) -> str:
        """Build system prompt for story title generation"""
        return """You are a creative title generator specializing in compelling story titles. Based on the comprehensive story summary provided, generate a captivating and memorable title that captures the essence, theme, and intrigue of the story.

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

Generate a JSON response with a "title" field containing your suggested story title. Focus on creating something that would intrigue potential listeners and capture the story's essence without giving away the plot."""

    def _build_story_title_user_prompt(self, story_summary: str) -> str:
        """Build user prompt with story data for title generation"""
        return f"""STORY SUMMARY:
{story_summary}"""

    def generate_story_title(self, story_summary: str, output_dir: str = "../input", resumable_state: ResumableState | None = None) -> str:
        """Generate a story title based on the comprehensive summary and save to 10.title.txt"""
        if not GENERATE_TITLE:
            print("📝 Story title generation disabled (GENERATE_TITLE = False)")
            return None
        
        # Check if resumable and story title already complete
        if resumable_state and resumable_state.is_story_title_complete():
            cached_title = resumable_state.get_story_title()
            if cached_title:
                print("\n=== USING CACHED STORY TITLE ===")
                print("Using cached story title from checkpoint")
                print(f"   📖 Title: {cached_title}")
                return cached_title
        
        print("\n=== GENERATING STORY TITLE ===")
        
        try:
            # Build title generation prompts
            system_prompt = self._build_story_title_system_prompt()
            user_prompt = self._build_story_title_user_prompt(story_summary)
            
            # Generate title with structured output
            start_time = time.time()
            raw_response = self._call_lm_studio(system_prompt, user_prompt, use_structured_output=True, is_title_generation=True)
            generation_time = time.time() - start_time
            
            # Parse structured JSON response
            story_title = self._parse_story_title_response(raw_response)
            
            print(f"✅ Story title generated:")
            print(f"   📖 Title: {story_title}")
            print(f"   ⏱️  Generation time: {generation_time:.2f}s")
            
            # Save title to 10.title.txt
            self._save_story_title(story_title, output_dir)
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                resumable_state.set_story_title(story_title)
            
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

    def _parse_meta_summary_response(self, raw_response: str) -> dict:
        """Parse the structured JSON meta-summary response to extract parts with title, short_summary, and summary"""
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
            
            if isinstance(json_obj, dict) and "parts" in json_obj:
                parts = json_obj.get("parts", [])
                
                # Validate that we have the expected structure
                if not isinstance(parts, list) or len(parts) == 0:
                    raise ValueError("Parts array is empty or invalid")
                
                # Sanitize each part's summaries to single paragraph
                for part in parts:
                    if "plot_summary" in part:
                        part["plot_summary"] = self._sanitize_single_paragraph(part["plot_summary"])
                    if "short_summary" in part:
                        part["short_summary"] = self._sanitize_single_paragraph(part["short_summary"])
                
                return {"parts": parts}
            else:
                raise ValueError("JSON response does not contain 'parts' array")
            
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error for meta-summary: {e}")
            print(f"Raw response: {raw_response[:200]}...")
            # Fallback: return empty parts
            return {"parts": []}
        except Exception as e:
            print(f"⚠️  Error parsing meta-summary response: {e}")
            # Fallback: return empty parts
            return {"parts": []}

    def _save_story_title(self, title: str, output_dir: str):
        """Save the generated story title to 10.title.txt"""
        output_path = os.path.join(output_dir, "10.title.txt")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(title)
            
            print(f"📄 Story title saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving story title file: {e}")

    def generate_chapter_summaries(self, story_text: str, output_dir: str = "../input", resumable_state: ResumableState | None = None) -> list:
        """Generate chapters from meta-summary parts (no LLM calls per chapter)"""
        print("\n=== CHAPTER GENERATION FROM META-SUMMARY ===")
        
        # Calculate story statistics
        story_stats = self._calculate_story_statistics(story_text)
        print(f"📊 Story Statistics:")
        print(f"   📏 Total characters (excluding [names]): {story_stats['total_characters']:,}")
        print(f"   📝 Total words: {story_stats['total_words']:,}")
        print(f"   📄 Total lines: {story_stats['total_lines']:,}")
        
        # Check if resumable and meta-summary already complete
        meta_summary_data = None
        if resumable_state and resumable_state.is_meta_summary_complete():
            cached_meta = resumable_state.get_meta_summary()
            if cached_meta:
                print("\n=== USING CACHED META-SUMMARY ===")
                print("Using cached meta-summary from checkpoint")
                meta_summary_data = cached_meta
        
        # Generate meta-summary if not cached
        if meta_summary_data is None:
            print("\n=== GENERATING META-SUMMARY ===")
            try:
                meta_summary_data = self._generate_meta_summary(story_text)
                print("✅ Meta-summary generated successfully")
                
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_meta_summary(meta_summary_data)
            except Exception as e:
                print(f"❌ Meta-summary generation failed: {e}")
                return []
        
        # Convert meta-summary parts to chapter format
        parts = meta_summary_data.get("parts", [])
        if not parts:
            print("❌ No parts found in meta-summary")
            return []
        
        total_parts = len(parts)
        chapters = []
        
        # Calculate percentage per part (should be roughly 100/total_parts)
        percentage_per_part = 100.0 / total_parts
        
        print(f"\nConverting {total_parts} meta-summary parts to chapters...")
        
        for i, part in enumerate(parts):
            chapter_number = i + 1
            percentage = percentage_per_part * chapter_number
            
            title = part.get("title", f"Chapter {chapter_number}")
            summary = part.get("plot_summary", "")
            short_summary = part.get("short_summary", "")
            
            word_count = len(summary.split()) if summary else 0
            char_count = len(summary) if summary else 0
            short_word_count = len(short_summary.split()) if short_summary else 0
            
            print(f"✅ Chapter {chapter_number}/{total_parts}:")
            print(f"   📖 Title: {title}")
            print(f"   📝 Summary words: {word_count}")
            print(f"   🔸 Short summary words: {short_word_count}")
            
            # Store chapter info
            chapter_info = {
                'part_number': chapter_number,
                'percentage': percentage,
                'title': title,
                'summary': summary,
                'short_summary': short_summary,
                'word_count': word_count,
                'char_count': char_count,
                'short_word_count': short_word_count
            }
            chapters.append(chapter_info)
        
        # Save chapters to files
        self._save_chapters_file(chapters, output_dir)
        
        # Save merged summary (concatenate all part summaries)
        merged_summary = " ".join([part.get("plot_summary", "") for part in parts])
        self._save_summary_file(merged_summary, output_dir, chapters)
        
        # Generate story title from meta-summary if enabled
        if merged_summary and GENERATE_TITLE:
            try:
                self.generate_story_title(merged_summary, output_dir, resumable_state)
            except Exception as e:
                print(f"⚠️  Title generation failed: {e}")
        
        print(f"\n🎉 Chapter generation completed!")
        print(f"📊 Successfully generated {len(chapters)} chapters from meta-summary")
        
        return chapters


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

    def _save_summary_file(self, merged_summary: str, output_dir: str, chapters: list):
        """Save the merged summary to 9.summary.txt and display statistics"""
        output_path = os.path.join(output_dir, "9.summary.txt")
        
        try:
            # Write to 9.summary.txt
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(merged_summary)
            
            # Calculate final statistics
            total_words = len(merged_summary.split())
            total_chars = len(merged_summary)
            
            print(f"📄 Summary saved to: {output_path}")
            print(f"📝 Total words: {total_words}")
            print(f"📏 Total characters: {total_chars}")
            
            # Display chapter statistics
            self._display_chapter_statistics(chapters, total_words, total_chars)
            
        except Exception as e:
            print(f"❌ Error saving summary file: {e}")
    
    def _display_chapter_statistics(self, chapters: list, total_output_words: int, total_output_chars: int):
        """Display a formatted table of chapter statistics"""
        print("\n📊 CHAPTER STATISTICS")
        print("=" * 80)
        print(f"{'Part':<6} {'Title':<30} {'Words':<10} {'Chars':<10} {'Short Words':<12}")
        print("-" * 80)
        
        for chapter in chapters:
            part_num = chapter.get('part_number', 0)
            title = chapter.get('title', 'Untitled')[:28]
            if len(chapter.get('title', '')) > 28:
                title += '..'
            word_count = chapter.get('word_count', 0)
            char_count = chapter.get('char_count', 0)
            short_word_count = chapter.get('short_word_count', 0)
            
            print(f"{part_num:<6} {title:<30} {word_count:<10} {char_count:<10} {short_word_count:<12}")
        
        print("-" * 80)
        print(f"{'TOTAL':<6} {f'{len(chapters)} chapters':<30} {total_output_words:<10} {total_output_chars:<10} {'-':<12}")
        print("=" * 80)


    def preprocess_story(self, story_text, resumable_state: ResumableState | None = None):
        """Preprocess the story to identify and assign voices to characters"""
        print("=== STORY PREPROCESSING ===")
        
        # Extract characters from story
        characters = self.extract_characters_from_story(story_text)
        print(f"Found {len(characters)} unique characters: {', '.join(characters)}")
        
        # Show available voices
        print(f"\nAvailable male voices: {', '.join(self.male_voices)}")
        print(f"Available female voices: {', '.join(self.female_voices)}")
        
        # Check if resumable and character voices already complete
        if resumable_state and resumable_state.is_character_voices_complete():
            cached_voices = resumable_state.get_character_voices()
            if cached_voices:
                print("Using cached character voice assignments from checkpoint")
                self.character_voices.update(cached_voices)
                character_assignments = cached_voices
            else:
                # Assign voices to characters
                character_assignments = self.assign_voices_to_characters(characters)
                # Save to checkpoint if resumable mode enabled
                if resumable_state:
                    resumable_state.set_character_voices(character_assignments)
        else:
            # Assign voices to characters
            character_assignments = self.assign_voices_to_characters(characters)
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                resumable_state.set_character_voices(character_assignments)
        
        # Generate chapter summaries with titles
        try:
            chapters = self.generate_chapter_summaries(story_text, "../input", resumable_state)
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
    parser.add_argument("--force-start", action="store_true", help="Force start from beginning, ignoring any existing checkpoint files")
    parser.add_argument("--disable-resumable", action="store_true", help="Disable resumable mode (default: enabled)")
    args = parser.parse_args()

    # Expose as module-level vars for use inside functions
    AUTO_GENDER = (args.auto_gender or None)
    AUTO_CONFIRM = (args.auto_confirm or None)
    AUTO_CHANGE_SETTINGS = (args.change_settings or None)
    AUTO_REGION = (args.region or None)
    AUTO_LANGUAGE = (args.language or None)

    start_time = time.time()
    
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
    character_manager.preprocess_story(story_text, resumable_state)
    preprocessing_time = time.time() - preprocessing_start
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Clean up checkpoint files if resumable mode was used and everything completed successfully
    if resumable_state:
        print("All operations completed successfully")
        print("Final progress:", resumable_state.get_progress_summary())
        resumable_state.cleanup()
    
    # Print detailed timing information
    print("\n" + "=" * 50)
    print("⏱️  TIMING SUMMARY")
    print("=" * 50)
    print(f"⚙️  Setup time: {setup_time:.3f} seconds")
    print(f"📖 Story reading time: {story_read_time:.3f} seconds")
    print(f"👥 Character preprocessing time: {preprocessing_time:.3f} seconds")
    print(f"⏱️  Total execution time: {total_time:.3f} seconds ({total_time/60:.3f} minutes)")
    print("=" * 50)
