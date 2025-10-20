import os
import re
import time
import argparse
from typing import List, Dict, Any
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

# Maximum number of timeline entries to check in sliding window
MAX_TIMELINE_WINDOW = 16

# Number of words to use for matching story dialogue to timeline
WORD_COUNT = 3

FUZZY_MATCH_CHAR_COUNT = 2

class AudioTimelineProcessor:
    """Processes story and timeline files to create a timeline script with scene and actor information.
    
    This processor reads:
    - Story file (1.story.txt) with character dialogue
    - Timeline file (2.timeline.txt) with durations and text
    - Scene file (3.scene.txt) with scene descriptions
    
    And outputs:
    - Timeline script (2.timeline.script.txt) with duration, scene, actor, and dialogue information
    """
    def __init__(self):
        # Input files
        self.timeline_file = "../../gen.audio/input/2.timeline.txt"
        self.story_file = "../../gen.audio/input/1.story.txt"
        # Scene input and timeline script output
        self.scene_file = "../input/3.scene.txt"
        self.timeline_script_output_file = "../../gen.audio/input/2.timeline.script.txt"
        
    
    def read_timeline_content(self) -> str:
        """Read timeline content from file"""
        try:
            with open(self.timeline_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Timeline file '{self.timeline_file}' not found.")
            return None
        except Exception as e:
            print(f"Error reading timeline file: {e}")
            return None
    
    def parse_timeline_entries(self, content: str) -> List[Dict[str, Any]]:
        """Parse timeline content into structured entries"""
        entries = []
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines, 1):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    try:
                        duration = float(parts[0].strip())
                        text = parts[1].strip()
                        entries.append({
                            'line_number': i,
                            'duration': duration,
                            'text': text,
                            'original_line': line
                        })
                    except ValueError:
                        print(f"Warning: Invalid duration format in line: {line}")
                        continue
        
        print(f"📋 Parsed {len(entries)} timeline entries")
        return entries
    
    def read_story_content(self) -> str:
        """Read story content from file"""
        try:
            with open(self.story_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Story file '{self.story_file}' not found.")
            return None
        except Exception as e:
            print(f"Error reading story file: {e}")
            return None
    
    def parse_story_entries(self, content: str) -> List[Dict[str, Any]]:
        """Parse story content into structured entries"""
        entries = []
        lines = content.strip().split('\n')
        
        for i, line in enumerate(lines, 1):
            if line.strip() and line.startswith('['):
                # Parse [character] dialogue format
                match = re.match(r'^\[([^\]]+)\]\s*(.*)$', line)
                if match:
                    character = match.group(1)
                    dialogue = match.group(2)
                    entries.append({
                        'line_number': i,
                        'character': character,
                        'dialogue': dialogue,
                        'original_line': line
                    })
        
        print(f"📋 Parsed {len(entries)} story entries")
        return entries
    
    def create_story_timeline_mapping(self, story_entries: List[Dict[str, Any]], 
                                     timeline_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create mapping between story and timeline using improved two-pointer algorithm
        
        Returns a list where each entry represents:
        - story_pointer: index of the story entry
        - timeline_pointer_start: start index of timeline range (inclusive)
        - timeline_pointer_end: end index of timeline range (inclusive)
        
        For unmatched story entries, both timeline pointers are set to -1.
        """
        print("🔗 Creating story-timeline mapping with improved two-pointer algorithm...")
        
        mapped_entries = []
        pointer_a = 0  # Pointer A at story entries
        pointer_b = 0  # Pointer B at timeline entries
        self.mismatch_summary = []  # Store all mismatches for summary
        
        while pointer_a < len(story_entries) and pointer_b < len(timeline_entries):
            story_entry = story_entries[pointer_a]
            
            # Use single word count for matching
            story_last_words = self.get_last_n_words(story_entry['dialogue'], WORD_COUNT)

            print()
            print(f"🔍 Checking story[{pointer_a}]: {story_last_words}")
            
            # Search from pointer B to end of timeline
            result, time_line_entries = self.find_best_timeline_match(
                story_last_words, timeline_entries, max(0, pointer_b - 1)
            )
            
            if result:  # Match found
                print(f"✅ Found exact match with {WORD_COUNT} words")
            
            if result:
                # Create single mapping entry for this story segment
                # Use the actual match start index, not the current pointer_b
                mapped_entries.append({
                    'story_pointer': pointer_a,
                    'timeline_pointer_start': result['start_idx'],
                    'timeline_pointer_end': result['end_idx']
                })
                
                # Move pointers: A to next story, B to next timeline after match
                pointer_a += 1
                pointer_b = result['end_idx'] + 1
                print(f"🟢 Mapped story[{pointer_a-1}] to timeline[{result['start_idx']}:{result['end_idx']}]")
                print(f"🟢 Moved pointer A to {pointer_a}, pointer B to {pointer_b}")

            else:
                # No match found - create entry with -1 for both timeline pointers
                mapped_entries.append({
                    'story_pointer': pointer_a,
                    'timeline_pointer_start': -1,
                    'timeline_pointer_end': -1
                })

                print()

                for entry in time_line_entries:
                    if isinstance(entry, dict) and 'timeline_indices' in entry:
                        words_str = str(entry['words'])
                        indices_str = str(entry['timeline_indices'])
                        print(f"   {words_str:<50} {indices_str}")
                    else:
                        words_str = str(entry)
                        print(f"   {words_str:<50}")
                
                if pointer_b < len(timeline_entries):
                    print(f"❌ No match found for story[{pointer_a}] - marked as unmatched")
                else:
                    print(f"❌ No more timeline entries available")
                    print(f"   Story[{pointer_a}]: '{story_entry['dialogue'][:50]}...'")
                    print(f"   Issue: Timeline file appears incomplete (ends at entry {len(timeline_entries)})")
                
                # Only move story pointer, keep timeline pointer for next story entry
                pointer_a += 1
            print("--------------------------------")
        
        print()
        print(f"✅ Created {len(mapped_entries)} story-timeline mappings")
        
        # Count matched vs unmatched story entries
        matched_stories = sum(1 for mapping in mapped_entries if mapping['timeline_pointer_start'] != -1)
        unmatched_stories = len(mapped_entries) - matched_stories
        print(f"📊 Story mapping: {matched_stories} matched, {unmatched_stories} unmatched")
        
        # Create timeline-to-story mapping for comprehensive coverage
        timeline_to_story_map = self.create_timeline_to_story_map(mapped_entries, len(timeline_entries))
        
        return mapped_entries, timeline_to_story_map

    def create_timeline_to_story_map(self, mapped_entries: List[Dict[str, Any]], 
                                   timeline_length: int) -> Dict[int, int]:
        """Create a mapping from timeline index to story index.
        
        Returns a dictionary where:
        - Key: timeline index (int)
        - Value: story index (int) or -1 for unmatched timeline entries
        """
        print("🗺️  Creating comprehensive timeline-to-story mapping...")
        
        # Initialize all timeline entries as unmatched (-1)
        timeline_to_story_map = {i: -1 for i in range(timeline_length)}
        
        # Fill in the matched entries
        for mapping in mapped_entries:
            story_idx = mapping['story_pointer']
            timeline_start = mapping['timeline_pointer_start']
            timeline_end = mapping['timeline_pointer_end']
            
            # Skip unmatched story entries
            if timeline_start == -1 or timeline_end == -1:
                continue
            
            # Map all timeline entries in the range to this story entry
            for timeline_idx in range(timeline_start, timeline_end + 1):
                timeline_to_story_map[timeline_idx] = story_idx
        
        # Count matched vs unmatched
        matched_count = sum(1 for story_idx in timeline_to_story_map.values() if story_idx != -1)
        unmatched_count = timeline_length - matched_count
        
        print(f"📊 Timeline mapping: {matched_count} matched, {unmatched_count} unmatched")
        
        return timeline_to_story_map

    def parse_scene_entries(self, content: str) -> List[Dict[str, Any]]:
        """Parse scene file entries into ordered list with scene_id per entry.

        Expects lines like: (scene_1.1) ...
        """
        entries: List[Dict[str, Any]] = []
        lines = content.split('\n')
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped.startswith('('):
                continue
            # Extract scene id and text after the scene identifier
            # Format: (scene_1.1) {settings}; ((characters)) ...
            m = re.match(r'^\(([^)]+)\)\s*(.*)$', line_stripped)
            if m:
                entries.append({
                    'scene_id': m.group(1),
                    'text': m.group(2).strip(),
                    'raw': line_stripped
                })
        print(f"📋 Parsed {len(entries)} scene entries from {self.scene_file}")
        return entries

    def read_scene_content(self) -> str:
        """Read scene content from scene file"""
        try:
            with open(self.scene_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Scene file '{self.scene_file}' not found.")
            return None
        except Exception as e:
            print(f"Error reading scene file: {e}")
            return None

    def save_timeline_script(self, timeline_to_story_map: Dict[int, int],
                           story_entries: List[Dict[str, Any]], 
                           timeline_entries: List[Dict[str, Any]]) -> None:
        """Write timeline script with duration, scene_id, actor_name and dialogue.
        Uses timeline-to-story mapping to ensure all timeline entries are included.

        Format per line:
        - Matched: duration: scene_id = <scene_#.#>, actor_name=<actor>, dialogue=<dialogue>
        - Unmatched: duration: UNKNOWN
        - Silence: duration : ...
        """
        try:
            scene_content = self.read_scene_content()
            if scene_content is None:
                print("❌ Could not read scene file; skipping timeline script output")
                return
            scene_entries = self.parse_scene_entries(scene_content)

            def get_scene_for_story_index(story_index: int) -> Dict[str, str]:
                if 0 <= story_index < len(scene_entries):
                    return scene_entries[story_index]
                return {'scene_id': 'unknown', 'text': ''}

            count_written = 0
            with open(self.timeline_script_output_file, 'w', encoding='utf-8') as f:
                # Process every timeline entry in order
                for timeline_idx, timeline_entry in enumerate(timeline_entries):
                    duration_val = timeline_entry['duration']
                    timeline_text = timeline_entry['text']
                    
                    # For silence, output only: "duration : ..."
                    if self.is_silence_marker(timeline_text.strip()):
                        f.write(f"{duration_val:.6f} : {timeline_text.strip()}\n")
                        count_written += 1
                        continue
                    
                    # Get corresponding story index from mapping
                    story_idx = timeline_to_story_map.get(timeline_idx, -1)
                    
                    if story_idx == -1:
                        # Unmatched timeline entry
                        f.write(f"{duration_val:.6f}: UNKNOWN\n")
                        count_written += 1
                    else:
                        # Matched timeline entry - get story information
                        story_entry = story_entries[story_idx]
                        scene_info = get_scene_for_story_index(story_idx)
                        scene_id = scene_info.get('scene_id', 'unknown')
                        scene_text = scene_info.get('text', '')
                        
                        actor_name = story_entry['character']
                        dialogue_text = story_entry['dialogue']
                        
                        f.write(f"{duration_val:.6f}: {scene_id} = {scene_text}, {actor_name} = {dialogue_text}\n")
                        count_written += 1

            print(f"✅ Saved timeline script to: {self.timeline_script_output_file} ({count_written} lines)")
        except Exception as e:
            print(f"❌ Error saving timeline script file: {e}")
            raise
    
    def find_best_timeline_match(self, story_words: List[str], timeline_entries: List[Dict[str, Any]], 
                                start_idx: int) -> Dict[str, Any]:
        if not story_words:
            return None
        
        required_word_count = len(story_words)
        time_line_entries = []
        
        # Use sliding word-based window approach
        max_timeline_end = min(start_idx + MAX_TIMELINE_WINDOW, len(timeline_entries))
        
        for timeline_end_idx in range(start_idx, max_timeline_end):
            # Build combined text from start_idx to current timeline_end_idx
            combined_text = ""
            timeline_indices = []
            for idx in range(start_idx, timeline_end_idx + 1):
                timeline_indices.append(idx)
                if combined_text:
                    combined_text += " " + timeline_entries[idx]['text']
                else:
                    combined_text = timeline_entries[idx]['text']
            
            # Extract all words from combined text
            all_words = re.findall(r'\b\w+\b', combined_text.lower())
            
            if len(all_words) < required_word_count:
                # Not enough words, track what we have and continue to next timeline entry
                entry_info = {
                    'words': all_words,
                    'timeline_indices': timeline_indices.copy(),
                    'timeline_texts': [timeline_entries[idx]['text'] for idx in timeline_indices]
                }
                time_line_entries.append(entry_info)
                continue
            
            # Now slide through words within this combined text
            for word_start in range(len(all_words) - required_word_count + 1):
                word_window = all_words[word_start:word_start + required_word_count]
                
                # Check for exact match
                is_match = self.fuzzy_match_word_lists(word_window, [word.lower() for word in story_words])
                
                # Create detailed entry
                entry_info = {
                    'words': word_window,
                    'timeline_indices': timeline_indices.copy(),
                    'timeline_texts': [timeline_entries[idx]['text'] for idx in timeline_indices]
                }
                time_line_entries.append(entry_info)
                
                if is_match:
                    return ({
                        'start_idx': start_idx,
                        'end_idx': timeline_end_idx,
                        'score': 1.0,  # Perfect match
                        'combined_text': combined_text,
                        'timeline_count': timeline_end_idx - start_idx + 1
                    }, time_line_entries)
        
        return (None, time_line_entries)
    
    def is_end_match(self, story_words: List[str], combined_text: str) -> bool:
        """Check if the last N words of combined text match the story words (with fuzzy matching for typos)"""
        # Clean both texts for better matching
        cleaned_combined = self.clean_text_for_matching(combined_text)
        cleaned_story_words = [self.clean_text_for_matching(word) for word in story_words]
        
        # Extract words from cleaned text
        combined_words = re.findall(r'\b\w+\b', cleaned_combined)
        
        if len(combined_words) < len(cleaned_story_words):
            return False
        
        # Get the last N words from combined text
        last_words = combined_words[-len(cleaned_story_words):]
        
        # Check fuzzy match (allows up to 2 character differences per word)
        is_match = self.fuzzy_match_word_lists(last_words, [word for word in cleaned_story_words if word])
        return (is_match, last_words)
    
    def fuzzy_match_word_lists(self, timeline_words: List[str], story_words: List[str]) -> bool:
        """Check if two word lists match with fuzzy matching (up to 2 character differences per word)"""
        if len(timeline_words) != len(story_words):
            return False
        
        for timeline_word, story_word in zip(timeline_words, story_words):
            if not self.fuzzy_match_words(timeline_word.lower(), story_word.lower()):
                return False
        
        return True
    
    def fuzzy_match_words(self, word1: str, word2: str) -> bool:
        """Check if two words match with up to max_diff character differences using edit distance"""

        # Exact match first (fastest)
        if word1 == word2:
            return True
        
        # Determine fuzzy match tolerance based on word length
        # Use longer word to determine tolerance
        max_len = max(len(word1), len(word2))
        if max_len >= 5:
            max_diff = FUZZY_MATCH_CHAR_COUNT
        elif max_len >= 3:
            max_diff = FUZZY_MATCH_CHAR_COUNT // 2
        else:
            max_diff = FUZZY_MATCH_CHAR_COUNT // 4
        
        # If length difference is too large, can't match with max_diff changes
        if abs(len(word1) - len(word2)) > max_diff:
            return False
        
        # Calculate Levenshtein distance (edit distance)
        edit_distance = self.levenshtein_distance(word1, word2)
        return edit_distance <= max_diff
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate the Levenshtein (edit) distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        # Create matrix
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, and substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_last_n_words(self, text: str, n: int) -> List[str]:
        """Get the last n words from a text (preserving case)"""
        words = re.findall(r'\b\w+\b', text)
        return words[-n:] if len(words) >= n else words
    
    def clean_text_for_matching(self, text: str) -> str:
        """Remove non-alphanumeric characters for better matching"""
        # Keep only alphanumeric characters and spaces
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def is_silence_marker(self, text: str) -> bool:
        """Check if text is a silence marker (dots)"""
        text = text.strip()
        return re.match(r'^\.+$', text) is not None
    

    def verify_scene_coverage(self) -> bool:
        """Verify scene coverage and order in the generated timeline script
        
        Returns:
            bool: True if validation passes, False if validation fails
        """
        print("\n" + "="*50)
        print("🔍 VERIFYING SCENE COVERAGE")
        print("="*50)
        
        try:
            # Parse original scene file
            original_scenes = self.parse_original_scenes()
            if not original_scenes:
                print("❌ Could not parse original scene file")
                return False
                
            # Parse generated timeline script
            timeline_scenes = self.parse_timeline_script_scenes()
            if not timeline_scenes:
                print("❌ Could not parse timeline script file")
                return False
                
            # Analyze coverage and return validation result
            return self.analyze_scene_coverage(original_scenes, timeline_scenes)
            
        except Exception as e:
            print(f"❌ Error during scene verification: {e}")
            return False
    
    def parse_original_scenes(self) -> List[str]:
        """Parse scene file and extract scene IDs in order"""
        scene_ids = []
        try:
            with open(self.scene_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('(scene_'):
                    # Extract scene ID like scene_1.1, scene_2.3, etc.
                    match = re.match(r'^\(([^)]+)\)', line)
                    if match:
                        scene_ids.append(match.group(1))
        except Exception as e:
            print(f"❌ Error reading scene file: {e}")
            return []
        
        return scene_ids
    
    def parse_timeline_script_scenes(self) -> List[str]:
        """Parse timeline.script and extract scene IDs that appear"""
        scene_ids = []
        try:
            with open(self.timeline_script_output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if ':' in line and 'scene_' in line:
                    # Look for pattern like "duration: scene_1.1 = ..."
                    match = re.search(r'(scene_\d+\.\d+)', line)
                    if match:
                        scene_ids.append(match.group(1))
        except Exception as e:
            print(f"❌ Error reading timeline script file: {e}")
            return []
        
        return scene_ids
    
    def analyze_scene_coverage(self, original_scenes: List[str], timeline_scenes: List[str]) -> bool:
        """Analyze and report scene coverage
        
        Returns:
            bool: True if validation passes, False if validation fails
        """
        original_set = set(original_scenes)
        timeline_set = set(timeline_scenes)
        
        # Missing scenes (in original but not in timeline)
        missing_scenes = original_set - timeline_set
        
        # Calculate coverage
        coverage_percent = (len(timeline_set) / len(original_set)) * 100 if original_set else 0
        
        # Report results
        print(f"📋 Total scenes in original: {len(original_scenes)}")
        print(f"📋 Unique scenes in timeline: {len(timeline_set)}")
        print(f"📋 Total scene references in timeline: {len(timeline_scenes)}")
        print(f"📊 Scene coverage: {coverage_percent:.1f}% ({len(timeline_set)}/{len(original_scenes)})")
        
        validation_passed = True
        
        if missing_scenes:
            print(f"\n❌ Missing {len(missing_scenes)} scenes:")
            for scene in sorted(missing_scenes)[:10]:  # Show first 10
                print(f"   - {scene}")
            if len(missing_scenes) > 10:
                print(f"   ... and {len(missing_scenes) - 10} more")
            validation_passed = False
        else:
            print("\n✅ All scenes are present in timeline!")
        
        # Check order for scenes that exist in both
        common_scenes = original_set & timeline_set
        original_order = [scene for scene in original_scenes if scene in common_scenes]
        
        # Get order from timeline (first occurrence of each scene)
        timeline_order = []
        seen = set()
        for scene in timeline_scenes:
            if scene in common_scenes and scene not in seen:
                timeline_order.append(scene)
                seen.add(scene)
        
        # Check if order matches
        order_matches = original_order == timeline_order
        
        if order_matches:
            print("✅ Scene order matches original!")
        else:
            print("❌ Scene order does NOT match original!")
            # Show first few differences
            max_check = min(len(original_order), len(timeline_order), 5)
            differences = 0
            for i in range(max_check):
                if original_order[i] != timeline_order[i]:
                    differences += 1
                    print(f"   Position {i+1}: Expected {original_order[i]}, Got {timeline_order[i]}")
            
            if differences == 0:
                print("   (Order matches for first few scenes)")
            validation_passed = False
        
        print("="*50)
        return validation_passed
 
    def process_audio_timeline(self, bypass_validation: bool = False) -> bool:
        """Process story+timeline+scenes -> timeline script"""
        print("🚀 Starting Audio Timeline Processing...")
        
        # Create Timeline Script
        print("\n" + "="*50)
        print("Creating Timeline Script")
        print("="*50)
        print(f"📁 Reading story from: {self.story_file}")
        print(f"📁 Reading timeline from: {self.timeline_file}")
        
        # Read story and timeline files
        story_content = self.read_story_content()
        timeline_content = self.read_timeline_content()
        
        if story_content is None or timeline_content is None:
            print("❌ Could not read one or more files")
            return False
        
        # Parse files
        story_entries = self.parse_story_entries(story_content)
        timeline_entries = self.parse_timeline_entries(timeline_content)
        
        if not story_entries or not timeline_entries:
            print("❌ No valid entries found in one or more files")
            return False
        
        print(f"📋 Processing {len(story_entries)} story lines, {len(timeline_entries)} timeline lines")
        
        # Create story-timeline mapping
        actor_mapped_entries, timeline_to_story_map = self.create_story_timeline_mapping(story_entries, timeline_entries)
        
        if not actor_mapped_entries:
            print("❌ No actor mappings could be created")
            return False
        
        # Store unmatched count and indices for later use in summary
        self.unmatched_story_count = sum(1 for mapping in actor_mapped_entries if mapping['timeline_pointer_start'] == -1)
        self.unmatched_story_indices = [mapping['story_pointer'] for mapping in actor_mapped_entries if mapping['timeline_pointer_start'] == -1]
        
        # Save the timeline script using comprehensive mapping
        self.save_timeline_script(timeline_to_story_map, story_entries, timeline_entries)
        
        print(f"\n✅ Processing completed successfully!")
        print(f"📄 Timeline script: {self.timeline_script_output_file}")
        
        # Verify scene coverage and order (unless bypassed)
        if not bypass_validation:
            validation_passed = self.verify_scene_coverage()
            if not validation_passed:
                print("❌ Scene coverage validation failed")
                return False
        else:
            print("Validation bypassed via --bypass-validation flag")
        
        return True

def main():
    """Main function"""
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process story and timeline files to create timeline script")
    parser.add_argument("--bypass-validation", action="store_true", 
                       help="Skip scene coverage validation")
    parser.add_argument("story_file", nargs="?", default="../../gen.audio/input/1.story.txt",
                       help="Path to story file (default: ../../gen.audio/input/1.story.txt)")
    parser.add_argument("timeline_file", nargs="?", default="../../gen.audio/input/2.timeline.txt",
                       help="Path to timeline file (default: ../../gen.audio/input/2.timeline.txt)")
    
    args = parser.parse_args()
    
    story_file = args.story_file
    timeline_file = args.timeline_file
    
    # Check if files exist
    if not os.path.exists(story_file):
        print(f"❌ Story file '{story_file}' not found")
        print("Usage: python 4.audio.py [--bypass-validation] [story_file] [timeline_file]")
        return 1
    
    if not os.path.exists(timeline_file):
        print(f"❌ Timeline file '{timeline_file}' not found")
        print("Usage: python 4.audio.py [--bypass-validation] [story_file] [timeline_file]")
        return 1
    
    # Create processor and process
    processor = AudioTimelineProcessor()
    processor.story_file = story_file
    processor.timeline_file = timeline_file
    
    start_time = time.time()
    success = processor.process_audio_timeline(bypass_validation=args.bypass_validation)
    end_time = time.time()
    
    if success:
        print(f"⏱️  Total processing time: {end_time - start_time:.2f} seconds")
        return 0
    else:
        print("❌ Processing failed")
        return 1

if __name__ == "__main__":
    exit(main())
