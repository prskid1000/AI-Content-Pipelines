import os
import re
import json
import requests
import argparse
from functools import partial
import builtins as _builtins
import time
from pathlib import Path


# Character and word count limits (min-max ranges)
CHARACTER_SUMMARY_CHARACTER_MIN = 300
CHARACTER_SUMMARY_CHARACTER_MAX = 600
CHARACTER_SUMMARY_WORD_MIN = 50
CHARACTER_SUMMARY_WORD_MAX = 160

LOCATION_SUMMARY_CHARACTER_MIN = 1200
LOCATION_SUMMARY_CHARACTER_MAX = 3000
LOCATION_SUMMARY_WORD_MIN = 250
LOCATION_SUMMARY_WORD_MAX = 375

STORY_DESCRIPTION_CHARACTER_MIN = 6600
STORY_DESCRIPTION_CHARACTER_MAX = 7200
STORY_DESCRIPTION_WORD_MIN = 1100
STORY_DESCRIPTION_WORD_MAX = 1200

# Feature flags
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them

# Model constants for easy switching
MODEL_STORY_DESCRIPTION = "qwen/qwen3-14b"  # Model for generating story descriptions
MODEL_CHARACTER_GENERATION = "qwen/qwen3-14b"  # Model for character description generation
MODEL_CHARACTER_SUMMARY = "qwen/qwen3-14b"  # Model for character summary generation
MODEL_LOCATION_EXPANSION = "qwen/qwen3-14b"  # Model for location expansion


# Resumable state management
class ResumableState:
    """Manages resumable state for expensive LLM operations."""
    
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
            "story_summary": {"completed": False, "result": None},
            "locations": {"completed": [], "results": {}},
            "location_summaries": {"completed": [], "results": {}},
            "characters": {"completed": [], "results": {}},
            "character_summaries": {"completed": [], "results": {}},
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
    
    def is_story_summary_complete(self) -> bool:
        """Check if story description generation is complete."""
        return self.state["story_summary"]["completed"]
    
    def get_story_summary(self) -> str | None:
        """Get cached story description if available."""
        return self.state["story_summary"]["result"]
    
    def set_story_summary(self, description: str):
        """Set story description and mark as complete."""
        self.state["story_summary"]["completed"] = True
        self.state["story_summary"]["result"] = description
        self._save_state()
    
    def is_location_complete(self, location_id: str) -> bool:
        """Check if specific location expansion is complete."""
        return location_id in self.state["locations"]["completed"]
    
    def get_location_description(self, location_id: str) -> str | None:
        """Get cached location description if available."""
        return self.state["locations"]["results"].get(location_id)
    
    def set_location_description(self, location_id: str, description: str):
        """Set location description and mark as complete."""
        if location_id not in self.state["locations"]["completed"]:
            self.state["locations"]["completed"].append(location_id)
        self.state["locations"]["results"][location_id] = description
        self._save_state()
    
    def is_location_summary_complete(self, location_id: str) -> bool:
        """Check if specific location summary is complete."""
        return location_id in self.state["location_summaries"]["completed"]
    
    def get_location_summary(self, location_id: str) -> str | None:
        """Get cached location summary if available."""
        return self.state["location_summaries"]["results"].get(location_id)
    
    def set_location_summary(self, location_id: str, summary: str):
        """Set location summary and mark as complete."""
        if location_id not in self.state["location_summaries"]["completed"]:
            self.state["location_summaries"]["completed"].append(location_id)
        self.state["location_summaries"]["results"][location_id] = summary
        self._save_state()
    
    def is_character_complete(self, character_name: str) -> bool:
        """Check if specific character description is complete."""
        return character_name in self.state["characters"]["completed"]
    
    def get_character_description(self, character_name: str) -> str | None:
        """Get cached character description if available."""
        return self.state["characters"]["results"].get(character_name)
    
    def set_character_description(self, character_name: str, description: str):
        """Set character description and mark as complete."""
        if character_name not in self.state["characters"]["completed"]:
            self.state["characters"]["completed"].append(character_name)
        self.state["characters"]["results"][character_name] = description
        self._save_state()
    
    def is_character_summary_complete(self, character_name: str) -> bool:
        """Check if specific character summary is complete."""
        return character_name in self.state["character_summaries"]["completed"]
    
    def get_character_summary(self, character_name: str) -> str | None:
        """Get cached character summary if available."""
        return self.state["character_summaries"]["results"].get(character_name)
    
    def set_character_summary(self, character_name: str, summary: str):
        """Set character summary and mark as complete."""
        if character_name not in self.state["character_summaries"]["completed"]:
            self.state["character_summaries"]["completed"].append(character_name)
        self.state["character_summaries"]["results"][character_name] = summary
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
        story_done = "✓" if self.is_story_summary_complete() else "✗"
        locations_done = len(self.state["locations"]["completed"])
        locations_total = len(self.state["locations"]["results"]) + len([k for k in self.state["locations"]["results"].keys() if k not in self.state["locations"]["completed"]])
        characters_done = len(self.state["characters"]["completed"])
        characters_total = len(self.state["characters"]["results"]) + len([k for k in self.state["characters"]["results"].keys() if k not in self.state["characters"]["completed"]])
        summaries_done = len(self.state["character_summaries"]["completed"])
        summaries_total = len(self.state["character_summaries"]["results"]) + len([k for k in self.state["character_summaries"]["results"].keys() if k not in self.state["character_summaries"]["completed"]])
        
        # Backward compatibility: check if location_summaries exists
        if "location_summaries" in self.state:
            location_summaries_done = len(self.state["location_summaries"]["completed"])
            location_summaries_total = len(self.state["location_summaries"]["results"]) + len([k for k in self.state["location_summaries"]["results"].keys() if k not in self.state["location_summaries"]["completed"]])
        else:
            location_summaries_done = 0
            location_summaries_total = 0
        
        return (
            f"Progress: Story({story_done}) Locations({locations_done}/{locations_total}) "
            f"Characters({characters_done}/{characters_total}) "
            f"CharSummaries({summaries_done}/{summaries_total}) LocSummaries({location_summaries_done}/{location_summaries_total})"
        )

# Ensure immediate flush on print (consistent with other scripts)
print = partial(_builtins.print, flush=True)


def read_file_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}")
        return None
    except Exception as ex:
        print(f"ERROR: Failed to read file: {path} ({ex})")
        return None


# --- New simplified paired parser for patterns ---
# File-wide single pattern required: either
#   A) [] ...  then  ()(()) ...
#   B) ()(()) ...  then  [] ...

_DIALOGUE_RE = re.compile(r"^\[([^\]]+)\]\s*:?\s*(.*)$")
_SCENE_RE = re.compile(r"^\(([^)]+)\)\s*:?\s*(.*)$")
_SCENE_CHARS_IN_TEXT_RE = re.compile(r"\(\(([^)]+)\)\)")
_CHARACTER_LINE_RE = re.compile(r"^\(\(([^)]+)\)\):\s+(.+)$")

# Location patterns: {{loc_1, description}} or {{loc_1}}
_LOCATION_FULL_RE = re.compile(r"\{\{([^,]+),\s*([^}]+)\}\}")
_LOCATION_REF_RE = re.compile(r"\{\{([^}]+)\}\}")

_SCENE_ID_NUMERIC_RE = re.compile(r"^(\d+)\.(\d+)$")


def _classify_line(raw_line: str):
    line = raw_line.strip()
    if not line:
        return None, None
    if line.startswith("["):
        m = _DIALOGUE_RE.match(line)
        if m:
            return "dialogue", {
                "character": m.group(1).strip(),
                "dialogue": m.group(2).strip(),
            }
        return None, None
    if line.startswith("("):
        m = _SCENE_RE.match(line)
        if not m:
            return None, None
        scene_id = m.group(1).strip()
        rest = m.group(2).strip()
        # Extract all ((character)) mentions anywhere in the rest of the line
        characters = [c.strip() for c in _SCENE_CHARS_IN_TEXT_RE.findall(rest) if c.strip()]
        # Remove character tokens from the description and normalize whitespace
        description = _SCENE_CHARS_IN_TEXT_RE.sub("", rest).strip()
        if description:
            description = re.sub(r"\s{2,}", " ", description)
        return "scene", {
            "scene_id": scene_id,
            "scene_characters": characters,
            "description": description,
        }
    return None, None


def _tokenize(content: str):
    tokens = []
    for i, raw in enumerate(content.splitlines(), 1):
        kind, payload = _classify_line(raw)
        if kind is None:
            continue
        tokens.append({
            "kind": kind,  # "dialogue" | "scene"
            "line_no": i,
            "raw": raw.strip(),
            "data": payload,
        })
    return tokens


def _detect_order(tokens):
    # Returns (first_kind, second_kind) or None if not enough info
    for t in tokens:
        if t["kind"] in ("dialogue", "scene"):
            if t["kind"] == "dialogue":
                return ("dialogue", "scene")
            return ("scene", "dialogue")
    return None


def _pair_tokens(tokens, order):
    expected_first, expected_second = order
    i = 0
    pairs = []
    pair_index = 0

    while i < len(tokens):
        pair_index += 1

        first = None
        second = None

        # first
        if i < len(tokens) and tokens[i]["kind"] == expected_first:
            first = tokens[i]
            i += 1
        else:
            print(f"PAIR {pair_index}: Missing {expected_first} '[]' expected before the other part.")

        # second
        if i < len(tokens) and tokens[i]["kind"] == expected_second:
            second = tokens[i]
            i += 1
        else:
            print(f"PAIR {pair_index}: Missing {expected_second} '()(())' counterpart.")

        # Normalize storage: always keep both roles explicit
        dialogue = first if (first and first["kind"] == "dialogue") else second if (second and second["kind"] == "dialogue") else None
        scene = first if (first and first["kind"] == "scene") else second if (second and second["kind"] == "scene") else None

        # Validate scene has at least one ((character))
        if scene is not None:
            ln = scene["line_no"]
            characters = scene["data"].get("scene_characters") or []
            if not characters:
                print(f"PAIR {pair_index}: Scene at line {ln} missing '((character))' in ()(()).")

        pairs.append({
            "pair_index": pair_index,
            "dialogue": dialogue,
            "scene": scene,
        })

    return pairs


def _collect_unique_character_sets(tokens):
    dchars = set()
    schars = set()
    for t in tokens:
        if t["kind"] == "dialogue":
            c = (t["data"].get("character") or "").strip()
            if c:
                dchars.add(c)
        elif t["kind"] == "scene":
            for c in (t["data"].get("scene_characters") or []):
                c = (c or "").strip()
                if c:
                    schars.add(c)
    return dchars, schars


def _validate_character_consistency(tokens, pairs) -> int:
    """
    Validates that:
    - Unique character set from dialogue lines equals unique character set from scenes.

    Returns number of validation errors.
    """
    errors = 0

    dialogue_chars, scene_chars = _collect_unique_character_sets(tokens)

    # Global set equality check
    if dialogue_chars != scene_chars:
        errors += 1
        print("ERROR: Character set mismatch between dialogues and scenes.")
        print(f"  Dialogue unique characters ({len(dialogue_chars)}): {sorted(dialogue_chars)}")
        print(f"  Scene unique characters     ({len(scene_chars)}): {sorted(scene_chars)}")
        only_in_dialogue = sorted(dialogue_chars - scene_chars)
        only_in_scenes = sorted(scene_chars - dialogue_chars)
        if only_in_dialogue:
            print(f"  Only in dialogues: {only_in_dialogue}")
        if only_in_scenes:
            print(f"  Only in scenes:    {only_in_scenes}")
    else:
        print(f"Characters match: {len(dialogue_chars)} names {sorted(dialogue_chars)}")

    return errors


def _validate_scene_id_continuity(tokens) -> int:
    """
    Validates numeric scene id continuity within each major group.
    Example: for 1.2, 1.3, 1.5 -> reports missing 1.4.

    Only scene ids matching "<int>.<int>" are considered. Other formats are ignored.
    Returns number of missing ids found.
    """
    from collections import defaultdict

    scenes = []
    for t in tokens:
        if t["kind"] == "scene":
            sid = (t["data"].get("scene_id") or "").strip()
            m = _SCENE_ID_NUMERIC_RE.match(sid)
            if not m:
                continue
            major = int(m.group(1))
            minor = int(m.group(2))
            scenes.append((major, minor, sid, t["line_no"]))

    if not scenes:
        return 0

    grouped: dict[int, list[tuple[int, str, int]]] = defaultdict(list)
    for major, minor, sid, ln in scenes:
        grouped[major].append((minor, sid, ln))

    errors = 0
    for major, items in sorted(grouped.items()):
        items.sort(key=lambda x: x[0])
        minors = [m for (m, _sid, _ln) in items]
        minor_set = set(minors)
        if not minors:
            continue
        low, high = min(minors), max(minors)
        for missing_minor in range(low, high):
            if missing_minor not in minor_set:
                errors += 1
                # find nearest neighbors for helpful context
                before = None
                after = None
                for m, sid, ln in items:
                    if m < missing_minor:
                        before = (sid, ln)
                    elif m > missing_minor and after is None:
                        after = (sid, ln)
                        break
                print(f"ERROR: Missing scene id {major}.{missing_minor}")
                ctx_parts = []
                if before:
                    ctx_parts.append(f"after {before[0]} (line {before[1]})")
                if after:
                    ctx_parts.append(f"before {after[0]} (line {after[1]})")
                if ctx_parts:
                    print("  " + " and ".join(ctx_parts))

    return errors

def _extract_locations_from_content(content: str) -> dict[str, str]:
    """Extract all unique locations from content and return mapping of loc_id -> description."""
    locations = {}
    
    # First pass: find all full location definitions {{loc_id, description}}
    for match in _LOCATION_FULL_RE.finditer(content):
        loc_id = match.group(1).strip()
        description = match.group(2).strip()
        if loc_id and description:
            locations[loc_id] = description
    
    # Second pass: find all location references {{loc_id}} and ensure they have descriptions
    for match in _LOCATION_REF_RE.finditer(content):
        loc_id = match.group(1).strip()
        if loc_id and loc_id not in locations:
            # This is a reference without definition - we'll keep it as is for now
            # The validation will catch this if needed
            pass
    
    return locations

def _sanitize_single_paragraph(text: str) -> str:
    if not text:
        return ""
    if text.startswith("```"):
        m = re.search(r"```(?:[a-zA-Z]+)?\s*([\s\S]*?)\s*```", text)
        if m:
            text = m.group(1)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _schema_character() -> dict[str, object]:
    """JSON schema for Western character description focusing on face, hair, eyes, skin, and clothing details."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "character_description",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "face": {
                        "type": "object",
                        "properties": {
                            "head_shape": {"type": "string", "enum": ["oval", "round", "square", "rectangular", "heart", "diamond", "triangle"]},
                            "skin_tone": {"type": "string", "enum": ["pale", "fair", "medium", "olive", "tan", "dark", "brown", "black"]},
                            "eyes_color": {"type": "string", "enum": ["brown", "blue", "green", "hazel", "gray", "amber", "black", "violet"]},
                            "hair_color": {"type": "string", "enum": ["black", "brown", "blonde", "red", "gray", "white", "auburn", "chestnut", "platinum"]},
                            "hair_texture": {"type": "string", "enum": ["straight", "wavy", "curly", "coily", "kinky"]},
                            "facial_hair": {"type": "string", "description": "Beard, mustache, goatee, or clean-shaven"}
                        },
                        "required": ["head_shape", "skin_tone", "eyes_color", "hair_color", "hair_texture"]
                    },
                    "clothing": {
                        "type": "object",
                        "properties": {
                            "tops": {
                        "type": "object",
                        "properties": {
                                    "type": {"type": "string", "enum": ["dress_shirt", "casual_shirt", "t-shirt", "polo_shirt", "sweater", "cardigan", "blazer", "suit_jacket", "hoodie", "tank_top", "turtleneck", "henley", "flannel_shirt", "oxford_shirt", "button_down", "long_sleeve", "short_sleeve", "polo", "crew_neck", "v_neck"]},
                                    "color": {"type": "string", "description": "Color with prefix (e.g., 'dark blue', 'light green', 'navy blue')"},
                                    "pattern": {"type": "string", "description": "Solid, striped, plaid, checkered, etc."},
                                    "material": {"type": "string", "description": "Cotton, silk, wool, polyester, linen, etc."},
                                    "fit": {"type": "string", "enum": ["tight", "fitted", "loose", "oversized"]}
                                },
                                "required": ["type", "color"]
                            },
                            "bottoms": {
                        "type": "object",
                        "properties": {
                                    "type": {"type": "string", "enum": ["dress_pants", "casual_pants", "jeans", "shorts", "cargo_pants", "chinos", "khakis", "trousers", "slacks", "corduroy_pants", "denim_shorts", "dress_shorts", "cargo_shorts", "athletic_shorts", "swim_trunks"]},
                                    "color": {"type": "string", "description": "Color with prefix (e.g., 'dark blue', 'light green', 'navy blue')"},
                                    "pattern": {"type": "string", "description": "Solid, striped, plaid, etc."},
                                    "material": {"type": "string", "description": "Denim, cotton, wool, polyester, etc."},
                                    "fit": {"type": "string", "enum": ["tight", "fitted", "loose", "baggy"]}
                                },
                                "required": ["type", "color"]
                            },
                            "uniform_professional": {
                        "type": "object",
                        "properties": {
                                    "type": {"type": "string", "enum": ["military_uniform", "police_uniform", "medical_scrubs", "chef_uniform", "nurse_uniform", "pilot_uniform", "flight_attendant", "security_guard", "firefighter", "paramedic", "business_suit", "formal_suit", "academic_robe", "judge_robe", "clerical_robe", "lab_coat", "apron", "overalls", "coveralls", "boiler_suit", "cargo_uniform", "tactical_gear", "dress_uniform", "service_uniform", "work_uniform"]},
                                    "color": {"type": "string", "description": "Color with prefix (e.g., 'dark blue', 'light green', 'navy blue')"},
                                    "rank_insignia": {"type": "string", "description": "Rank, badges, patches, or insignia if applicable"},
                                    "material": {"type": "string", "description": "Cotton, polyester, wool, etc."},
                                    "condition": {"type": "string", "enum": ["pristine", "well_worn", "weathered", "tattered"]}
                                },
                                "required": ["type", "color"]
                            },
                            "outerwear": {
                        "type": "object",
                        "properties": {
                                    "type": {"type": "string", "enum": ["coat", "jacket", "raincoat", "blazer", "overcoat", "pea_coat", "hoodie", "cardigan", "vest", "windbreaker", "bomber_jacket", "leather_jacket", "denim_jacket", "suit_jacket", "sports_jacket"]},
                                    "color": {"type": "string", "description": "Color with prefix (e.g., 'dark blue', 'light green', 'navy blue')"},
                                    "material": {"type": "string", "description": "Leather, wool, denim, polyester, etc."},
                                    "fit": {"type": "string", "enum": ["tight", "fitted", "loose", "oversized"]}
                                }
                            }
                        },
                        "required": ["tops", "bottoms"]
                    },
                    "footwear": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["dress_shoes", "loafers", "oxfords", "sneakers", "boots", "ankle_boots", "work_boots", "hiking_boots", "sandals", "flip_flops", "moccasins", "boat_shoes", "wingtip_shoes", "chelsea_boots", "combat_boots", "running_shoes", "basketball_shoes", "tennis_shoes", "dress_boots", "casual_shoes"]},
                            "color": {"type": "string", "description": "Color with prefix (e.g., 'dark blue', 'light green', 'navy blue')"},
                            "material": {"type": "string", "description": "Leather, canvas, suede, rubber, etc."},
                            "style": {"type": "string", "description": "Casual, formal, athletic, etc."}
                        },
                        "required": ["type", "color"]
                    },
                    "accessories": {
                        "type": "array",
                        "description": "List of accessories worn by the character",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "description": "Type of accessory (e.g., 'glasses', 'hat', 'watch', 'jewelry', 'bag', 'gloves', 'tie', 'scarf', 'belt', 'piercings')"},
                                "description": {"type": "string", "description": "Detailed description including color, material, style, and any distinctive features"},
                                "location": {"type": "string", "description": "Where the accessory is worn (e.g., 'on head', 'around neck', 'on wrist', 'in hand', 'on face')"}
                            },
                            "required": ["type", "description"]
                        }
                    },
                    "overall_style": {
                        "type": "object",
                        "properties": {
                            "style_category": {"type": "string", "enum": ["casual", "formal", "business", "sporty", "elegant", "bohemian", "vintage", "modern", "streetwear", "preppy", "western", "athletic"]},
                            "formality_level": {"type": "string", "enum": ["very_casual", "casual", "smart_casual", "business_casual", "business_formal", "semi_formal", "formal", "black_tie"]},
                            "season": {"type": "string", "enum": ["summer", "winter", "spring", "autumn", "all_season"]}
                        },
                        "required": ["style_category", "formality_level", "season"]
                    }
                },
                "required": ["face", "clothing", "footwear", "overall_style"]
            },
            "strict": True
        }
    }


def _schema_character_summary() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "character_summary",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": CHARACTER_SUMMARY_CHARACTER_MIN,
                        "maxLength": CHARACTER_SUMMARY_CHARACTER_MAX,
                        "description": f"Entire character all details in short version."
                    }
                },
                "required": ["summary"]
            },
            "strict": True
        }
    }


def _schema_location() -> dict[str, object]:
    """JSON schema for essential location description - only absolutely necessary visual information."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "location_expansion",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "place": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "description": "Location (e.g., 'bedroom', 'forest', 'office', 'street')"},
                            "size": {"type": "string", "enum": ["tiny", "small", "medium", "large", "massive"]},
                            "style": {"type": "string", "description": "Visual style (e.g., 'modern', 'rustic', 'Victorian', 'industrial')"}
                        },
                        "required": ["type", "size"]
                    },
                    "lighting": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Object that provides light (e.g., 'sunlight', 'lamp', 'candle')"},
                            "color": {"type": "string", "enum": ["warm", "cool", "natural", "golden", "white", "dim"]},
                            "brightness": {"type": "string", "enum": ["bright", "moderate", "dim", "dark"]},
                            "time": {"type": "string", "enum": ["morning", "noon", "afternoon", "evening", "night"]}
                        },
                        "required": ["source", "brightness", "time"]
                    },
                    "ground": {
                        "type": "object",
                        "properties": {
                            "material": {"type": "string", "description": "Material that covers the ground (e.g., 'wood floor', 'grass', 'concrete')"},
                            "color": {"type": "string", "description": "Color with prefix (e.g., 'dark blue', 'light green', 'navy blue')"}
                        },
                        "required": ["material", "color"]
                    },
                    "walls_or_surroundings": {
                        "type": "object",
                        "properties": {
                            "material": {"type": "string", "description": "Material that walls/surroundings are made of (e.g., 'painted walls', 'trees', 'brick')"},
                            "color": {
                                "type": "string", "description": "Color with prefix (e.g., 'dark blue', 'light green', 'navy blue')"
                            }
                        },
                        "required": ["material", "color"]
                    },
                    "objects": {
                        "type": "array",
                        "description": "Visible objects in the scene - must include 15-20 detailed objects with hierarchical positioning. HIERARCHY: 1) Large objects (sofas, tables, trees) positioned relative to room/scene, 2) Medium objects (lamps, chairs) positioned relative to large objects, 3) Small objects (books, vases) positioned relative to medium objects",
                        "minItems": 15,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "What the object is (e.g., 'wooden chair', 'red lamp', 'oak tree')"},
                                "type": {"type": "string", "enum": ["furniture", "decoration", "plant", "window", "door", "lighting", "natural", "building", "vehicle", "other"]},
                                "color": {"type": "string", "description": "Color with prefix (e.g., 'dark blue', 'light green', 'navy blue')"},
                                "material": {"type": "string", "description": "What it's made of"},
                                "size": {"type": "string", "enum": ["tiny", "small", "medium", "large", "huge"]},
                                "position": {"type": "string", "description": "Hierarchical positioning: for large objects use room/scene references (e.g., 'center of room', 'left wall', 'back corner'), for small objects reference large objects (e.g., 'on the wooden table', 'next to the sofa', 'under the window')"},
                                "positioning_priority": {"type": "string", "enum": ["primary", "secondary", "tertiary"], "description": "Positioning hierarchy: 'primary' for large anchor objects, 'secondary' for medium objects, 'tertiary' for small decorative items"}
                            },
                            "required": ["name", "type", "color", "material", "size", "position", "positioning_priority"]
                        }
                    },
                    "atmosphere": {
                        "type": "object",
                        "properties": {
                            "weather": {"type": "string", "description": "Weather if visible (e.g., 'sunny', 'rainy', 'foggy', 'not visible')"},
                            "season": {"type": "string", "enum": ["spring", "summer", "autumn", "winter"]},
                            "mood": {"type": "string", "description": "Visual feeling (e.g., 'cozy', 'dramatic', 'peaceful', 'busy')"}
                        },
                        "required": ["mood"]
                    }
                },
                "required": ["place", "lighting", "ground", "walls_or_surroundings", "objects", "atmosphere"]
            },
            "strict": True
        }
    }


def _schema_location_summary() -> dict[str, object]:
    """JSON schema for location summary description."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "location_summary",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": LOCATION_SUMMARY_CHARACTER_MIN,
                        "maxLength": LOCATION_SUMMARY_CHARACTER_MAX,
                        "description": f"Entire location with all details in short version."
                    }
                },
                "required": ["summary"]
            },
            "strict": True
        }
    }

def _schema_story_summary() -> dict[str, object]:
    """JSON schema for story description."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "story_summary",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": STORY_DESCRIPTION_CHARACTER_MIN,
                        "maxLength": STORY_DESCRIPTION_CHARACTER_MAX,
                        "description": f"Entire story all details in short version."
                    }
                },
                "required": ["summary"]
            },
            "strict": True
        }
    }

def _build_character_system_prompt() -> str:
    return (
        f"Create detailed visual-only attributes and characteristics for the character based its role in story for AI image generation.\n\n"
    )

def _build_character_user_prompt(story_desc: str, character_name: str, all_characters: dict[str, str]) -> str:
    return (
        f"Story: {story_desc}\n"
        f"Character: {character_name}\n"
    )

def _build_character_summary_prompt() -> str:
    return (
        f"Transform it into a continuous paragraph of {CHARACTER_SUMMARY_CHARACTER_MIN}-{CHARACTER_SUMMARY_CHARACTER_MAX} characters, approximately {CHARACTER_SUMMARY_WORD_MIN}-{CHARACTER_SUMMARY_WORD_MAX} words.\n"
        f"It must always include all visual details from the original description, preserving all visual attributes and characteristics.\n"
    )

def _build_character_summary_user_prompt(character_name: str, detailed_description: str) -> str:
    return (
        f"Character: {character_name}\n\n"
        f"Original description: {detailed_description}"
    )


def _build_story_summary_prompt() -> str:
    return (
        f"Transform it into a continuous paragraph of {STORY_DESCRIPTION_CHARACTER_MIN}-{STORY_DESCRIPTION_CHARACTER_MAX} characters, approximately {STORY_DESCRIPTION_WORD_MIN}-{STORY_DESCRIPTION_WORD_MAX} words.\n"
        f"It must always include all actors and their roles, all locations and settings, complete chronological events in details.\n"
    )

def _build_story_summary_user_prompt(story_content: str) -> str:
    """Extract only dialogue lines from story content using existing regex"""
    lines = story_content.split('\n')
    dialogue_lines = []
    
    for line in lines:
        # Use existing dialogue regex pattern to match dialogue lines
        if _DIALOGUE_RE.match(line.strip()):
            # Remove brackets, braces, and parentheses from the line
            cleaned_line = line.replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('(', '').replace(')', '')
            dialogue_lines.append(cleaned_line)
    
    dialogue_content = '\n'.join(dialogue_lines)
    return (
        f"Story content: {dialogue_content}"
    )

def _build_location_system_prompt() -> str:
    return (
        f"Create a detailed location that includes all possible object that can be seen in such type of location, postioning large objects relative to the room, medium objects relative to large ones, small items relative to medium objects, for AI image generation.\n\n"
    )

def _build_location_user_prompt(story_desc: str, location_id: str, all_locations: dict[str, str]) -> str:
    return (
        f"Story: {story_desc}\n"
        f"Location: {all_locations[location_id]}\n"
    )


def _build_location_summary_prompt() -> str:
    return (
        f"Transform it into a continuous paragraph of {LOCATION_SUMMARY_CHARACTER_MIN}-{LOCATION_SUMMARY_CHARACTER_MAX} characters, approximately {LOCATION_SUMMARY_WORD_MIN}-{LOCATION_SUMMARY_WORD_MAX} words.\n"
        f"It must always include all visual details from the original description, preserving all visual attributes, characteristics and postioning relationships.\n"
    )

def _build_location_summary_user_prompt(location_id: str, detailed_description: str) -> str:
    return (
        f"Location: {location_id}\n\n"
        f"Original description: {detailed_description}"
    )


def _call_lm_studio(system_prompt: str, user_prompt: str, lm_studio_url: str, model: str, response_format: dict[str, object] | None = None, temperature: float = 1.0) -> str:
    headers = {"Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt + " \n/no_think"}, {"role": "user", "content": user_prompt + "\n/no_think"}]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    
    if response_format is not None:
        payload["response_format"] = response_format
    
    resp = requests.post(f"{lm_studio_url}/chat/completions", headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"LM Studio API error: {resp.status_code} {resp.text}")
    data = resp.json()
    if not data.get("choices"):
        raise RuntimeError("LM Studio returned no choices")

    return data["choices"][0]["message"]["content"]


def _parse_structured_response(content: str) -> dict[str, object] | None:
    """Parse structured JSON response, handling code blocks if present."""
    text = content.strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def _validate_character_count(text: str, min_chars: int, max_chars: int) -> bool:
    """Validate that text meets character count requirements."""
    char_count = len(text)
    word_count = len(text.split())
    
    if char_count < min_chars:
        print(f"WARNING: Summary too short: {char_count} characters (minimum: {min_chars})")
        return False
    elif char_count > max_chars:
        print(f"WARNING: Summary too long: {char_count} characters (maximum: {max_chars})")
        return False
    
    print(f"✓ Summary within limits: {char_count} characters, {word_count} words")
    return True


def _recursively_format_character_data(data: object, prefix: str = "", sentences: list = None) -> list[str]:
    """
    Recursively parse any JSON structure and convert it to descriptive sentences.
    This generic parser can handle any nested structure without hardcoded keys.
    """
    if sentences is None:
        sentences = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            # Skip certain keys that don't contribute to physical description
            if key.lower() in ['shared_elements', 'professional_relationships', 'relationships']:
                continue
                
            if isinstance(value, (dict, list)):
                # Recursively process nested structures
                _recursively_format_character_data(value, f"{prefix}{key}_" if prefix else f"{key}_", sentences)
            elif isinstance(value, str) and value.strip():
                # Convert key-value pairs to descriptive sentences
                formatted_key = key.replace('_', ' ').title()
                sentences.append(f"{formatted_key.lower()}: {value},")
    
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _recursively_format_character_data(item, prefix, sentences)
            elif isinstance(item, str) and item.strip():
                sentences.append(f"{item},")
    
    return sentences


def _format_structured_data(data: dict[str, object], data_type: str = "character") -> str:
    """
    Generic function to convert any structured data to readable description format.
    Works for both character and location data.
    
    Example:
    {
        "head": {
            "color": "brown",
            "shape": {
                "upper": "round",
                "lower": "square"
            }
        }
    }
    
    Returns: "Head: (Color: brown, Shape: (Upper: round, Lower: square))"
    """
    try:
        formatted_text = _recursively_format_generic_data(data)
        if not formatted_text.strip():
            # Fallback message based on data type
            if data_type == "location":
                return "The location has distinctive visual features."
            else:
                return "The character has a distinctive appearance with unique physical features."
        
        return formatted_text
    except Exception as ex:
        print(f"WARNING: Failed to format {data_type} data: {ex}")
        if data_type == "location":
            return "The location has distinctive visual features."
        else:
            return "The character has a distinctive appearance with unique physical features."


def _recursively_format_generic_data(data: dict | list, level: int = 0) -> str:
    """
    Recursively format nested data structure into readable text.
    Handles nested dictionaries and arrays.
    """
    if isinstance(data, dict):
        parts = []
        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, dict):
                # Nested dictionary: Key: (nested content)
                nested_content = _recursively_format_generic_data(value, level + 1)
                parts.append(f"{formatted_key}: ({nested_content})")
            elif isinstance(value, list):
                # Array: Key: [item1, item2, ...]
                if value:
                    if isinstance(value[0], dict):
                        # Array of objects
                        items = []
                        for item in value:
                            item_content = _recursively_format_generic_data(item, level + 1)
                            items.append(f"({item_content})")
                        parts.append(f"{formatted_key}: [{', '.join(items)}]")
                    else:
                        # Array of primitives
                        parts.append(f"{formatted_key}: [{', '.join(str(v) for v in value)}]")
            elif isinstance(value, str) and value.strip():
                # Simple key-value pair
                parts.append(f"{formatted_key}: {value}")
            elif value is not None:
                # Other types (numbers, booleans, etc.)
                parts.append(f"{formatted_key}: {value}")
        
        return ", ".join(parts)
    
    elif isinstance(data, list):
        items = []
        for item in data:
            if isinstance(item, dict):
                item_content = _recursively_format_generic_data(item, level + 1)
                items.append(f"({item_content})")
            else:
                items.append(str(item))
        return ", ".join(items)
    
    else:
        return str(data)


def _format_character_description(char_data: dict[str, object]) -> str:
    """
    Convert any structured character data to readable description format using generic recursive parsing.
    This unified function works for both initial character generation and character rewrites.
    """
    return _format_structured_data(char_data, data_type="character")


def _generate_structured_descriptions(
    items: list[str] | dict[str, str],
    story_desc: str,
    lm_studio_url: str,
    model: str,
    schema_func,
    prompt_func,
    user_prompt_func,
    format_func,
    item_type: str,
    resumable_state: ResumableState | None = None,
    get_cached_func=None,
    set_cached_func=None,
    is_complete_func=None
) -> dict[str, str]:
    """
    Common function to generate structured descriptions for both characters and locations.
    
    Args:
        items: List of items (characters) or dict of items (locations with original descriptions)
        story_desc: Story description for context
        lm_studio_url: LM Studio URL
        model: Model to use
        schema_func: Function that returns the schema
        prompt_func: Function to build the prompt
        user_prompt_func: Function to build the user prompt
        format_func: Function to format structured data
        item_type: Type of item ("character" or "location")
        resumable_state: Resumable state manager
        get_cached_func: Function to get cached description
        set_cached_func: Function to set cached description
        is_complete_func: Function to check if item is complete
    """
    item_to_desc: dict[str, str] = {}
    
    # Convert items to dict if it's a list
    if isinstance(items, list):
        items_dict = {item: item for item in items}
    else:
        items_dict = items
    
    total = len(items_dict)
    print(f"Generating structured {item_type} descriptions: {total} total")
    
    for idx, (item_id, item_value) in enumerate(items_dict.items(), 1):
        # Check if resumable and already complete
        if resumable_state and is_complete_func and is_complete_func(item_id):
            cached_desc = get_cached_func(item_id) if get_cached_func else None
            if cached_desc:
                print(f"({idx}/{total}) {item_id}: using cached description from checkpoint")
                item_to_desc[item_id] = cached_desc
                continue
        
        print(f"({idx}/{total}) {item_id}: generating structured description...")


        print(f"Prompt: {items}")

        prompt = prompt_func()
        user_prompt = user_prompt_func(story_desc, item_id, items)
        
        try:
            raw = _call_lm_studio(prompt, user_prompt, lm_studio_url, model, schema_func())
            structured_data = _parse_structured_response(raw)
            
            if not structured_data:
                raise RuntimeError("Failed to parse structured response")
            
            # Convert structured data to readable description
            desc = format_func(structured_data)
            if not desc:
                raise RuntimeError("Empty description generated")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state and set_cached_func:
                set_cached_func(item_id, desc)
                
        except Exception as ex:
            print(f"({idx}/{total}) {item_id}: FAILED - {ex}")
            print(f"ERROR: Failed to generate description for '{item_id}': {ex}")
            raise
            
        item_to_desc[item_id] = desc
        print(f"({idx}/{total}) {item_id}: done ({len(desc.split())} words)")
        
    return item_to_desc


def _generate_character_descriptions(story_desc: str, characters: list[str], lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    """Generate structured character descriptions using the common function."""
    return _generate_structured_descriptions(
        items=characters,
        story_desc=story_desc,
        lm_studio_url=lm_studio_url,
        model=MODEL_CHARACTER_GENERATION,
        schema_func=_schema_character,
        prompt_func=_build_character_system_prompt,
        user_prompt_func=_build_character_user_prompt,
        format_func=_format_character_description,
        item_type="character",
        resumable_state=resumable_state,
        get_cached_func=resumable_state.get_character_description if resumable_state else None,
        set_cached_func=resumable_state.set_character_description if resumable_state else None,
        is_complete_func=resumable_state.is_character_complete if resumable_state else None
    )

def _generate_character_summaries(name_to_desc: dict[str, str], lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    """Generate character summaries using the common function."""
    return _generate_summaries(
        item_to_desc=name_to_desc,
        lm_studio_url=lm_studio_url,
        model=MODEL_CHARACTER_SUMMARY,
        schema_func=_schema_character_summary,
        user_prompt_func=_build_character_summary_user_prompt,
        prompt_func=_build_character_summary_prompt,
        item_type="character",
        resumable_state=resumable_state,
        get_cached_func=resumable_state.get_character_summary if resumable_state else None,
        set_cached_func=resumable_state.set_character_summary if resumable_state else None,
        is_complete_func=resumable_state.is_character_summary_complete if resumable_state else None,
        temperature=0.1
    )


def _generate_summaries(
    item_to_desc: dict[str, str],
    lm_studio_url: str,
    model: str,
    schema_func,
    prompt_func,
    user_prompt_func,
    item_type: str,
    resumable_state: ResumableState | None = None,
    get_cached_func=None,
    set_cached_func=None,
    is_complete_func=None,
    temperature: float = 0.1
) -> dict[str, str]:
    """
    Common function to generate summaries for both characters and locations.
    """
    item_to_summary: dict[str, str] = {}
    total = len(item_to_desc)
    print(f"Generating {item_type} summaries: {total} total")
    
    for idx, (item_id, detailed_desc) in enumerate(item_to_desc.items(), 1):
        if resumable_state and is_complete_func and is_complete_func(item_id):
            cached_summary = get_cached_func(item_id) if get_cached_func else None
            if cached_summary:
                print(f"({idx}/{total}) {item_id}: using cached summary from checkpoint")
                item_to_summary[item_id] = cached_summary
                continue
        
        print(f"({idx}/{total}) {item_id}: generating summary...")
        prompt = prompt_func()
        user_prompt = user_prompt_func(item_id, detailed_desc)
        
        try:
            raw = _call_lm_studio(prompt, user_prompt, lm_studio_url, model, schema_func(), temperature=temperature)
            structured_data = _parse_structured_response(raw)
            
            if not structured_data:
                raise RuntimeError("Failed to parse structured summary response")
            
            summary = structured_data.get("summary", "").strip()
            if not summary:
                raise RuntimeError("Empty summary generated")
            
            if resumable_state and set_cached_func:
                set_cached_func(item_id, summary)
                
        except Exception as ex:
            print(f"({idx}/{total}) {item_id}: FAILED - {ex}")
            print(f"ERROR: Failed to generate summary for '{item_id}': {ex}")
            raise
            
        item_to_summary[item_id] = summary
        print(f"({idx}/{total}) {item_id}: done ({len(summary.split())} words)")

        _validate_character_count(summary, LOCATION_SUMMARY_CHARACTER_MIN, LOCATION_SUMMARY_CHARACTER_MAX)
        
    return item_to_summary


def _generate_story_summary(story_content: str, lm_studio_url: str, resumable_state: ResumableState | None = None) -> str:
    """Generate story description from story content using LLM."""
    # Check if resumable and already complete
    if resumable_state and resumable_state.is_story_summary_complete():
        cached_desc = resumable_state.get_story_summary()
        if cached_desc:
            print("Using cached story description from checkpoint")
            return cached_desc
    
    print("Generating story description from dialogue content...")
    prompt = _build_story_summary_prompt()
    user_prompt = _build_story_summary_user_prompt(story_content)
    
    try:
        # Use model constant for story description generation
        model = MODEL_STORY_DESCRIPTION
        # Call with structured output using the story description schema
        raw = _call_lm_studio(prompt, user_prompt, lm_studio_url, model, _schema_story_summary())
        structured_data = _parse_structured_response(raw)

        
        if not structured_data:
            raise RuntimeError("Failed to parse structured story description response")
        
        story_desc = structured_data.get("summary", "").strip()
        if not story_desc:
            raise RuntimeError("Empty story description generated")

        _validate_character_count(story_desc, STORY_DESCRIPTION_CHARACTER_MIN, STORY_DESCRIPTION_CHARACTER_MAX)
        
        # Save to checkpoint if resumable mode enabled
        if resumable_state:
            resumable_state.set_story_summary(story_desc)
            print("Saved story description to checkpoint")
            
    except Exception as ex:
        print(f"ERROR: Failed to generate story description: {ex}")
        raise
        
    print(f"Generated story description: {story_desc}")
    return story_desc

def _generate_location_descriptions(story_desc: str, locations: dict[str, str], lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    """Generate structured location descriptions using the common function."""
    return _generate_structured_descriptions(
        items=locations,
        story_desc=story_desc,
        lm_studio_url=lm_studio_url,
        model=MODEL_LOCATION_EXPANSION,
        schema_func=_schema_location,
        prompt_func=_build_location_system_prompt,
        user_prompt_func=_build_location_user_prompt,
        format_func=_format_location_description,
        item_type="location",
        resumable_state=resumable_state,
        get_cached_func=resumable_state.get_location_description if resumable_state else None,
        set_cached_func=resumable_state.set_location_description if resumable_state else None,
        is_complete_func=resumable_state.is_location_complete if resumable_state else None
    )


def _format_location_description(location_data: dict[str, object]) -> str:
    """
    Convert any structured location data to readable description format using generic recursive parsing.
    This unified function works for both initial location generation and location summaries.
    """
    return _format_structured_data(location_data, data_type="location")

def _generate_location_summaries(location_id_to_desc: dict[str, str], lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    """Generate location summaries using the common function."""
    return _generate_summaries(
        item_to_desc=location_id_to_desc,
        lm_studio_url=lm_studio_url,
        model=MODEL_LOCATION_EXPANSION,
        schema_func=_schema_location_summary,
        prompt_func=_build_location_summary_prompt,
        user_prompt_func=_build_location_summary_user_prompt,
        item_type="location",
        resumable_state=resumable_state,
        get_cached_func=resumable_state.get_location_summary if resumable_state else None,
        set_cached_func=resumable_state.set_location_summary if resumable_state else None,
        is_complete_func=resumable_state.is_location_summary_complete if resumable_state else None,
        temperature=0.1
    )


def _validate_and_write_character_file(name_to_desc: dict[str, str], out_path: str, expected_names: set[str]) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Write in strict format: ((name)): description\n\n
    with open(out_path, "w", encoding="utf-8") as f:
        for name in sorted(name_to_desc.keys()):
            desc = _sanitize_single_paragraph(name_to_desc[name])
            f.write(f"(({name})): {desc}\n\n")

    # Re-validate file format and name set
    errors = 0
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as ex:
        print(f"ERROR: Failed to re-read character file: {ex}")
        return 1

    lines = [ln for ln in content.splitlines() if ln.strip()]
    seen_names: set[str] = set()
    for ln in lines:
        m = _CHARACTER_LINE_RE.match(ln)
        if not m:
            errors += 1
            print(f"ERROR: Invalid character line format: {ln}")
            continue
        nm, desc = m.group(1).strip(), m.group(2).strip()
        if not nm or not desc:
            errors += 1
            print(f"ERROR: Missing name or description in line: {ln}")
            continue
        seen_names.add(nm)

    if seen_names != expected_names:
        errors += 1
        print("ERROR: Character names in output do not match detected names.")
        print(f"  Expected ({len(expected_names)}): {sorted(expected_names)}")
        print(f"  Found    ({len(seen_names)}): {sorted(seen_names)}")
        only_expected = sorted(expected_names - seen_names)
        only_found = sorted(seen_names - expected_names)
        if only_expected:
            print(f"  Missing in output: {only_expected}")
        if only_found:
            print(f"  Unexpected in output: {only_found}")

    if errors == 0:
        print(f"Wrote {len(seen_names)} character descriptions to: {out_path}")

    return errors


def _validate_and_write_character_summary_file(name_to_summary: dict[str, str], out_path: str, expected_names: set[str]) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Write in strict format: ((name)): summary\n\n
    with open(out_path, "w", encoding="utf-8") as f:
        for name in sorted(name_to_summary.keys()):
            summary = _sanitize_single_paragraph(name_to_summary[name])
            f.write(f"(({name})): {summary}\n\n")

    # Re-validate file format and name set
    errors = 0
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as ex:
        print(f"ERROR: Failed to re-read character summary file: {ex}")
        return 1

    lines = [ln for ln in content.splitlines() if ln.strip()]
    seen_names: set[str] = set()
    for ln in lines:
        m = _CHARACTER_LINE_RE.match(ln)
        if not m:
            errors += 1
            print(f"ERROR: Invalid character summary line format: {ln}")
            continue
        nm, summary = m.group(1).strip(), m.group(2).strip()
        if not nm or not summary:
            errors += 1
            print(f"ERROR: Missing name or summary in line: {ln}")
            continue
        seen_names.add(nm)

    if seen_names != expected_names:
        errors += 1
        print("ERROR: Character names in summary output do not match detected names.")
        print(f"  Expected ({len(expected_names)}): {sorted(expected_names)}")
        print(f"  Found    ({len(seen_names)}): {sorted(seen_names)}")
        only_expected = sorted(expected_names - seen_names)
        only_found = sorted(seen_names - expected_names)
        if only_expected:
            print(f"  Missing in output: {only_expected}")
        if only_found:
            print(f"  Unexpected in output: {only_found}")

    if errors == 0:
        print(f"Wrote {len(seen_names)} character summaries to: {out_path}")

    return errors


def write_dialogues_file_from_pairs(pairs, out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            d = p.get("dialogue")
            if not d:
                continue
            character = d["data"].get("character")
            dialogue = d["data"].get("dialogue")
            if not character or not dialogue:
                continue
            f.write(f"[{character}] {dialogue}\n\n")
            written += 1
    return written


def write_locations_file(locations: dict[str, str], out_path: str) -> int:
    """Write locations to file in format {{loc_id}} description..."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for loc_id in sorted(locations.keys()):
            description = locations[loc_id].strip()
            f.write(f"{{{{{loc_id}}}}} {description}\n\n")
            written += 1
    return written


def write_scenes_file_from_pairs(pairs, out_path: str, locations: dict[str, str] = None) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            s = p.get("scene")
            if not s:
                continue
            
            # Get the raw scene text
            scene_text = s['raw']
            
            # Replace any location references with descriptions to simple format
            # Convert {{loc_1, description}} to {{loc_1}}
            scene_text = _LOCATION_FULL_RE.sub(r'{{\1}}', scene_text)
            
            # Write the scene with simplified location references
            f.write(f"{scene_text}\n\n")
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse story file and generate character descriptions")
    parser.add_argument("--bypass-validation", action="store_true", 
                       help="Skip character consistency and scene ID continuity validation")
    parser.add_argument("--force-start", action="store_true",
                       help="Force start from beginning, ignoring any existing checkpoint files")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    audio_story_path = os.path.normpath(os.path.join(base_dir, "../../gen.audio/input/1.story.txt"))
    image_story_path = os.path.normpath(os.path.join(base_dir, "../input/1.story.txt"))
    scene_out_path = os.path.normpath(os.path.join(base_dir, "../input/3.scene.txt"))
    characters_out_path = os.path.normpath(os.path.join(base_dir, "../input/2.character.txt"))
    character_summary_out_path = os.path.normpath(os.path.join(base_dir, "../input/3.character.txt"))
    locations_out_path = os.path.normpath(os.path.join(base_dir, "../input/2.location.txt"))
    location_summary_out_path = os.path.normpath(os.path.join(base_dir, "../input/3.location.txt"))
    
    # Initialize resumable state if enabled
    resumable_state = None
    if ENABLE_RESUMABLE_MODE:
        checkpoint_dir = os.path.normpath(os.path.join(base_dir, "../output/tracking"))
        script_name = Path(__file__).stem  # Automatically get script name without .py extension
        resumable_state = ResumableState(checkpoint_dir, script_name, args.force_start)
        print(f"Resumable mode enabled - checkpoint directory: {checkpoint_dir}")
        if resumable_state.state_file.exists():
            print(f"Found existing checkpoint: {resumable_state.state_file}")
            print(resumable_state.get_progress_summary())
        else:
            print("No existing checkpoint found - starting fresh")

    content = read_file_text(image_story_path)
    if content is None:
        print("Skipping parsing due to missing image story file.")
        return 1

    # Extract locations from content
    locations = _extract_locations_from_content(content)
    print(f"Extracted {len(locations)} unique locations: {sorted(locations.keys())}")
    
    tokens = _tokenize(content)
    if not tokens:
        print("No recognizable lines found (no [] or ()(()) pattern). Nothing to do.")
        # Still write empty outputs to clear previous runs
        write_dialogues_file_from_pairs([], audio_story_path)
        write_scenes_file_from_pairs([], scene_out_path)
        return 0

    order = _detect_order(tokens)
    if order is None:
        print("Could not detect order. Expecting a file-wide pattern of [] then ()(()) or the reverse.")
        return 1

    if order == ("dialogue", "scene"):
        print("Detected order: [] first, then ()(())")
    else:
        print("Detected order: ()(()) first, then []")

    pairs = _pair_tokens(tokens, order)

    # PRE-VALIDATION: Validate characters and scenes early - exit if validation fails
    if not args.bypass_validation:
        print("Running pre-validation checks...")
        validation_errors = _validate_character_consistency(tokens, pairs)
        if validation_errors:
            print(f"PRE-VALIDATION FAILED: {validation_errors} character error(s). Exiting.")
            return 1

        # Validate scene id continuity (e.g., 1.2, 1.3, 1.5 -> missing 1.4)
        gap_errors = _validate_scene_id_continuity(tokens)
        if gap_errors:
            print(f"PRE-VALIDATION FAILED: {gap_errors} missing scene id(s). Exiting.")
            return 1
        
        print("Pre-validation passed. Proceeding with generation...")
    else:
        print("Validation bypassed via --bypass-validation flag")

    # Generate story description from content and expand locations using LLM
    story_desc = None
    if locations:
        try:
            lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
            
            # Generate story description from content using qwen/qwen3-14b
            story_desc = _generate_story_summary(content, lm_studio_url, resumable_state)
            
            # Generate structured location descriptions
            expanded_locations = _generate_location_descriptions(story_desc, locations, lm_studio_url, resumable_state)
            
            # Write initial detailed location descriptions to 2.location.txt
            written_locations = write_locations_file(expanded_locations, locations_out_path)
            print(f"Wrote {written_locations} detailed location entries to: {locations_out_path}")
            
            # Generate location summaries from expanded descriptions
            location_summaries = _generate_location_summaries(expanded_locations, lm_studio_url, resumable_state)
            
            # Write location summaries to 3.location.txt
            written_summaries = write_locations_file(location_summaries, location_summary_out_path)
            print(f"Wrote {written_summaries} location summary entries to: {location_summary_out_path}")
            
            # Use expanded locations for scene processing
            locations = expanded_locations
        except Exception as ex:
            print(f"WARNING: Location expansion failed: {ex}")
            print("Using original location descriptions...")
            written_locations = write_locations_file(locations, locations_out_path)
            print(f"Wrote {written_locations} original location entries to: {locations_out_path}")
    else:
        # Write empty locations file if no locations found
        written_locations = write_locations_file({}, locations_out_path)
        print(f"Wrote {written_locations} location entries to: {locations_out_path}")

    # Generate per-character visual descriptions using LM Studio and write 2.character.txt
    if not story_desc:
        print("Skipping character description generation due to missing story description")
    else:
        dchars, schars = _collect_unique_character_sets(tokens)
        unique_names = sorted(schars)
        if unique_names:
            try:
                lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
                
                # Step 1: Generate initial character descriptions
                name_to_desc = _generate_character_descriptions(story_desc, unique_names, lm_studio_url, resumable_state)
                
                # Step 3: Write the main character file with final descriptions
                char_errors = _validate_and_write_character_file(name_to_desc, characters_out_path, set(unique_names))
                if char_errors:
                    print(f"Character description output had {char_errors} error(s).")
                    return 1
                
                # Step 4: Generate character summaries from final descriptions
                name_to_summary = _generate_character_summaries(name_to_desc, lm_studio_url, resumable_state)
                summary_errors = _validate_and_write_character_summary_file(name_to_summary, character_summary_out_path, set(unique_names))
                if summary_errors:
                    print(f"Character summary output had {summary_errors} error(s).")
                    return 1
            except Exception as ex:
                print(f"ERROR: Character description generation failed: {ex}")
                return 1
        else:
            # Clear files if no characters
            _validate_and_write_character_file({}, characters_out_path, set())
            _validate_and_write_character_summary_file({}, character_summary_out_path, set())

    written_dialogues = write_dialogues_file_from_pairs(pairs, audio_story_path)
    print(f"Wrote {written_dialogues} dialogue lines to: {audio_story_path}")

    written_scenes = write_scenes_file_from_pairs(pairs, scene_out_path, locations)
    print(f"Wrote {written_scenes} scene entries to: {scene_out_path}")

    # Per-pair completeness check (both parts present)
    for p in pairs:
        idx = p["pair_index"]
        if not p.get("dialogue"):
            print(f"PAIR {idx}: Missing [] line.")
        if not p.get("scene"):
            print(f"PAIR {idx}: Missing ()(()) line.")

    # Clean up checkpoint files if resumable mode was used and everything completed successfully
    if resumable_state:
        print("All operations completed successfully")
        print("Final progress:", resumable_state.get_progress_summary())
        resumable_state.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
