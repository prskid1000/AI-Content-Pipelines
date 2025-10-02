import os
import re
import json
import requests
import argparse
from functools import partial
import builtins as _builtins
import time
from pathlib import Path


CHARACTER_SUMMARY_CHARACTER_COUNT = 1200
LOCATION_SUMMARY_CHARACTER_COUNT = 3600
STORY_DESCRIPTION_CHARACTER_COUNT = 16000

# Feature flags
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them
ENABLE_THINKING = True  # Set to True to enable thinking in LM Studio responses

# Model constants for easy switching
MODEL_STORY_DESCRIPTION = "qwen/qwen3-14b"  # Model for generating story descriptions
MODEL_CHARACTER_GENERATION = "qwen/qwen3-14b"  # Model for character description generation
MODEL_CHARACTER_SUMMARY = "qwen/qwen3-14b"  # Model for character summary generation
MODEL_LOCATION_EXPANSION = "qwen/qwen3-14b"  # Model for location expansion

ART_STYLE = "Realistic Anime"

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
            "story_description": {"completed": False, "result": None},
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
    
    def is_story_description_complete(self) -> bool:
        """Check if story description generation is complete."""
        return self.state["story_description"]["completed"]
    
    def get_story_description(self) -> str | None:
        """Get cached story description if available."""
        return self.state["story_description"]["result"]
    
    def set_story_description(self, description: str):
        """Set story description and mark as complete."""
        self.state["story_description"]["completed"] = True
        self.state["story_description"]["result"] = description
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
        story_done = "✓" if self.is_story_description_complete() else "✗"
        locations_done = len(self.state["locations"]["completed"])
        locations_total = len(self.state["locations"]["results"]) + len([k for k in self.state["locations"]["results"].keys() if k not in self.state["locations"]["completed"]])
        characters_done = len(self.state["characters"]["completed"])
        characters_total = len(self.state["characters"]["results"]) + len([k for k in self.state["characters"]["results"].keys() if k not in self.state["characters"]["completed"]])
        summaries_done = len(self.state["character_summaries"]["completed"])
        summaries_total = len(self.state["character_summaries"]["results"]) + len([k for k in self.state["character_summaries"]["results"].keys() if k not in self.state["character_summaries"]["completed"]])
        location_summaries_done = len(self.state["location_summaries"]["completed"])
        location_summaries_total = len(self.state["location_summaries"]["results"]) + len([k for k in self.state["location_summaries"]["results"].keys() if k not in self.state["location_summaries"]["completed"]])
        
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


def _extract_dialogue_only_from_content(content: str) -> str:
    """Extract only dialogue lines from story content, excluding scene descriptions."""
    lines = content.split('\n')
    dialogue_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is dialogue (starts with [character])
        if line.startswith('[') and ']' in line:
            # Extract character and dialogue content
            m = _DIALOGUE_RE.match(line)
            if m:
                character = m.group(1).strip()
                dialogue = m.group(2).strip()
                if character and dialogue:
                    dialogue_lines.append(f"[{character}] {dialogue}")
    
    return '\n'.join(dialogue_lines)


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


def _replace_location_references(text: str, locations: dict[str, str]) -> str:
    """Replace {{loc_id}} and {{loc_id, ...}} with full descriptions in text."""
    def replace_func(match):
        full_match = match.group(0)
        
        # Try full location pattern first {{loc_id, description}}
        full_loc_match = _LOCATION_FULL_RE.match(full_match)
        if full_loc_match:
            loc_id = full_loc_match.group(1).strip()
            if loc_id in locations:
                return f"{{{{{locations[loc_id]}}}}}"
            return full_match
        
        # Try simple reference pattern {{loc_id}}
        ref_match = _LOCATION_REF_RE.match(full_match)
        if ref_match:
            loc_id = ref_match.group(1).strip()
            if loc_id in locations:
                return f"{{{{{locations[loc_id]}}}}}"
            return full_match
        
        return full_match
    
    # Replace all location patterns
    result = _LOCATION_FULL_RE.sub(replace_func, text)
    result = _LOCATION_REF_RE.sub(replace_func, result)
    
    return result


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
    """JSON schema for simplified character description focusing on face, hair, eyes, skin, and clothing details."""
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
                                    "type": {"type": "string", "enum": ["shirt", "blouse", "t-shirt", "sweater", "jacket", "tank_top", "cardigan", "blazer", "bustier", "camisole", "cape", "corset", "hoodie", "polo", "turtleneck"]},
                                    "color": {"type": "string", "description": "Primary color of the top"},
                                    "pattern": {"type": "string", "description": "Solid, striped, plaid, floral, etc."},
                                    "material": {"type": "string", "description": "Cotton, silk, wool, polyester, etc."},
                                    "fit": {"type": "string", "enum": ["tight", "fitted", "loose", "oversized"]}
                                },
                                "required": ["type", "color"]
                            },
                            "bottoms": {
                        "type": "object",
                        "properties": {
                                    "type": {"type": "string", "enum": ["pants", "shorts", "skirt", "trousers", "jeans", "jumpsuit", "leggings", "cargo_pants", "dress_pants"]},
                                    "color": {"type": "string", "description": "Primary color of the bottom"},
                                    "pattern": {"type": "string", "description": "Solid, striped, plaid, etc."},
                                    "material": {"type": "string", "description": "Denim, cotton, wool, etc."},
                                    "fit": {"type": "string", "enum": ["tight", "fitted", "loose", "baggy"]}
                                },
                                "required": ["type", "color"]
                            },
                            "uniform_professional": {
                        "type": "object",
                        "properties": {
                                    "type": {"type": "string", "enum": ["military_uniform", "police_uniform", "medical_scrubs", "chef_uniform", "nurse_uniform", "pilot_uniform", "flight_attendant", "security_guard", "firefighter", "paramedic", "business_suit", "formal_suit", "academic_robe", "judge_robe", "clerical_robe", "lab_coat", "apron", "overalls", "coveralls", "boiler_suit", "cargo_uniform", "tactical_gear", "dress_uniform", "service_uniform", "work_uniform"]},
                                    "color": {"type": "string", "description": "Primary color of the uniform"},
                                    "rank_insignia": {"type": "string", "description": "Rank, badges, patches, or insignia if applicable"},
                                    "material": {"type": "string", "description": "Cotton, polyester, wool, etc."},
                                    "condition": {"type": "string", "enum": ["pristine", "well_worn", "weathered", "tattered"]}
                                },
                                "required": ["type", "color"]
                            },
                            "outerwear": {
                        "type": "object",
                        "properties": {
                                    "type": {"type": "string", "enum": ["coat", "jacket", "raincoat", "blazer", "overcoat", "pea_coat", "gown", "hoodie", "cardigan", "vest"]},
                                    "color": {"type": "string", "description": "Primary color of the outerwear"},
                                    "material": {"type": "string", "description": "Leather, wool, denim, etc."},
                                    "fit": {"type": "string", "enum": ["tight", "fitted", "loose", "oversized"]}
                                }
                            }
                        },
                        "required": ["tops", "bottoms"]
                    },
                    "footwear": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["sneakers", "boots", "heels", "flats", "sandals", "loafers", "oxfords", "pumps", "ankle_boots", "knee_high_boots", "stilettos", "wedges", "ballet_flats", "moccasins", "slippers"]},
                            "color": {"type": "string", "description": "Primary color of the footwear"},
                            "material": {"type": "string", "description": "Leather, canvas, suede, rubber, etc."},
                            "style": {"type": "string", "description": "Casual, formal, athletic, etc."}
                        },
                        "required": ["type", "color"]
                    },
                    "accessories": {
                        "type": "object",
                        "properties": {
                            "glasses": {
                        "type": "object",
                        "properties": {
                                    "type": {"type": "string", "enum": ["reading_glasses", "sunglasses", "prescription_glasses", "safety_glasses", "aviator", "cat_eye", "round", "square", "rectangular", "rimless", "bifocal", "transitional"]},
                                    "color": {"type": "string", "description": "Frame color"},
                                    "material": {"type": "string", "description": "Metal, plastic, acetate, titanium, etc."}
                                }
                            },
                            "tie": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["necktie", "bow_tie", "bolo_tie", "ascot", "string_tie", "clip_on", "skinny_tie", "wide_tie"]},
                                    "color": {"type": "string", "description": "Primary color of the tie"},
                                    "pattern": {"type": "string", "description": "Solid, striped, polka dot, paisley, etc."},
                                    "material": {"type": "string", "description": "Silk, polyester, cotton, etc."}
                                }
                            },
                            "gloves": {
                        "type": "object",
                        "properties": {
                                    "type": {"type": "string", "enum": ["fingerless", "full_finger", "mittens", "driving_gloves", "work_gloves", "dress_gloves", "winter_gloves", "leather_gloves", "cotton_gloves"]},
                                    "color": {"type": "string", "description": "Primary color of the gloves"},
                                    "material": {"type": "string", "description": "Leather, cotton, wool, synthetic, etc."}
                                }
                            },
                            "hat": {
                                    "type": "object",
                                    "properties": {
                                    "type": {"type": "string", "enum": ["baseball_cap", "fedora", "beanie", "beret", "cowboy_hat", "top_hat", "sun_hat", "winter_hat", "helmet", "visor", "turban", "headband"]},
                                    "color": {"type": "string", "description": "Primary color of the hat"},
                                    "material": {"type": "string", "description": "Cotton, wool, leather, synthetic, etc."}
                                }
                            },
                            "jewelry": {
                        "type": "object",
                        "properties": {
                                    "necklaces": {"type": "string", "description": "Type, color, and material of necklaces"},
                                    "rings": {"type": "string", "description": "Type, color, and material of rings"},
                                    "earrings": {"type": "string", "description": "Type, color, and material of earrings"},
                                    "bracelets": {"type": "string", "description": "Type, color, and material of bracelets"},
                                    "piercings": {"type": "string", "description": "Type and location of piercings"}
                                }
                            },
                            "bag": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["handbag", "backpack", "briefcase", "messenger_bag", "tote_bag", "clutch", "satchel", "duffel_bag", "purse", "wallet", "fanny_pack", "laptop_bag"]},
                                    "color": {"type": "string", "description": "Primary color of the bag"},
                                    "material": {"type": "string", "description": "Leather, canvas, nylon, synthetic, etc."},
                                    "size": {"type": "string", "enum": ["small", "medium", "large", "oversized"]}
                                }
                            },
                            "watch": {
                                    "type": "object",
                                    "properties": {
                                    "type": {"type": "string", "enum": ["analog", "digital", "smartwatch", "dress_watch", "sports_watch", "vintage_watch", "luxury_watch", "casual_watch"]},
                                    "color": {"type": "string", "description": "Primary color of the watch"},
                                    "material": {"type": "string", "description": "Metal, leather, rubber, plastic, etc."},
                                    "style": {"type": "string", "description": "Formal, casual, sporty, etc."}
                                }
                            }
                        }
                    },
                    "overall_style": {
                                    "type": "object",
                                    "properties": {
                            "style_category": {"type": "string", "enum": ["casual", "formal", "business", "sporty", "elegant", "bohemian", "vintage", "modern", "streetwear", "preppy"]},
                            "color_scheme": {"type": "string", "description": "Overall color palette (e.g., 'neutral tones', 'bright colors', 'monochrome')"},
                            "season": {"type": "string", "enum": ["summer", "winter", "spring", "autumn", "all_season"]}
                        },
                        "required": ["style_category", "color_scheme"]
                    }
                },
                "required": ["face", "clothing", "footwear", "overall_style"]
            },
            "strict": True
        }
    }


def _schema_character_summary() -> dict[str, object]:
    f"""JSON schema for character summary description ({CHARACTER_SUMMARY_CHARACTER_COUNT} characters).""".format(CHARACTER_SUMMARY_CHARACTER_COUNT)
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
                        "description": f"A summary ({CHARACTER_SUMMARY_CHARACTER_COUNT} characters) of the character's all mentioned visual features."
                    }
                },
                "required": ["summary"]
            },
            "strict": True
        }
    }


def _schema_location_expansion() -> dict[str, object]:
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
                            "type": {"type": "string", "description": "What kind of place (e.g., 'bedroom', 'forest', 'office', 'street')"},
                            "size": {"type": "string", "enum": ["tiny", "small", "medium", "large", "massive"]},
                            "style": {"type": "string", "description": "Visual style (e.g., 'modern', 'rustic', 'Victorian', 'industrial')"}
                        },
                        "required": ["type", "size"]
                    },
                    "lighting": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "Where light comes from (e.g., 'sunlight', 'lamp', 'candle')"},
                            "color": {"type": "string", "enum": ["warm", "cool", "natural", "golden", "white", "dim"]},
                            "brightness": {"type": "string", "enum": ["bright", "moderate", "dim", "dark"]},
                            "time": {"type": "string", "enum": ["morning", "noon", "afternoon", "evening", "night"]}
                        },
                        "required": ["source", "brightness", "time"]
                    },
                    "ground": {
                        "type": "object",
                        "properties": {
                            "material": {"type": "string", "description": "What covers the ground (e.g., 'wood floor', 'grass', 'concrete')"},
                            "color": {"type": "string", "description": "Ground color"}
                        },
                        "required": ["material", "color"]
                    },
                    "walls_or_surroundings": {
                        "type": "object",
                        "properties": {
                            "material": {"type": "string", "description": "What walls/surroundings are (e.g., 'painted walls', 'trees', 'brick')"},
                            "color": {"type": "string", "description": "Wall/surrounding color"}
                        },
                        "required": ["material", "color"]
                    },
                    "objects": {
                        "type": "array",
                        "description": "Visible objects in the scene",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "What the object is (e.g., 'wooden chair', 'red lamp', 'oak tree')"},
                                "type": {"type": "string", "enum": ["furniture", "decoration", "plant", "window", "door", "lighting", "natural", "building", "vehicle", "other"]},
                                "color": {"type": "string", "description": "Object color"},
                                "material": {"type": "string", "description": "What it's made of"},
                                "size": {"type": "string", "enum": ["tiny", "small", "medium", "large", "huge"]},
                                "position": {"type": "string", "description": "Where it is (e.g., 'left side', 'center', 'right corner', 'background')"}
                            },
                            "required": ["name", "type", "color", "material", "size", "position"]
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
                        "description": f"A summary ({LOCATION_SUMMARY_CHARACTER_COUNT} characters) of the location's all mentioned visual features."
                    }
                },
                "required": ["summary"]
            },
            "strict": True
        }
    }


def _schema_story_description() -> dict[str, object]:
    """JSON schema for story description generation."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "story_description",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "description": {
                                "type": "string",
                        "description": f"a Short Version of the COMEPLTE STORY in ({STORY_DESCRIPTION_CHARACTER_COUNT} characters) mentioning the story setting, tone, all events, all characters, all locations, and context for character and location generation"
                    }
                },
                "required": ["description"]
            },
            "strict": True
        }
    }


def _build_character_system_prompt(story_desc: str, character_name: str, all_characters: list[str]) -> str:
    other_characters = [name for name in all_characters if name != character_name]
    other_characters_text = ", ".join(other_characters) if other_characters else "none"
    
    return (
        f"You are a visual director creating focused character descriptions for AI image generation. "
        "Focus on the most important visual elements: face, hair, eyes, skin, and clothing details with colors. "
        "Analyze the story context and character name to determine appropriate visual choices. "
        "Consider the character's profession, role, and story setting when making clothing and style decisions. "
        "For professional characters, prioritize uniform/professional clothing over casual wear. "
        "For characters with relationships to others, consider matching elements like wedding rings or shared accessories. "
        "Keep descriptions concise but specific - focus on colors, textures, and key visual features. "
        f"Describe the character in {ART_STYLE} style. Strictly follow {ART_STYLE} Style.\n\n"
        f"STORY CONTEXT: {story_desc}\n\n"
        f"CHARACTER TO DESCRIBE: {character_name}\n"
        f"OTHER CHARACTERS IN STORY: {other_characters_text}"
    )


def _build_character_summary_prompt(character_name: str, detailed_description: str) -> str:
    return (
        f"You are a visual AI prompt specialist creating concise character summaries ({CHARACTER_SUMMARY_CHARACTER_COUNT} characters) for AI image generation. "
        "Your task is to SHORTEN and CONVERT the detailed description into clear, to-the-point sentences. "
        "ONLY include visual details that are ALREADY MENTIONED in the original description. "
        "DO NOT add any new visual elements, colors, or details that are not present in the original. "
        "PRESERVE ALL MENTIONED DETAILS: Include every visual detail that appears in the original description. "
        "Write as clear, descriptive sentences that capture the exact visual aspects mentioned. "
        "Focus on colors, shapes, textures, and materials that are specifically mentioned. "
        f"Create a concise visual summary ({CHARACTER_SUMMARY_CHARACTER_COUNT} characters) that includes only the visual details from the original description. "
        f"Describe the character in {ART_STYLE} style. Strictly, Accurately, Precisely, always must Follow {ART_STYLE} Style.\n\n"
        f"CHARACTER: {character_name}\n\n"
        f"DETAILED DESCRIPTION: {detailed_description}\n\n"
    )


def _build_story_description_prompt(story_content: str) -> str:
    return (
        f"You are a story analyst creating a Short Version of the COMPLETE STORY for AI character and location generation. "
        f"Analyze the given story content and create a Short Version of the COMPLETE STORY (exactly {STORY_DESCRIPTION_CHARACTER_COUNT} characters) that includes ALL details in proper chronological sequence.\n\n"
        
        "REQUIREMENTS:\n"
        "1. Complete chronological sequence of ALL events from beginning to end\n"
        "2. Every character mentioned in the story with their role and relationships\n"
        "3. Every location visited or mentioned with their significance\n"
        "4. Setting details: time period, place, atmosphere, tone\n"
        "5. Character interactions and dialogue key points\n"
        "6. Plot developments and story progression\n"
        "7. Resolution and conclusion\n\n"
        
        "Write as a a Short Version of the COMPLETE STORY that flows chronologically, covering every detail while staying within the character limit. "
        "This will be used to generate consistent character and location descriptions, so include all visual and contextual information."
        
        f"Create a Short Version of the COMPLETE STORY (exactly {STORY_DESCRIPTION_CHARACTER_COUNT} characters) covering all characters, events, and locations in proper chronological sequence. "
        f"Describe the story in {ART_STYLE} style. Strictly, Accurately, Precisely, always must Follow {ART_STYLE} Style.\n\n"

        f"STORY CONTENT: {story_content}\n\n"
    )


def _build_location_system_prompt(story_desc: str, location_id: str, all_locations: list[str]) -> str:
    other_locations = [loc for loc in all_locations if loc != location_id]
    other_locations_text = ", ".join(other_locations) if other_locations else "none"
    
    return (
        f"You are a visual director creating focused location descriptions for AI image generation. "
        "Focus on the most important visual elements: architecture, lighting, atmosphere, colors, materials, and environmental details. "
        "Analyze the story context and location to determine appropriate visual choices. "
        "Consider the location's purpose, setting, and story context when making architectural and environmental decisions. "
        "For outdoor locations, prioritize natural elements and weather conditions. "
        "For indoor locations, focus on architectural style, lighting, and interior design. "
        "Keep descriptions specific and detailed - focus on colors, textures, materials, and key visual features. "
        f"Describe the location in {ART_STYLE} style. Strictly follow {ART_STYLE} Style.\n\n"
        f"STORY CONTEXT: {story_desc}\n\n"
        f"LOCATION TO DESCRIBE: {location_id}\n"
        f"OTHER LOCATIONS IN STORY: {other_locations_text}"
    )


def _build_location_summary_prompt(location_id: str, detailed_description: str) -> str:
    return (
        f"You are a visual AI prompt specialist creating concise location summaries ({LOCATION_SUMMARY_CHARACTER_COUNT} characters) for AI image generation. "
        "Your task is to SHORTEN and CONVERT the detailed location description into clear, to-the-point sentences. "
        "ONLY include visual details that are ALREADY MENTIONED in the original description. "
        "DO NOT add any new visual elements, colors, or details that are not present in the original. "
        "PRESERVE ALL MENTIONED DETAILS: Include every visual detail that appears in the original description. "
        "Write as clear, descriptive sentences that capture the exact visual aspects mentioned. "
        "Focus on colors, shapes, textures, and materials that are specifically mentioned. "
        f"Create a concise visual summary ({LOCATION_SUMMARY_CHARACTER_COUNT} characters) that includes only the visual details from the original description. "
        f"Describe the location in {ART_STYLE} style. Strictly, Accurately, Precisely, always must Follow {ART_STYLE} Style.\n\n"
        f"LOCATION: {location_id}\n\n"
        f"DETAILED DESCRIPTION: {detailed_description}\n\n"
    )


def _call_lm_studio(system_prompt: str, lm_studio_url: str, model: str, user_payload: str = "", response_format: dict[str, object] | None = None, temperature: float = 1.0) -> str:
    headers = {"Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}]
    if user_payload:
        # Add thinking control to user payload
        thinking_suffix = "" if ENABLE_THINKING else "\n/no_think"
        messages.append({"role": "user", "content": f"{user_payload}{thinking_suffix}"})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 8192,
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


def _format_character_description(char_data: dict[str, object]) -> str:
    """
    Convert any structured character data to readable description format using generic recursive parsing.
    This unified function works for both initial character generation and character rewrites.
    """
    try:
        sentences = _recursively_format_character_data(char_data)
        if not sentences:
            # Fallback: if no sentences generated, try to extract any string values
            sentences = []
            for key, value in char_data.items():
                if isinstance(value, str) and value.strip():
                    formatted_key = key.replace('_', ' ').title()
                    sentences.append(f"{formatted_key.lower()}: {value},")
        
        result = " ".join(sentences)
        if not result.strip():
            # Final fallback: return a generic message
            return "The character has a distinctive appearance with unique physical features."
        
        return result
        
    except Exception as ex:
        print(f"WARNING: Error in generic character formatting: {ex}")
        # Fallback to a safe description
        return "The character has a distinctive appearance with unique physical features."


def _generate_character_descriptions(story_desc: str, characters: list[str], lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    name_to_desc: dict[str, str] = {}
    total = len(characters)
    print(f"Generating structured character descriptions: {total} total")
    
    for idx, name in enumerate(characters, 1):
        # Check if resumable and already complete
        if resumable_state and resumable_state.is_character_complete(name):
            cached_desc = resumable_state.get_character_description(name)
            if cached_desc:
                print(f"({idx}/{total}) {name}: using cached description from checkpoint")
                name_to_desc[name] = cached_desc
                continue
        
        print(f"({idx}/{total}) {name}: generating structured description...")
        prompt = _build_character_system_prompt(story_desc, name, characters)
        
        try:
            # Use structured output with JSON schema
            user_payload = json.dumps({
                "character_name": name,
                "story_context": story_desc,
                "all_characters": characters
            }, ensure_ascii=False)
            
            raw = _call_lm_studio(prompt, lm_studio_url, MODEL_CHARACTER_GENERATION, user_payload, _schema_character())
            structured_data = _parse_structured_response(raw)
            
            if not structured_data:
                raise RuntimeError("Failed to parse structured response")
            
            # Convert structured data to readable description
            desc = _format_character_description(structured_data)
            if not desc:
                raise RuntimeError("Empty description generated")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                resumable_state.set_character_description(name, desc)
                
        except Exception as ex:
            print(f"({idx}/{total}) {name}: FAILED - {ex}")
            print(f"ERROR: Failed to generate description for '{name}': {ex}")
            raise
            
        name_to_desc[name] = desc
        print(f"({idx}/{total}) {name}: done ({len(desc.split())} words)")
        
    return name_to_desc






def _generate_character_summaries(name_to_desc: dict[str, str], lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    name_to_summary: dict[str, str] = {}
    total = len(name_to_desc)
    print(f"Generating character summaries: {total} total")
    
    for idx, (name, detailed_desc) in enumerate(name_to_desc.items(), 1):
        # Check if resumable and already complete
        if resumable_state and resumable_state.is_character_summary_complete(name):
            cached_summary = resumable_state.get_character_summary(name)
            if cached_summary:
                print(f"({idx}/{total}) {name}: using cached summary from checkpoint")
                name_to_summary[name] = cached_summary
                continue
        
        print(f"({idx}/{total}) {name}: generating summary...")
        prompt = _build_character_summary_prompt(name, detailed_desc)
        
        try:
            # Use structured output with JSON schema for summary
            user_payload = json.dumps({
                "character_name": name,
                "detailed_description": detailed_desc
            }, ensure_ascii=False)
            
            raw = _call_lm_studio(prompt, lm_studio_url, MODEL_CHARACTER_SUMMARY, user_payload, _schema_character_summary(), temperature=0.1)
            structured_data = _parse_structured_response(raw)
            
            if not structured_data:
                raise RuntimeError("Failed to parse structured summary response")
            
            summary = structured_data.get("summary", "").strip()
            if not summary:
                raise RuntimeError("Empty summary generated")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                resumable_state.set_character_summary(name, summary)
                
        except Exception as ex:
            print(f"({idx}/{total}) {name}: FAILED - {ex}")
            print(f"ERROR: Failed to generate summary for '{name}': {ex}")
            raise
            
        name_to_summary[name] = summary
        print(f"({idx}/{total}) {name}: done ({len(summary.split())} words)")
        
    return name_to_summary


def _generate_story_description(story_content: str, lm_studio_url: str, resumable_state: ResumableState | None = None) -> str:
    """Generate story description from story content using LLM."""
    # Check if resumable and already complete
    if resumable_state and resumable_state.is_story_description_complete():
        cached_desc = resumable_state.get_story_description()
        if cached_desc:
            print("Using cached story description from checkpoint")
            return cached_desc
    
    print("Generating story description from dialogue content...")
    # Extract only dialogue lines from the story content
    dialogue_content = _extract_dialogue_only_from_content(story_content)
    prompt = _build_story_description_prompt(dialogue_content)
    
    try:
        # Use model constant for story description generation
        model = MODEL_STORY_DESCRIPTION
        
        # Use structured output with JSON schema for story description
        user_payload = json.dumps({
            "story_content": dialogue_content
        }, ensure_ascii=False)
        
        raw = _call_lm_studio(prompt, lm_studio_url, model, user_payload, _schema_story_description())
        structured_data = _parse_structured_response(raw)
        
        if not structured_data:
            raise RuntimeError("Failed to parse structured story description response")
        
        story_desc = structured_data.get("description", "").strip()
        if not story_desc:
            raise RuntimeError("Empty story description generated")
        
        # Save to checkpoint if resumable mode enabled
        if resumable_state:
            resumable_state.set_story_description(story_desc)
            print("Saved story description to checkpoint")
            
    except Exception as ex:
        print(f"ERROR: Failed to generate story description: {ex}")
        raise
        
    print(f"Generated story description: {story_desc}")
    return story_desc


def _filter_story_content_for_location(content: str, pairs: list, location_id: str) -> str:
    """Filter story content to include only dialogue-scene pairs that use the specific location."""
    filtered_pairs = []
    
    for pair in pairs:
        scene = pair.get("scene")
        dialogue = pair.get("dialogue")
        
        # Check if the scene contains the location reference
        scene_uses_location = False
        if scene:
            scene_text = scene.get("raw", "")
            if f"{{{{{location_id}}}}}" in scene_text in scene_text:
                scene_uses_location = True
        
        # If scene uses this location, include both dialogue and scene
        if scene_uses_location:
            filtered_pairs.append(pair)
    
    # Format the filtered pairs back into story format
    if not filtered_pairs:
        return f"No dialogue-scene pairs found that use location '{location_id}'"
    
    formatted_lines = []
    for pair in filtered_pairs:
        dialogue = pair.get("dialogue")
        scene = pair.get("scene")
        
        if dialogue:
            formatted_lines.append(dialogue.get("raw", ""))
        if scene:
            formatted_lines.append(scene.get("raw", ""))
    
    return "\n".join(formatted_lines)


def _generate_location_descriptions(story_desc: str, locations: dict[str, str], lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    """Generate structured location descriptions using LLM."""
    location_id_to_desc: dict[str, str] = {}
    total = len(locations)
    print(f"Generating structured location descriptions: {total} total")
    
    for idx, (loc_id, original_desc) in enumerate(locations.items(), 1):
        # Check if resumable and already complete
        if resumable_state and resumable_state.is_location_complete(loc_id):
            cached_desc = resumable_state.get_location_description(loc_id)
            if cached_desc:
                print(f"({idx}/{total}) {loc_id}: using cached description from checkpoint")
                location_id_to_desc[loc_id] = cached_desc
                continue
        
        print(f"({idx}/{total}) {loc_id}: generating structured description...")
        prompt = _build_location_system_prompt(story_desc, loc_id, list(locations.keys()))
        
        try:
            # Use structured output with JSON schema
            user_payload = json.dumps({
                "location_id": loc_id,
                "story_context": story_desc,
                "all_locations": list(locations.keys())
            }, ensure_ascii=False)
            
            raw = _call_lm_studio(prompt, lm_studio_url, MODEL_LOCATION_EXPANSION, user_payload, _schema_location_expansion())
            structured_data = _parse_structured_response(raw)
            
            if not structured_data:
                raise RuntimeError("Failed to parse structured response")
            
            # Convert structured data to readable description
            desc = _format_location_description(structured_data)
            if not desc:
                raise RuntimeError("Empty description generated")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                resumable_state.set_location_description(loc_id, desc)
                
        except Exception as ex:
            print(f"({idx}/{total}) {loc_id}: FAILED - {ex}")
            print(f"ERROR: Failed to generate description for '{loc_id}': {ex}")
            raise
            
        location_id_to_desc[loc_id] = desc
        print(f"({idx}/{total}) {loc_id}: done ({len(desc.split())} words)")
        
    return location_id_to_desc


def _format_location_description(location_data: dict[str, object]) -> str:
    """
    Convert any structured location data to readable description format using generic recursive parsing.
    This unified function works for both initial location generation and location summaries.
    """
    try:
        sentences = _recursively_format_location_data(location_data)
        if not sentences:
            # Fallback: if no sentences generated, try to extract any string values
            sentences = []
            for key, value in location_data.items():
                if isinstance(value, str) and value.strip():
                    formatted_key = key.replace('_', ' ').title()
                    sentences.append(f"{formatted_key.lower()}: {value},")
        
        result = " ".join(sentences)
        if not result.strip():
            # Final fallback: return a generic message
            return "The location has distinctive architectural and environmental features."
        
        return result
    except Exception as ex:
        print(f"WARNING: Failed to format location description: {ex}")
        return "The location has distinctive architectural and environmental features."


def _recursively_format_location_data(data: dict, prefix: str = "") -> list[str]:
    """Recursively format location data into readable sentences."""
    sentences = []
    
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively process nested objects
            nested_sentences = _recursively_format_location_data(value, f"{prefix}{key}_" if prefix else f"{key}_")
            sentences.extend(nested_sentences)
        elif isinstance(value, str) and value.strip():
            # Format key-value pairs into sentences
            formatted_key = key.replace('_', ' ').title()
            if prefix:
                formatted_key = f"{prefix.replace('_', ' ').title()} {formatted_key}"
            
            # Create natural sentence
            if formatted_key.lower() in ['building type', 'style', 'materials', 'colors']:
                sentences.append(f"The {formatted_key.lower()} is {value}.")
            elif formatted_key.lower() in ['lighting', 'atmosphere', 'mood']:
                sentences.append(f"The {formatted_key.lower()} is {value}.")
            elif formatted_key.lower() in ['dimensions', 'height', 'layout']:
                sentences.append(f"The space is {value} in {formatted_key.lower()}.")
            else:
                sentences.append(f"{formatted_key}: {value}.")
    
    return sentences


def _convert_location_structured_to_text(structured_data: dict) -> str:
    """Convert structured location data to flowing text description."""
    parts = []
    
    # Architecture
    arch = structured_data.get("architecture", {})
    if arch:
        building_type = arch.get("building_type", "")
        style = arch.get("style", "")
        materials = arch.get("materials", "")
        colors = arch.get("colors", "")
        condition = arch.get("condition", "")
        
        arch_desc = f"A {style} {building_type}"
        if materials:
            arch_desc += f" constructed of {materials}"
        if colors:
            arch_desc += f" in {colors}"
        if condition:
            arch_desc += f", appearing {condition}"
        parts.append(arch_desc)
    
    # Lighting
    lighting = structured_data.get("lighting", {})
    if lighting:
        light_type = lighting.get("type", "")
        intensity = lighting.get("intensity", "")
        source = lighting.get("source", "")
        direction = lighting.get("direction", "")
        
        light_desc = f"The lighting is {intensity} and {light_type}"
        if source:
            light_desc += f" from {source}"
        if direction:
            light_desc += f" coming from {direction}"
        parts.append(light_desc)
    
    # Atmosphere
    atmosphere = structured_data.get("atmosphere", {})
    if atmosphere:
        mood = atmosphere.get("mood", "")
        weather = atmosphere.get("weather", "")
        temperature = atmosphere.get("temperature", "")
        
        atm_desc = f"The atmosphere is {mood}"
        if weather:
            atm_desc += f" with {weather} weather"
        if temperature:
            atm_desc += f" and {temperature} temperature"
        parts.append(atm_desc)
    
    # Colors and palette
    colors = structured_data.get("colors_palette", {})
    if colors:
        primary = colors.get("primary_colors", "")
        accent = colors.get("accent_colors", "")
        tone = colors.get("overall_tone", "")
        
        color_desc = f"The color palette features {primary}"
        if accent:
            color_desc += f" with {accent} accents"
        if tone:
            color_desc += f", creating a {tone} tone"
        parts.append(color_desc)
    
    # Size and scale
    size = structured_data.get("size_scale", {})
    if size:
        dimensions = size.get("dimensions", "")
        height = size.get("height", "")
        layout = size.get("layout", "")
        
        size_desc = f"The space is {dimensions}"
        if height:
            size_desc += f" with {height} ceilings"
        if layout:
            size_desc += f" in a {layout} layout"
        parts.append(size_desc)
    
    # Furniture and objects
    furniture = structured_data.get("furniture_objects", {})
    if furniture:
        furn_items = []
        if furniture.get("furniture"):
            furn_items.append(furniture["furniture"])
        if furniture.get("decorations"):
            furn_items.append(furniture["decorations"])
        if furniture.get("appliances"):
            furn_items.append(furniture["appliances"])
        if furniture.get("storage"):
            furn_items.append(furniture["storage"])
        
        if furn_items:
            parts.append(f"The space contains {', '.join(furn_items)}")
    
    # Textures and surfaces
    textures = structured_data.get("textures_surfaces", {})
    if textures:
        surf_items = []
        if textures.get("walls"):
            surf_items.append(f"walls with {textures['walls']}")
        if textures.get("flooring"):
            surf_items.append(f"{textures['flooring']} flooring")
        if textures.get("ceilings"):
            surf_items.append(f"ceilings featuring {textures['ceilings']}")
        if textures.get("windows"):
            surf_items.append(f"windows with {textures['windows']}")
        
        if surf_items:
            parts.append(f"Surfaces include {', '.join(surf_items)}")
    
    # Outdoor elements
    outdoor = structured_data.get("outdoor_elements", {})
    if outdoor:
        outdoor_items = []
        if outdoor.get("vegetation"):
            outdoor_items.append(outdoor["vegetation"])
        if outdoor.get("water_features"):
            outdoor_items.append(outdoor["water_features"])
        if outdoor.get("terrain"):
            outdoor_items.append(f"{outdoor['terrain']} terrain")
        if outdoor.get("wildlife"):
            outdoor_items.append(outdoor["wildlife"])
        
        if outdoor_items:
            parts.append(f"Outdoor elements include {', '.join(outdoor_items)}")
    
    # Time period
    time_period = structured_data.get("time_period", {})
    if time_period:
        era = time_period.get("era", "")
        season = time_period.get("season", "")
        time_of_day = time_period.get("time_of_day", "")
        
        time_desc = f"Set in the {era} era"
        if season:
            time_desc += f" during {season}"
        if time_of_day:
            time_desc += f" at {time_of_day}"
        parts.append(time_desc)
    
    return ". ".join(parts) + "." if parts else ""


def _generate_location_summaries(location_id_to_desc: dict[str, str], lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    """Generate location summaries from detailed descriptions."""
    location_id_to_summary: dict[str, str] = {}
    total = len(location_id_to_desc)
    print(f"Generating location summaries: {total} total")
    
    for idx, (location_id, detailed_desc) in enumerate(location_id_to_desc.items(), 1):
        # Check if resumable and already complete
        if resumable_state and resumable_state.is_location_summary_complete(location_id):
            cached_summary = resumable_state.get_location_summary(location_id)
            if cached_summary:
                print(f"({idx}/{total}) {location_id}: using cached summary from checkpoint")
                location_id_to_summary[location_id] = cached_summary
                continue
        
        print(f"({idx}/{total}) {location_id}: generating summary...")
        
        prompt = _build_location_summary_prompt(location_id, detailed_desc)
        
        try:
            # Use structured output with JSON schema for location summary
            user_payload = json.dumps({
                "location_id": location_id,
                "detailed_description": detailed_desc
            }, ensure_ascii=False)
            
            raw = _call_lm_studio(prompt, lm_studio_url, MODEL_LOCATION_EXPANSION, user_payload, _schema_location_summary())
            structured_data = _parse_structured_response(raw)
            
            if not structured_data:
                raise RuntimeError("Failed to parse structured location summary response")
            
            summary = structured_data.get("summary", "").strip()
            if not summary:
                raise RuntimeError("Empty location summary generated")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                resumable_state.set_location_summary(location_id, summary)
                
        except Exception as ex:
            print(f"({idx}/{total}) {location_id}: FAILED - {ex}")
            # Use a fallback summary
            summary = f"Location {location_id}: {detailed_desc[:200]}..."
            location_id_to_summary[location_id] = summary
            continue
        
        location_id_to_summary[location_id] = summary
        print(f"({idx}/{total}) {location_id}: done ({len(summary.split())} words)")
        
    return location_id_to_summary


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
    parser.add_argument("--enable-thinking", action="store_true",
                       help="Enable thinking in LM Studio responses (default: disabled)")
    args = parser.parse_args()

    # Update ENABLE_THINKING based on CLI argument
    if args.enable_thinking:
        ENABLE_THINKING = True
        print("🧠 Thinking enabled in LM Studio responses")
    else:
        print("🚫 Thinking disabled in LM Studio responses (using /no_think)")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    audio_story_path = os.path.normpath(os.path.join(base_dir, "../../gen.audio/input/1.story.txt"))
    image_story_path = os.path.normpath(os.path.join(base_dir, "../input/1.story.txt"))
    scene_out_path = os.path.normpath(os.path.join(base_dir, "../input/3.scene.txt"))
    characters_out_path = os.path.normpath(os.path.join(base_dir, "../input/2.character.txt"))
    character_summary_out_path = os.path.normpath(os.path.join(base_dir, "../input/3.character.txt"))
    locations_out_path = os.path.normpath(os.path.join(base_dir, "../input/3.location.txt"))
    
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
            story_desc = _generate_story_description(content, lm_studio_url, resumable_state)
            
            # Generate structured location descriptions
            expanded_locations = _generate_location_descriptions(story_desc, locations, lm_studio_url, resumable_state)
            
            # Generate location summaries from expanded descriptions
            location_summaries = _generate_location_summaries(expanded_locations, lm_studio_url, resumable_state)
            
            # Write expanded locations file
            written_locations = write_locations_file(expanded_locations, locations_out_path)
            print(f"Wrote {written_locations} expanded location entries to: {locations_out_path}")
            
            # Write location summaries file
            location_summaries_path = os.path.join(os.path.dirname(locations_out_path), "3.location.txt")
            written_summaries = write_locations_file(location_summaries, location_summaries_path)
            print(f"Wrote {written_summaries} location summary entries to: {location_summaries_path}")
            
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
