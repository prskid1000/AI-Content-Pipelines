import os
import re
import json
import requests
import argparse
from functools import partial
import builtins as _builtins
import time
from pathlib import Path


CHARACTER_SUMMARY_CHARACTER_COUNT = 600
LOCATION_CHARACTER_COUNT = 3600
STORY_DESCRIPTION_CHARACTER_COUNT = 16000

# Feature flags
ENABLE_CHARACTER_REWRITE = True  # Set to False to skip character rewriting step
ENABLE_RESUMABLE_MODE = True  # Set to False to disable resumable mode
CLEANUP_TRACKING_FILES = False  # Set to True to delete tracking JSON files after completion, False to preserve them
ENABLE_THINKING = False  # Set to True to enable thinking in LM Studio responses

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
            "characters": {"completed": [], "results": {}},
            "character_rewrite": {"completed": False, "result": None},
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
    
    def is_character_rewrite_complete(self) -> bool:
        """Check if character rewriting is complete."""
        return self.state["character_rewrite"]["completed"]
    
    def get_character_rewrite(self) -> dict[str, str] | None:
        """Get cached character rewrite results if available."""
        return self.state["character_rewrite"]["result"]
    
    def set_character_rewrite(self, rewritten_descriptions: dict[str, str]):
        """Set character rewrite results and mark as complete."""
        self.state["character_rewrite"]["completed"] = True
        self.state["character_rewrite"]["result"] = rewritten_descriptions
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
        rewrite_done = "✓" if self.is_character_rewrite_complete() else "✗"
        summaries_done = len(self.state["character_summaries"]["completed"])
        summaries_total = len(self.state["character_summaries"]["results"]) + len([k for k in self.state["character_summaries"]["results"].keys() if k not in self.state["character_summaries"]["completed"]])
        
        return (
            f"Progress: Story({story_done}) Locations({locations_done}/{locations_total}) "
            f"Characters({characters_done}/{characters_total}) Rewrite({rewrite_done}) "
            f"Summaries({summaries_done}/{summaries_total})"
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
    """JSON schema for detailed character description with all body parts and properties."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "character_description",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "head_face": {
                        "type": "object",
                        "properties": {
                            "head_shape": {"type": "string", "enum": ["oval", "round", "square", "rectangular", "heart", "diamond", "triangle"]},
                            "skin_tone": {"type": "string", "enum": ["pale", "fair", "medium", "olive", "tan", "dark"]},
                            "skin_texture": {"type": "string", "enum": ["smooth", "slightly textured", "rough", "weathered"]},
                            "freckles": {"type": "string", "enum": ["none", "light", "moderate", "heavy"]},
                            "scars": {"type": "string", "enum": ["none", "small", "prominent", "patterned"]},
                            "complexion_details": {"type": "string"},
                            "forehead": {"type": "string"},
                            "eyebrows": {"type": "string"},
                            "eyes": {"type": "string"},
                            "nose": {"type": "string"},
                            "mouth_lips": {"type": "string"},
                            "cheeks": {"type": "string"},
                            "jawline_chin": {"type": "string"},
                            "hair": {"type": "string"},
                            "facial_hair": {"type": "string"}
                        },
                        "required": ["head_shape", "skin_tone", "skin_texture", "eyes", "nose", "mouth_lips", "hair"]
                    },
                    "neck": {
                        "type": "object",
                        "properties": {
                            "length": {"type": "string", "enum": ["short", "medium", "long"]},
                            "thickness": {"type": "string", "enum": ["thin", "medium", "thick"]},
                            "details": {"type": "string"}
                        },
                        "required": ["length", "thickness"]
                    },
                    "torso_upper_body": {
                        "type": "object",
                        "properties": {
                            "shoulders": {"type": "string"},
                            "chest": {"type": "string"},
                            "torso_shape": {"type": "string"},
                            "waist": {"type": "string"},
                            "posture": {"type": "string"}
                        },
                        "required": ["shoulders", "torso_shape", "posture"]
                    },
                    "arms_hands": {
                        "type": "object",
                        "properties": {
                            "arm_length": {"type": "string", "enum": ["short", "medium", "long"]},
                            "muscle_definition": {"type": "string", "enum": ["lean", "toned", "muscular", "soft"]},
                            "hands": {"type": "string"},
                            "details": {"type": "string"}
                        },
                        "required": ["arm_length", "muscle_definition", "hands"]
                    },
                    "legs_feet": {
                        "type": "object",
                        "properties": {
                            "leg_length": {"type": "string", "enum": ["short", "medium", "long"]},
                            "muscle_definition": {"type": "string", "enum": ["lean", "toned", "muscular", "soft"]},
                            "feet": {"type": "string"},
                            "details": {"type": "string"}
                        },
                        "required": ["leg_length", "muscle_definition"]
                    },
                    "body_hair": {
                        "type": "object",
                        "properties": {
                            "density": {"type": "string", "enum": ["none", "light", "moderate", "heavy"]},
                            "distribution": {"type": "string"},
                            "color": {"type": "string"}
                        },
                        "required": ["density"]
                    },
                    "skin_details": {
                        "type": "object",
                        "properties": {
                            "texture_details": {"type": "string"},
                            "color_variations": {"type": "string"},
                            "reflectivity": {"type": "string", "enum": ["matte", "dewy", "shiny", "oily"]},
                            "markings": {"type": "string"}
                        },
                        "required": ["reflectivity"]
                    },
                    "clothing_accessories": {
                        "type": "object",
                        "properties": {
                            "clothing_fit": {"type": "string", "enum": ["tight", "fitted", "loose", "layered"]},
                            "clothing_style": {"type": "string"},
                            "clothing_material": {"type": "string"},
                            "clothing_colors": {"type": "string"},
                            "clothing_condition": {"type": "string", "enum": ["pristine", "well-worn", "weathered", "tattered"]},
                            "accessories": {"type": "string"},
                            "footwear": {"type": "string"},
                            "props": {"type": "string"}
                        },
                        "required": ["clothing_fit", "clothing_style", "clothing_material", "footwear"]
                    },
                    "overall_impression": {
                        "type": "object",
                        "properties": {
                            "height": {"type": "string", "enum": ["medium", "tall"]},
                            "build": {"type": "string", "enum": ["petite", "slim", "average", "stocky", "robust", "imposing"]},
                            "age_appearance": {
                                "type": "object",
                                "properties": {
                                    "age_category": {"type": "string", "enum": ["child", "teenager", "young_adult", "adult", "middle_aged", "elderly", "old"]},
                                    "specific_age": {"type": "string", "description": "Specific age number (e.g., '24', '45', '12')"},
                                    "age_description": {"type": "string", "description": "Detailed age appearance description (e.g., 'appears to be in early twenties', 'looks like a teenager')"}
                                },
                                "required": ["age_category", "specific_age", "age_description"]
                            },
                            "distinctive_traits": {"type": "string"},
                            "presence": {"type": "string"}
                        },
                        "required": ["height", "build", "age_appearance", "distinctive_traits"]
                    },
                    "relationships": {
                        "type": "object",
                        "properties": {
                            "profession": {"type": "string", "description": "Character's profession or role"},
                            "organization": {"type": "string", "description": "Organization, company, or group they belong to"},
                            "relationships": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "character_name": {"type": "string"},
                                        "relationship_type": {"type": "string", "enum": ["family", "romantic", "professional", "friend", "colleague", "superior", "subordinate", "rival", "mentor", "student", "teammate", "partner"]},
                                        "relationship_description": {"type": "string"}
                                    },
                                    "required": ["character_name", "relationship_type", "relationship_description"]
                                }
                            }
                        },
                        "required": ["profession", "relationships"]
                    },
                    "shared_elements": {
                        "type": "object",
                        "properties": {
                            "uniform_requirements": {
                                "type": "object",
                                "properties": {
                                    "has_uniform": {"type": "boolean"},
                                    "uniform_type": {"type": "string"},
                                    "uniform_details": {"type": "string"},
                                    "shared_with_characters": {"type": "array", "items": {"type": "string"}}
                                }
                            },
                            "matching_accessories": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "accessory_type": {"type": "string"},
                                        "accessory_description": {"type": "string"},
                                        "shared_with_characters": {"type": "array", "items": {"type": "string"}}
                                    },
                                    "required": ["accessory_type", "accessory_description", "shared_with_characters"]
                                }
                            },
                            "professional_equipment": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "equipment_type": {"type": "string"},
                                        "equipment_description": {"type": "string"},
                                        "shared_with_characters": {"type": "array", "items": {"type": "string"}}
                                    },
                                    "required": ["equipment_type", "equipment_description", "shared_with_characters"]
                                }
                            }
                        },
                        "required": ["uniform_requirements", "matching_accessories", "professional_equipment"]
                    }
                },
                "required": ["head_face", "neck", "torso_upper_body", "arms_hands", "legs_feet", "clothing_accessories", "overall_impression", "relationships", "shared_elements"]
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
                        "description": f"A summary ({CHARACTER_SUMMARY_CHARACTER_COUNT} characters) of the character's key visual features, suitable for AI image generation"
                    }
                },
                "required": ["summary"]
            },
            "strict": True
        }
    }


def _schema_location_expansion() -> dict[str, object]:
    """JSON schema for location expansion (500 characters)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "location_expansion",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "expanded_description": {
                        "type": "string",
                        "description": f"An expanded, detailed background description of the location (approximately {LOCATION_CHARACTER_COUNT} characters), focusing ONLY on environmental and architectural elements, excluding all character references, rich in visual details for AI image generation"
                    }
                },
                "required": ["expanded_description"]
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


def _schema_character_rewrite() -> dict[str, object]:
    """JSON schema for character description rewriting with paragraph descriptions."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "character_rewrite",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rewritten_characters": {
                        "type": "object",
                        "description": "Dictionary mapping character names to their rewritten character data with preserved age information",
                        "patternProperties": {
                            ".*": {
                                "type": "object",
                                "properties": {
                                    "description": {
                                        "type": "string",
                                        "description": "Complete character description written as a flowing paragraph with different physical appearance but preserved relationships and shared elements"
                                    },
                                    "age_appearance": {
                                        "type": "object",
                                        "properties": {
                                            "age_category": {"type": "string", "enum": ["child", "teenager", "young_adult", "adult", "middle_aged", "elderly", "old"]},
                                            "specific_age": {"type": "string", "description": "Specific age number (e.g., '24', '45', '12')"},
                                            "age_description": {"type": "string", "description": "Detailed age appearance description"}
                                        },
                                        "required": ["age_category", "specific_age", "age_description"]
                                    }
                                },
                                "required": ["description", "age_appearance"]
                            }
                        }
                    }
                },
                "required": ["rewritten_characters"]
            },
            "strict": True
        }
    }


def _build_character_system_prompt(story_desc: str, character_name: str, all_characters: list[str]) -> str:
    other_characters = [name for name in all_characters if name != character_name]
    other_characters_text = ", ".join(other_characters) if other_characters else "none"
    
    return (
        f"You are a visual director creating detailed character descriptions for visual AI generation. "
        "Analyze the story context and character name to create a comprehensive physical description. "
        "Be specific and detailed for each body part and clothing element. Ground all choices in the story context and character role. "
        "Consider the character's profession, social status, personality traits, and story setting when making choices. "
        "Pay special attention to professional relationships and shared elements with other characters. "
        "If the character has a profession that requires uniforms or specific equipment, include those details. "
        "If the character is in relationships with other characters, consider matching elements like wedding rings or family jewelry. "
        "When determining relationships, consider if this character shares a profession with other characters in the story. "
        "If characters are colleagues (same profession), they should have matching uniforms/equipment but different physical appearances. "
        "If characters are family members or romantic partners, they might share matching accessories like wedding rings. "
        "Provide rich, specific details that will help create a vivid and consistent visual representation."
        "Don't use any keyword/adjective unless needed, like scar from **childhood** accident can make model understand it as chracter is a **child** though actually an adult, instead use scar from old accident"
        f"Describe the character in {ART_STYLE} style. Strictly, Accurately, Precisely, always must Follow {ART_STYLE} Style.\n\n"
        f"STORY CONTEXT: {story_desc}\n\n"
        f"CHARACTER TO DESCRIBE: {character_name}\n"
        f"OTHER CHARACTERS IN STORY: {other_characters_text}"
    )


def _build_character_summary_prompt(character_name: str, detailed_description: str) -> str:
    return (
        f"You are a visual AI prompt specialist creating concise character summaries ({CHARACTER_SUMMARY_CHARACTER_COUNT} characters) for AI image generation. "
        "Extract ONLY the most important visual details from the character description for AI image generation. "
        "FOCUS EXCLUSIVELY ON VISUAL ELEMENTS: facial features (eyes, nose, mouth, eyebrows), hair (color, style, texture), skin tone/texture/complexion, age appearance, and clothing (style, colors, materials, fit). "
        "Include age category and specific age number for proper character rendering. "
        "IGNORE: personality traits, backstory, relationships, body proportions, posture, hands, legs, feet, and non-visual elements. "
        "Use precise visual terms separated by commas. Focus on colors, shapes, textures, and materials that AI can render. "
        f"Create a highly visual summary ({CHARACTER_SUMMARY_CHARACTER_COUNT} characters) optimized for AI image generation focusing on facial features, hair, skin, age, and clothing. "
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


def _build_location_expansion_prompt(filtered_story_content: str, location_id: str, original_description: str) -> str:
    return (
        f"You are a visual director creating detailed background location descriptions (approximately {LOCATION_CHARACTER_COUNT} characters) for AI image generation. "
        "Based on the filtered story content that specifically uses this location, generate a comprehensive background description that covers ONLY the environmental and architectural elements of this place. "
        "Focus exclusively on the physical setting, atmosphere, lighting, textures, environmental details, architecture, furniture, objects, and any distinctive visual features of the location itself. "
        "EXCLUDE all character references, character actions, character interactions, or any human elements. "
        "Consider the story context and setting to ensure the description fits the narrative tone and period. "
        "Include specific details about architecture, furniture, lighting conditions, colors, materials, objects, and any distinctive visual features of the place itself. "
        "Make the description vivid, immersive, and focused purely on the background/environment for AI image generation.\n\n"
        f"Create a comprehensive background location description (approximately {LOCATION_CHARACTER_COUNT} characters) that covers ONLY the environmental and architectural elements of location {location_id}, excluding all character references. "
        f"Describe the location in {ART_STYLE} style. Strictly, Accurately, Precisely, always must Follow {ART_STYLE} Style.\n\n"

        f"FILTERED STORY CONTENT (only scenes using this location): {filtered_story_content}\n\n"
        f"LOCATION ID: {location_id}\n"
        f"ORIGINAL DESCRIPTION: {original_description}\n\n"
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


def _build_character_rewrite_prompt(name_to_desc: dict[str, str], story_desc: str) -> str:
    """Build the prompt for rewriting all character descriptions."""
    # Build combined prompt with all character descriptions
    characters_text = ""
    character_names = list(name_to_desc.keys())
    for name, desc in name_to_desc.items():
        characters_text += f"Character: {name}\nDescription: {desc}\n\n"
    
    return f"""You are a visual character designer specializing in creating distinct visual appearances for AI image generation. Rewrite character descriptions to make each character visually unique while preserving essential story elements.

Story Context: {story_desc}

All Characters in Story: {', '.join(character_names)}

Current Character Descriptions:
{characters_text}

VISUAL FOCUS INSTRUCTIONS:
- **PRIMARY FOCUS**: Create visually distinct facial features, skin tone, hair, and body characteristics
- **VISUAL CHANGES**: Modify eyes, nose, mouth, hair color/style, skin tone, facial structure, body build, and clothing style
- **PRESERVE**: Keep age information, professions, and essential story relationships
- **FORMAT**: Write each description as a flowing visual paragraph with structured age data

VISUAL DISTINCTIVENESS REQUIREMENTS:
- Make facial features completely different between characters (eye shape, nose type, mouth, jawline)
- Vary skin tones, textures, and complexions significantly
- Create distinct hair colors, styles, and textures
- Vary body builds, heights, and physical proportions
- Modify clothing styles, colors, and personal accessories
- **CRITICAL**: Preserve exact age category, specific age number, and age description for each character

VISUAL QUALITY STANDARDS:
- Focus on vivid, specific visual descriptions that AI can render
- Emphasize physical appearance over personality or background details
- Use descriptive visual terms (colors, shapes, textures, materials)
- Ensure each character has a completely unique visual identity
- Write descriptions optimized for AI image generation
- Avoid non-visual elements like personality traits or backstory details

Return structured character data with highly visual, distinct physical appearances formatted as paragraphs while preserving age and essential story elements."""


def _rewrite_all_character_descriptions(name_to_desc: dict[str, str], story_desc: str, lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    """Rewrite all character descriptions at once using Qwen model to make them look distinct from each other while preserving primary features and relationships."""
    # Check if resumable and already complete
    if resumable_state and resumable_state.is_character_rewrite_complete():
        cached_rewrite = resumable_state.get_character_rewrite()
        if cached_rewrite:
            print("Using cached character rewrite from checkpoint")
            return cached_rewrite
    
    print(f"Rewriting all character descriptions using Qwen model: {len(name_to_desc)} characters")
    
    # Build prompt using the separate function
    prompt = _build_character_rewrite_prompt(name_to_desc, story_desc)

    try:
        # Use structured output with JSON schema for character rewriting
        user_payload = json.dumps({
            "story_context": story_desc,
            "character_count": len(name_to_desc)
        }, ensure_ascii=False)
        
        raw = _call_lm_studio(prompt, lm_studio_url, MODEL_CHARACTER_GENERATION, user_payload, _schema_character_rewrite())
        structured_data = _parse_structured_response(raw)
        
        if not structured_data:
            raise RuntimeError("Failed to parse structured character rewrite response")
        
        rewritten_characters = structured_data.get("rewritten_characters", {})
        if not rewritten_characters:
            raise RuntimeError("No rewritten character data generated")
            
        # Validate that all characters are present
        if set(rewritten_characters.keys()) != set(name_to_desc.keys()):
            missing = set(name_to_desc.keys()) - set(rewritten_characters.keys())
            extra = set(rewritten_characters.keys()) - set(name_to_desc.keys())
            raise RuntimeError(f"Character mismatch - Missing: {missing}, Extra: {extra}")
        
        # The model now returns structured character data with descriptions and age information
        rewritten_descriptions = {}
        for char_name, char_data in rewritten_characters.items():
            if not isinstance(char_data, dict):
                raise RuntimeError(f"Invalid character data format for {char_name}")
            
            description = char_data.get("description", "").strip()
            age_appearance = char_data.get("age_appearance", {})
            
            if not description:
                raise RuntimeError(f"Missing description for {char_name}")
            
            # Validate age information is preserved
            if not age_appearance or not all(key in age_appearance for key in ["age_category", "specific_age", "age_description"]):
                raise RuntimeError(f"Missing or incomplete age information for {char_name}")
            
            # Format the final description with age information
            age_category = age_appearance.get("age_category", "")
            specific_age = age_appearance.get("specific_age", "")
            age_desc = age_appearance.get("age_description", "")
            
            # Combine description with age information
            full_description = f"{description} Age: {age_category} ({specific_age} years old) - {age_desc}"
            rewritten_descriptions[char_name] = full_description.strip()
        
        # Save to checkpoint if resumable mode enabled
        if resumable_state:
            resumable_state.set_character_rewrite(rewritten_descriptions)
            print("Saved character rewrite to checkpoint")
            
        print(f"Successfully rewrote {len(rewritten_descriptions)} character descriptions")
        return rewritten_descriptions
        
    except Exception as ex:
        print(f"ERROR: Failed to rewrite character descriptions: {ex}")
        raise


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


def _expand_locations(locations: dict[str, str], pairs: list, content: str, lm_studio_url: str, resumable_state: ResumableState | None = None) -> dict[str, str]:
    """Expand location descriptions using LLM to generate 500-character descriptions."""
    expanded_locations: dict[str, str] = {}
    total = len(locations)
    print(f"Expanding location descriptions: {total} total")
    
    # Use model constant for location expansion
    model = MODEL_LOCATION_EXPANSION
    
    for idx, (loc_id, original_desc) in enumerate(locations.items(), 1):
        # Check if resumable and already complete
        if resumable_state and resumable_state.is_location_complete(loc_id):
            cached_desc = resumable_state.get_location_description(loc_id)
            if cached_desc:
                print(f"({idx}/{total}) {loc_id}: using cached description from checkpoint")
                expanded_locations[loc_id] = cached_desc
                continue
        
        print(f"({idx}/{total}) {loc_id}: filtering story content and expanding description...")
        
        # Filter story content to include only pairs that use this location
        filtered_content = _filter_story_content_for_location(content, pairs, loc_id)
        print(f"({idx}/{total}) {loc_id}: filtered content length: {len(filtered_content)} characters")
        
        prompt = _build_location_expansion_prompt(filtered_content, loc_id, original_desc)
        
        try:
            # Use structured output with JSON schema for location expansion
            user_payload = json.dumps({
                "location_id": loc_id,
                "original_description": original_desc,
                "filtered_story_content": filtered_content[:2000]  # Limit to avoid token limits
            }, ensure_ascii=False)
            
            raw = _call_lm_studio(prompt, lm_studio_url, model, user_payload, _schema_location_expansion())
            structured_data = _parse_structured_response(raw)
            
            if not structured_data:
                raise RuntimeError("Failed to parse structured location expansion response")
            
            expanded_desc = structured_data.get("expanded_description", "").strip()
            if not expanded_desc:
                raise RuntimeError("Empty expanded description generated")
            
            # Save to checkpoint if resumable mode enabled
            if resumable_state:
                resumable_state.set_location_description(loc_id, expanded_desc)
                
        except Exception as ex:
            print(f"({idx}/{total}) {loc_id}: FAILED - {ex}")
            print(f"ERROR: Failed to expand location '{loc_id}': {ex}")
            raise
            
        expanded_locations[loc_id] = expanded_desc
        print(f"({idx}/{total}) {loc_id}: done ({len(expanded_desc)} characters)")
        
    return expanded_locations


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
            
            # Expand locations using the filtered story content and pairs
            expanded_locations = _expand_locations(locations, pairs, content, lm_studio_url, resumable_state)
            
            # Write expanded locations file
            written_locations = write_locations_file(expanded_locations, locations_out_path)
            print(f"Wrote {written_locations} expanded location entries to: {locations_out_path}")
            
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
                
                # Step 2: Optionally rewrite all character descriptions at once using Qwen model
                if ENABLE_CHARACTER_REWRITE:
                    print("Rewriting character descriptions to make them look completely different...")
                    final_descriptions = _rewrite_all_character_descriptions(name_to_desc, story_desc, lm_studio_url, resumable_state)
                else:
                    print("Character rewriting disabled - using original descriptions")
                    final_descriptions = name_to_desc
                
                # Step 3: Write the main character file with final descriptions
                char_errors = _validate_and_write_character_file(final_descriptions, characters_out_path, set(unique_names))
                if char_errors:
                    print(f"Character description output had {char_errors} error(s).")
                    return 1
                
                # Step 4: Generate character summaries from final descriptions
                name_to_summary = _generate_character_summaries(final_descriptions, lm_studio_url, resumable_state)
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
