import os
import re
import json
import requests
import argparse
from functools import partial
import builtins as _builtins


CHARACTER_SUMMARY_CHARACTER_COUNT = 500
LOCATION_CHARACTER_COUNT = 1600
STORY_DESCRIPTION_CHARACTER_COUNT = 3600

# Feature flags
ENABLE_CHARACTER_REWRITE = True  # Set to False to skip character rewriting step

# Model constants for easy switching
MODEL_STORY_DESCRIPTION = "qwen2.5-omni-7b"  # Model for generating story descriptions
MODEL_CHARACTER_GENERATION = "qwen/qwen3-14b"  # Model for character description generation
MODEL_CHARACTER_SUMMARY = "qwen/qwen3-14b"  # Model for character summary generation
MODEL_LOCATION_EXPANSION = "qwen/qwen3-14b"  # Model for location expansion

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
                            "height": {"type": "string", "enum": ["short", "medium", "tall", "very tall"]},
                            "build": {"type": "string", "enum": ["petite", "slim", "average", "stocky", "robust", "imposing"]},
                            "age_appearance": {"type": "string"},
                            "distinctive_traits": {"type": "string"},
                            "presence": {"type": "string"}
                        },
                        "required": ["height", "build", "age_appearance", "distinctive_traits"]
                    }
                },
                "required": ["head_face", "neck", "torso_upper_body", "arms_hands", "legs_feet", "clothing_accessories", "overall_impression"]
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
                        "description": f"A concise summary ({STORY_DESCRIPTION_CHARACTER_COUNT} characters) of the story setting, tone, all events, all characters, all locations, and context for character and location generation"
                    }
                },
                "required": ["description"]
            },
            "strict": True
        }
    }


def _schema_character_rewrite() -> dict[str, object]:
    """JSON schema for character description rewriting."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "character_rewrite",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rewritten_descriptions": {
                        "type": "object",
                        "description": "Dictionary mapping character names to their rewritten descriptions",
                        "patternProperties": {
                            ".*": {
                                "type": "string",
                                "description": "Rewritten character description that looks completely different while preserving primary features"
                            }
                        }
                    }
                },
                "required": ["rewritten_descriptions"]
            },
            "strict": True
        }
    }


def _build_character_system_prompt(story_desc: str, character_name: str) -> str:
    return (
        f"You are a visual director creating detailed character descriptions for visual AI generation. "
        "Analyze the story context and character name to create a comprehensive physical description. "
        "Be specific and detailed for each body part and clothing element. Ground all choices in the story context and character role. "
        "Consider the character's profession, social status, personality traits, and story setting when making choices. "
        "Provide rich, specific details that will help create a vivid and consistent visual representation.\n\n"
        f"STORY CONTEXT: {story_desc}\n\n"
        f"CHARACTER TO DESCRIBE: {character_name}"
    )


def _build_character_summary_prompt(character_name: str, detailed_description: str) -> str:
    return (
        f"You are a visual AI prompt specialist creating concise character summaries ({CHARACTER_SUMMARY_CHARACTER_COUNT} characters) for AI image generation. "
        "Take the detailed character description and extract ONLY the head, face, and full clothing details into a comma-separated list. "
        "Focus specifically on: head shape, facial features (eyes, nose, mouth, hair, facial hair), skin tone/texture, and full clothing style/fit/material/colors. "
        "Ignore body proportions, posture, hands, legs, feet, and other body parts. "
        "Use clear, descriptive terms separated by commas. Avoid unnecessary words and focus on visual impact.\n\n"
        f"CHARACTER: {character_name}\n\n"
        f"DETAILED DESCRIPTION: {detailed_description}\n\n"
        f"Create a summary ({CHARACTER_SUMMARY_CHARACTER_COUNT} characters) focusing ONLY on head, face, and clothing features."
    )


def _build_story_description_prompt(story_content: str) -> str:
    return (
        f"You are a story analyst creating concise story descriptions for AI character and location generation. "
        f"Analyze the given story content and create a brief summary ({STORY_DESCRIPTION_CHARACTER_COUNT} characters) that captures the setting, time period, tone, all events, all characters, all locations, and overall context. "
        "Focus on the essential elements that would help generate appropriate character descriptions and location expansions. "
        "Include details about the setting (time period, location type), atmosphere, and narrative tone.\n\n"
        f"STORY CONTENT:\n{story_content}\n\n"
        f"Create a concise story description ({STORY_DESCRIPTION_CHARACTER_COUNT} characters) focusing on setting, time period, tone, all events, all characters, all locations, and context."
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
        f"FILTERED STORY CONTENT (only scenes using this location):\n{filtered_story_content}\n\n"
        f"LOCATION ID: {location_id}\n"
        f"ORIGINAL DESCRIPTION: {original_description}\n\n"
        f"Create a comprehensive background location description (approximately {LOCATION_CHARACTER_COUNT} characters) that covers ONLY the environmental and architectural elements of location {location_id}, excluding all character references."
    )


def _call_lm_studio(system_prompt: str, lm_studio_url: str, model: str, user_payload: str = "", response_format: dict[str, object] | None = None) -> str:
    headers = {"Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}]
    if user_payload:
        messages.append({"role": "user", "content": user_payload})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
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


def _format_character_description(char_data: dict[str, object]) -> str:
    """Convert structured character data to readable description format."""
    lines = []
    
    # Head/Face section
    head_face = char_data.get("head_face", {})
    if head_face:
        face_parts = []
        if head_face.get("head_shape"): face_parts.append(f"{head_face['head_shape']} head shape")
        if head_face.get("skin_tone"): face_parts.append(f"{head_face['skin_tone']} skin tone")
        if head_face.get("eyes"): face_parts.append(str(head_face['eyes']))
        if head_face.get("nose"): face_parts.append(str(head_face['nose']))
        if head_face.get("mouth_lips"): face_parts.append(str(head_face['mouth_lips']))
        if head_face.get("hair"): face_parts.append(str(head_face['hair']))
        if face_parts:
            lines.append("Face: " + ", ".join(face_parts) + ".")
    
    # Overall impression
    overall = char_data.get("overall_impression", {})
    if overall:
        impression_parts = []
        if overall.get("height") and overall.get("build"):
            impression_parts.append(f"{overall['height']} height with {overall['build']} build")
        if overall.get("age_appearance"): impression_parts.append(str(overall['age_appearance']))
        if overall.get("distinctive_traits"): impression_parts.append(str(overall['distinctive_traits']))
        if impression_parts:
            lines.append("Build: " + ", ".join(impression_parts) + ".")
    
    # Clothing and accessories
    clothing = char_data.get("clothing_accessories", {})
    if clothing:
        clothing_parts = []
        if clothing.get("clothing_style"): clothing_parts.append(str(clothing['clothing_style']))
        if clothing.get("clothing_material"): clothing_parts.append(f"made of {clothing['clothing_material']}")
        if clothing.get("clothing_colors"): clothing_parts.append(str(clothing['clothing_colors']))
        if clothing.get("accessories"): clothing_parts.append(str(clothing['accessories']))
        if clothing.get("footwear"): clothing_parts.append(str(clothing['footwear']))
        if clothing_parts:
            lines.append("Clothing: " + ", ".join(clothing_parts) + ".")
    
    # Additional details from other sections
    details = []
    
    # Neck details
    neck = char_data.get("neck", {})
    if neck and neck.get("details"):
        details.append(str(neck['details']))
    
    # Torso details
    torso = char_data.get("torso_upper_body", {})
    if torso:
        if torso.get("posture"): details.append(f"posture: {torso['posture']}")
        if torso.get("shoulders"): details.append(str(torso['shoulders']))
    
    # Arms and hands
    arms = char_data.get("arms_hands", {})
    if arms and arms.get("hands"):
        details.append(str(arms['hands']))
    
    # Skin details
    skin = char_data.get("skin_details", {})
    if skin:
        if skin.get("texture_details"): details.append(str(skin['texture_details']))
        if skin.get("markings"): details.append(str(skin['markings']))
    
    if details:
        lines.append("Details: " + ", ".join(details) + ".")
    
    return " ".join(lines)


def _generate_character_descriptions(story_desc: str, characters: list[str], lm_studio_url: str) -> dict[str, str]:
    name_to_desc: dict[str, str] = {}
    total = len(characters)
    print(f"Generating structured character descriptions: {total} total")
    
    for idx, name in enumerate(characters, 1):
        print(f"({idx}/{total}) {name}: generating structured description...")
        prompt = _build_character_system_prompt(story_desc, name)
        
        try:
            # Use structured output with JSON schema
            user_payload = json.dumps({
                "character_name": name,
                "story_context": story_desc
            }, ensure_ascii=False)
            
            raw = _call_lm_studio(prompt, lm_studio_url, MODEL_CHARACTER_GENERATION, user_payload, _schema_character())
            structured_data = _parse_structured_response(raw)
            
            if not structured_data:
                raise RuntimeError("Failed to parse structured response")
            
            # Convert structured data to readable description
            desc = _format_character_description(structured_data)
            if not desc:
                raise RuntimeError("Empty description generated")
                
        except Exception as ex:
            print(f"({idx}/{total}) {name}: FAILED - {ex}")
            print(f"ERROR: Failed to generate description for '{name}': {ex}")
            raise
            
        name_to_desc[name] = desc
        print(f"({idx}/{total}) {name}: done ({len(desc.split())} words)")
        
    return name_to_desc


def _rewrite_all_character_descriptions(name_to_desc: dict[str, str], story_desc: str, lm_studio_url: str) -> dict[str, str]:
    """Rewrite all character descriptions at once using Qwen model to make them look completely different while preserving primary features."""
    print(f"Rewriting all character descriptions using Qwen model: {len(name_to_desc)} characters")
    
    # Build combined prompt with all character descriptions
    characters_text = ""
    for name, desc in name_to_desc.items():
        characters_text += f"Character: {name}\nDescription: {desc}\n\n"
    
    prompt = f"""You are a creative character designer. Your task is to rewrite character descriptions to make each character look completely different while preserving their primary/core features and personality traits.

Story Context: {story_desc}

Current Character Descriptions:
{characters_text}

Instructions:
1. Rewrite each character's physical appearance to be completely different from the original
2. Keep all primary personality traits, motivations, and core characteristics
3. Ensure each character looks visually distinct from all others
4. Maintain the same level of detail and quality
5. Make the descriptions engaging and vivid

Return the rewritten descriptions in the same format as the input, with each character on a new line starting with "Character: [name]" followed by the new description."""

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
        
        rewritten_descriptions = structured_data.get("rewritten_descriptions", {})
        if not rewritten_descriptions:
            raise RuntimeError("No rewritten descriptions generated")
            
        # Validate that all characters are present
        if set(rewritten_descriptions.keys()) != set(name_to_desc.keys()):
            missing = set(name_to_desc.keys()) - set(rewritten_descriptions.keys())
            extra = set(rewritten_descriptions.keys()) - set(name_to_desc.keys())
            raise RuntimeError(f"Character mismatch - Missing: {missing}, Extra: {extra}")
            
        print(f"Successfully rewrote {len(rewritten_descriptions)} character descriptions")
        return rewritten_descriptions
        
    except Exception as ex:
        print(f"ERROR: Failed to rewrite character descriptions: {ex}")
        raise


def _generate_character_summaries(name_to_desc: dict[str, str], lm_studio_url: str) -> dict[str, str]:
    name_to_summary: dict[str, str] = {}
    total = len(name_to_desc)
    print(f"Generating character summaries: {total} total")
    
    for idx, (name, detailed_desc) in enumerate(name_to_desc.items(), 1):
        print(f"({idx}/{total}) {name}: generating summary...")
        prompt = _build_character_summary_prompt(name, detailed_desc)
        
        try:
            # Use structured output with JSON schema for summary
            user_payload = json.dumps({
                "character_name": name,
                "detailed_description": detailed_desc
            }, ensure_ascii=False)
            
            raw = _call_lm_studio(prompt, lm_studio_url, MODEL_CHARACTER_SUMMARY, user_payload, _schema_character_summary())
            structured_data = _parse_structured_response(raw)
            
            if not structured_data:
                raise RuntimeError("Failed to parse structured summary response")
            
            summary = structured_data.get("summary", "").strip()
            if not summary:
                raise RuntimeError("Empty summary generated")
                
        except Exception as ex:
            print(f"({idx}/{total}) {name}: FAILED - {ex}")
            print(f"ERROR: Failed to generate summary for '{name}': {ex}")
            raise
            
        name_to_summary[name] = summary
        print(f"({idx}/{total}) {name}: done ({len(summary.split())} words)")
        
    return name_to_summary


def _generate_story_description(story_content: str, lm_studio_url: str) -> str:
    """Generate story description from story content using LLM."""
    print("Generating story description from story content...")
    prompt = _build_story_description_prompt(story_content)
    
    try:
        # Use model constant for story description generation
        model = MODEL_STORY_DESCRIPTION
        
        # Use structured output with JSON schema for story description
        user_payload = json.dumps({
            "story_content": story_content  # Limit content to avoid token limits
        }, ensure_ascii=False)
        
        raw = _call_lm_studio(prompt, lm_studio_url, model, user_payload, _schema_story_description())
        structured_data = _parse_structured_response(raw)
        
        if not structured_data:
            raise RuntimeError("Failed to parse structured story description response")
        
        story_desc = structured_data.get("description", "").strip()
        if not story_desc:
            raise RuntimeError("Empty story description generated")
            
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


def _expand_locations(locations: dict[str, str], pairs: list, content: str, lm_studio_url: str) -> dict[str, str]:
    """Expand location descriptions using LLM to generate 500-character descriptions."""
    expanded_locations: dict[str, str] = {}
    total = len(locations)
    print(f"Expanding location descriptions: {total} total")
    
    # Use model constant for location expansion
    model = MODEL_LOCATION_EXPANSION
    
    for idx, (loc_id, original_desc) in enumerate(locations.items(), 1):
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
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    audio_story_path = os.path.normpath(os.path.join(base_dir, "../../gen.audio/input/1.story.txt"))
    image_story_path = os.path.normpath(os.path.join(base_dir, "../input/1.story.txt"))
    scene_out_path = os.path.normpath(os.path.join(base_dir, "../input/3.scene.txt"))
    characters_out_path = os.path.normpath(os.path.join(base_dir, "../input/2.character.txt"))
    character_summary_out_path = os.path.normpath(os.path.join(base_dir, "../input/3.character.txt"))
    locations_out_path = os.path.normpath(os.path.join(base_dir, "../input/3.location.txt"))

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
            story_desc = _generate_story_description(content, lm_studio_url)
            
            # Expand locations using the filtered story content and pairs
            expanded_locations = _expand_locations(locations, pairs, content, lm_studio_url)
            
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
                name_to_desc = _generate_character_descriptions(story_desc, unique_names, lm_studio_url)
                
                # Step 2: Optionally rewrite all character descriptions at once using Qwen model
                if ENABLE_CHARACTER_REWRITE:
                    print("Rewriting character descriptions to make them look completely different...")
                    final_descriptions = _rewrite_all_character_descriptions(name_to_desc, story_desc, lm_studio_url)
                else:
                    print("Character rewriting disabled - using original descriptions")
                    final_descriptions = name_to_desc
                
                # Step 3: Write the main character file with final descriptions
                char_errors = _validate_and_write_character_file(final_descriptions, characters_out_path, set(unique_names))
                if char_errors:
                    print(f"Character description output had {char_errors} error(s).")
                    return 1
                
                # Step 4: Generate character summaries from final descriptions
                name_to_summary = _generate_character_summaries(final_descriptions, lm_studio_url)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
