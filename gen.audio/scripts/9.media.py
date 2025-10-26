import os
import re
import json
import math
import time
import requests
from typing import List, Dict, Tuple
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

# Model constants for easy switching
MODEL_MEDIA_TAGS = "qwen3-30b-a3b-thinking-2507"  # Model for generating YouTube tags
MODEL_MEDIA_TITLE = "qwen3-30b-a3b-thinking-2507"  # Model for generating YouTube titles
MODEL_MEDIA_HOOK = "qwen3-30b-a3b-thinking-2507"  # Model for generating YouTube hooks
MODEL_MEDIA_BULLETS = "qwen3-30b-a3b-thinking-2507"  # Model for generating YouTube bullet points
MODEL_DESCRIPTION_GENERATION = "qwen3-30b-a3b-thinking-2507"  # Model for description generation
class DiffusionPromptGenerator:
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1", model: str = MODEL_DESCRIPTION_GENERATION):
        self.lm_studio_url = lm_studio_url
        self.model = model
        self.input_file = "../input/9.summary.txt"
        self.output_file = "../input/10.thumbnail.txt"

    def _read_text(self, path: str) -> str | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                return text if text else None
        except Exception:
            return None

    def _write_text(self, path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _schema_prompt(self) -> Dict[str, object]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "diffusion_prompt",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "description": "Image generation prompt for thumbnail",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "minLength": 60,
                            "maxLength": 150,
                            "description": "Single continuous paragraph with 300-350 words describing the thumbnail scene"
                        }
                    },
                    "required": ["prompt"],
                },
                "strict": True,
            },
        }

    def _build_system_prompt(self) -> str:
        return """You are a visual director CREATIVELY generating one Image Generation Model Prompt for Thumbnail within the word limit of 300-350 words from the following story summary.
        
        CONSTRAINTS: 
         - highly specific spatial and material details, and technical quality flags. 
         - Include: main character(s) with detailed physical descriptions and clothing positioned specifically in the scene (center-left, background-right, etc.)
         - the central object or narrative focus placed precisely in the composition with detailed condition and appearance
         - the setting environment with exact spatial descriptions of furniture, walls, windows, and atmospheric elements
         - secondary characters positioned clearly with actions and props
         - background elements like weather, time period indicators, and contextual details
         - all object positions using directional terms (left wall, center focus, far background)
         - precise material descriptions for textures and surfaces (dark oak, brass fittings, weathered leather)
         - Ensure every element supports the story and maintains spatial clarity and visual coherence.
         
         Output must be a CREATIVELY generated single continuous paragraph within the word limit of 300-350 words without line breaks."""

    def _build_user_prompt(self, story_desc: str) -> str:
        return f"""STORY SUMMARY: {story_desc}"""

    def _call_lm_studio(self, system_prompt: str, user_prompt: str, response_format: Dict[str, object] | None = None) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt + "\nOnly use English Language for Input, Thinking, and Output\n/no_think"},
                {"role": "user", "content": user_prompt + "\nOnly use English Language for Input, Thinking, and Output\n/no_think"},
            ],
            "temperature": 1,
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        resp = requests.post(f"{self.lm_studio_url}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"LM Studio API error: {resp.status_code} {resp.text}")
        data = resp.json()
        if not data.get("choices"):
            raise RuntimeError("LM Studio returned no choices")
        content = data["choices"][0]["message"]["content"]
        return content

    def _parse_structured_response(self, content: str) -> Dict[str, object] | None:
        text = content.strip()
        if text.startswith("```"):
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
            if m:
                text = m.group(1).strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def _sanitize_single_paragraph(self, text: str) -> str:
        if not text:
            return ""
        # Collapse newlines/tabs and excessive whitespace
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def generate_and_save(self) -> str | None:
        story_desc = self._read_text(self.input_file)
        if not story_desc:
            print("ERROR: No story description found in 9.summary.txt")
            return None

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(story_desc)

        try:
            raw = self._call_lm_studio(system_prompt, user_prompt, response_format=self._schema_prompt())
            data = self._parse_structured_response(raw)
            
            if not data or "prompt" not in data:
                raise RuntimeError("Invalid structured response from LM Studio")
            
            prompt = self._sanitize_single_paragraph(str(data["prompt"]))
            if not prompt:
                raise RuntimeError("Empty prompt generated")
        except Exception as e:
            print(f"ERROR: LM Studio generation failed: {e}")
            return None

        # Ensure no line breaks
        prompt = self._sanitize_single_paragraph(prompt)

        word_count = len(prompt.split())
        print(f"Characters: {len(prompt)}")
        print(f"Words: {word_count}")
        
        if word_count < 300 or word_count > 350:
            print(f"⚠️ WARNING: Word count ({word_count}) is outside the 300-350 word limit")

        out_path = self.output_file
        self._write_text(out_path, prompt)
        return out_path


class YouTubeDescriptionGenerator:
    def __init__(
        self,
        lm_studio_url: str = "http://localhost:1234/v1",
        model: str = MODEL_MEDIA_TAGS,
        diffusion_file: str = "../input/9.summary.txt",
        title_file: str = "../input/10.title.txt",
        chapters_file: str = "../input/12.chapters.txt",
        output_file: str = "../output/description.txt",
        tags_output_file: str = "../output/tags.txt",
        num_chapters: int = 8,
        hook_char_limit: int = 500,
    ) -> None:
        self.lm_studio_url = lm_studio_url
        self.model = model
        self.diffusion_file = diffusion_file
        self.title_file = title_file
        self.chapters_file = chapters_file
        self.output_file = output_file
        self.tags_output_file = tags_output_file
        self.num_chapters = max(3, int(num_chapters))
        self.hook_char_limit = max(120, int(hook_char_limit))

    def _read_text(self, path: str) -> str | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
                return txt if txt else None
        except Exception:
            return None

    def _write_text(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _parse_chapters_file(self) -> List[Dict[str, object]]:
        """Parse chapters.txt file to extract percentage, title, and summary for each chapter."""
        chapters_text = self._read_text(self.chapters_file)
        if not chapters_text:
            return []
        
        chapters = []
        lines = chapters_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Parse format: "28%: The Cyclist's Shadow"
            # Next line: "SHORT_SUMMARY : Description..."
            if ':' in line and '%' in line:
                # Extract percentage and title
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    percentage_part = parts[0].strip()
                    title = parts[1].strip()
                    
                    # Extract percentage number
                    percentage_match = re.search(r'(\d+)%', percentage_part)
                    if percentage_match:
                        percentage = int(percentage_match.group(1))
                        chapters.append({
                            'percentage': percentage,
                            'title': title,
                            'summary': ''  # Will be filled by next line if it's a summary
                        })
            elif line.startswith('SHORT_SUMMARY') and chapters:
                # Extract summary for the last chapter
                summary_parts = line.split(':', 1)
                if len(summary_parts) >= 2:
                    summary = summary_parts[1].strip()
                    chapters[-1]['summary'] = summary
        
        return chapters

    def _generate_chapters_from_file(self, total_seconds: float) -> List[Dict[str, object]]:
        """Generate chapters with timestamps based on percentages from chapters file."""
        file_chapters = self._parse_chapters_file()
        if not file_chapters:
            return []
        
        # Convert percentages to timestamps - keep original order from file
        formatted_chapters = []
        for i, chapter in enumerate(file_chapters):
            percentage = chapter['percentage']
            title = chapter['title']
            summary = chapter['summary']
            
            if i == 0:
                # First chapter always starts at 00:00
                timestamp_seconds = 0.0
            else:
                # Calculate timestamp based on percentage of total duration
                # Note: percentage represents how much of the story is remaining, 
                # so we need to convert it to elapsed time
                elapsed_percentage = 100 - percentage
                timestamp_seconds = (elapsed_percentage / 100.0) * total_seconds
            
            formatted_chapters.append({
                'timestamp': self._format_chapter_time(timestamp_seconds),
                'title': title,
                'description': summary
            })
        
        return formatted_chapters

    def _estimate_total_seconds_from_text(self, text: str, min_minutes: int = 8, max_minutes: int = 60, words_per_minute: int = 160) -> float:
        words = re.findall(r"\w+", text or "")
        est_minutes = max(min_minutes, min(max_minutes, int(math.ceil(len(words) / max(1, words_per_minute)))))
        return float(est_minutes * 60)

    def _format_chapter_time(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        hh = seconds // 3600
        mm = (seconds % 3600) // 60
        ss = seconds % 60
        if hh > 0:
            return f"{hh:d}:{mm:02d}:{ss:02d}"
        return f"{mm:02d}:{ss:02d}"

    def _get_audio_duration_seconds(self, audio_path: str = "../output/final.wav") -> float | None:
        try:
            import wave
            with wave.open(audio_path, 'rb') as w:
                frames = w.getnframes()
                fr = w.getframerate()
                if fr and fr > 0:
                    return frames / float(fr)
        except Exception:
            return None
        return None

    def _call_lm_studio(self, system_prompt: str, user_payload: str, response_format: Dict[str, object] | None = None, model: str = None) -> str:
        headers = {"Content-Type": "application/json"}
        body = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt + "\nOnly use English Language for Input, Thinking, and Output\n/no_think"},
                {"role": "user", "content": user_payload + "\nOnly use English Language for Input, Thinking, and Output\n/no_think"},
            ],
            "temperature": 1,
            "stream": False,
        }
        if response_format is not None:
            body["response_format"] = response_format
        resp = requests.post(f"{self.lm_studio_url}/chat/completions", headers=headers, json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"LM Studio API error: {resp.status_code} {resp.text}")
        data = resp.json()
        if not data.get("choices"):
            raise RuntimeError("LM Studio returned no choices")
        return data["choices"][0]["message"]["content"]

    def _parse_structured_response(self, content: str) -> Dict[str, object] | None:
        text = content.strip()
        if text.startswith("```"):
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
            if m:
                text = m.group(1).strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def _schema_tags(self) -> Dict[str, object]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "tags",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "description": "Tags for a YouTube video",
                    "maxLength": 500,
                    "minLength": 475,
                    "properties": {
                        "core_sherlock_holmes_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 15,
                            "maxItems": 10,
                            "description": "Core Sherlock Holmes Story terms"
                        },
                        "audience_targeting": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 30,
                            "maxItems": 30,
                            "description": "Tags targeting mystery fans, audiobook listeners, commuters"
                        },
                        "story_specific_elements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 15,
                            "maxItems": 15,
                            "description": "Story-specific plot and character elements"
                        }
                    },
                    "required": ["core_sherlock_holmes_terms", "audience_targeting", "audio_format_appeal", "story_specific_elements"],
                },
                "strict": True,
            },
        }

    def _gen_tags_initial(self, title: str, summary: str) -> List[str]:
        sys = (
            "TASK: Using the story title and summary, create 60 YouTube tags for an audio story channel.\n"
            "- No repeated words across all tags.\n"
            "- Include story-specific characters/plot elements.\n"
            "- Target: mystery fans, audiobook listeners, commuters.\n"
            "- Mix popular + niche terms for discovery.\n"
            "- Two/One word tags only.\n\n"
            "- core_sherlock_holmes_terms: 15 tags (Sherlock Holmes, Watson, detective, mystery, etc.)\n"
            "- audience_targeting: 20 tags (audiobook, podcast, commute, bedtime story, etc.)\n"
            "- story_specific_elements: 15 tags (specific plot points, characters, locations from the story)\n\n"
            "Return JSON with the four arrays as specified in the schema; no commentary."
        )
        payload = {"title": title, "summary": summary}
        raw = self._call_lm_studio(sys, json.dumps(payload, ensure_ascii=False), response_format=self._schema_tags(), model=MODEL_MEDIA_TAGS)
        obj = self._parse_structured_response(raw) or {}
        
        # Combine all tag categories into a single list
        all_tags = []
        for category in ["core_sherlock_holmes_terms", "audience_targeting", "audio_format_appeal", "story_specific_elements"]:
            category_tags = obj.get(category, [])
            all_tags.extend([str(t).strip() for t in category_tags if str(t).strip()])
        
        return all_tags

    def _normalize_and_trim_tags(self, tags: List[str]) -> List[str]:
        # enforce one/two words, remove duplicates, keep order
        seen = set()
        cleaned: List[str] = []
        for t in tags:
            t = re.sub(r"\s+", " ", t.strip())
            words = t.split(" ")
            if 1 <= len(words) <= 2:
                key = t.lower()
                if key not in seen:
                    seen.add(key)
                    cleaned.append(t)
        return cleaned

    def _deduplicate_by_words(self, tags: List[str]) -> List[str]:
        used_words = set()
        result: List[str] = []
        for t in tags:
            words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", t)]
            if any(w in used_words for w in words):
                continue
            for w in words:
                used_words.add(w)
            result.append(t)
        return result

    def _fit_to_500_chars(self, tags: List[str]) -> List[str]:
        # drop from end until comma-joined length <= 500
        while tags and len(", ".join(tags)) > 500:
            tags.pop()
        return tags

    def _render_tags_line(self, tags: List[str]) -> str:
        line = ", ".join(tags)
        return f"{line}"

    def _render_description(self, parts: Dict[str, object]) -> str:
        title_line = str(parts.get("title_line", "")).strip()
        hook = str(parts.get("hook", "")).strip()
        bullets = parts.get("bullets") or []
        chapters = parts.get("chapters") or []
        ctas = parts.get("ctas") or []
        hashtags = str(parts.get("hashtags", "")).strip()

        lines: List[str] = []
        if title_line:
            lines.append(title_line)
        if hook:
            lines.append("")
            lines.append(hook)

        if bullets:
            lines.append("")
            for b in bullets:
                lines.append(str(b))
        lines.append("")
        lines.append("📜 Chapters:")
        lines.append("")
        for ch in chapters:
            ts = str(ch.get("timestamp", "")).strip()
            ti = str(ch.get("title", "")).strip()
            desc = str(ch.get("description", "")).strip()
            if ts and ti:
                lines.append(f"{ts} – {ti}")
                if desc:
                    lines.append(f"   {desc}")
        lines.append("")
        for c in ctas:
            lines.append(str(c))
        if hashtags:
            lines.append("")
            lines.append(hashtags)
        return "\n".join([ln for ln in lines if ln is not None])

    # ---------- Multi-call schemas ----------
    def _schema_title(self) -> Dict[str, object]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "title_line",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "genre": {"type": "string"}
                    },
                    "required": ["title", "genre"],
                },
                "strict": True,
            },
        }

    def _schema_hook(self) -> Dict[str, object]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "hook_line",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"hook": {"type": "string"}},
                    "required": ["hook"],
                },
                "strict": True,
            },
        }

    def _schema_bullets(self) -> Dict[str, object]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "bullets",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "bullets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 5,
                        }
                    },
                    "required": ["bullets"],
                },
                "strict": True,
            },
        }

    def _schema_chapters(self) -> Dict[str, object]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "chapters",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "chapters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "timestamp": {"type": "string"},
                                    "title": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["timestamp", "title", "description"],
                            },
                            "minItems": 5,
                            "maxItems": 16,
                        }
                    },
                    "required": ["chapters"],
                },
                "strict": True,
            },
        }

    def _parse_time_seconds(self, ts: str) -> float | None:
        ts = (ts or "").strip()
        m3 = re.match(r"^(\d+):(\d{1,2}):(\d{2})$", ts)
        if m3:
            h, m, s = map(int, m3.groups())
            if 0 <= m <= 59 and 0 <= s <= 59:
                return float(h * 3600 + m * 60 + s)
        m2 = re.match(r"^(\d{1,2}):(\d{2})$", ts)
        if m2:
            m, s = map(int, m2.groups())
            if 0 <= m and 0 <= s <= 59:
                return float(m * 60 + s)
        return None

    def _postprocess_chapters(self, chapters: List[Dict[str, object]], total_seconds: float) -> List[Dict[str, object]]:
        # Simplified postprocessing for file-based chapters
        # The file-based generation already handles formatting and ordering
        return chapters

    # ---------- Part generators ----------
    def _gen_title_line(self, title: str, summary: str) -> str:
        sys = (
            "You are a YouTube content editor. Generate a title and genre/content type label for YouTube.\n"
            "The title should start with an emoji and be engaging.\n"
            "The genre should be a concise content type/genre label (e.g., 'Mystery Audiobook', 'Detective Story', 'Audio Drama').\n"
            "Keep both concise and appropriate to the summary."
        )
        payload = {"title": title, "summary": summary}
        raw = self._call_lm_studio(sys, json.dumps(payload, ensure_ascii=False), response_format=self._schema_title(), model=MODEL_MEDIA_TITLE)
        obj = self._parse_structured_response(raw) or {}
        title_part = str(obj.get("title", title)).strip()
        genre_part = str(obj.get("genre", "Audiobook")).strip()
        return f"{title_part} | {genre_part}"

    def _gen_hook(self, summary: str, hook_limit: int) -> str:
        sys = (
            f"Write a single-sentence hook (<= {hook_limit} chars), engaging and spoiler-light, starting with an emoji. Return JSON."
        )
        payload = {"summary": summary, "constraints": {"max_chars": hook_limit}}
        raw = self._call_lm_studio(sys, json.dumps(payload, ensure_ascii=False), response_format=self._schema_hook(), model=MODEL_MEDIA_HOOK)
        obj = self._parse_structured_response(raw) or {}
        hook = str(obj.get("hook", "")).strip()
        if len(hook) > hook_limit:
            hook = hook[:hook_limit - 1].rstrip() + "…"
        return hook

    def _gen_bullets(self, summary: str) -> List[str]:
        sys = (
            "Produce 3–5 concise bullet lines starting with an emoji, highlighting appeal and features. Return JSON."
        )
        payload = {"summary": summary}
        raw = self._call_lm_studio(sys, json.dumps(payload, ensure_ascii=False), response_format=self._schema_bullets(), model=MODEL_MEDIA_BULLETS)
        obj = self._parse_structured_response(raw) or {}
        bullets = obj.get("bullets") or []
        return [str(b).strip() for b in bullets if str(b).strip()]

    def _gen_chapters(self, summary: str, total_seconds: float) -> List[Dict[str, object]]:
        # Use file-based chapter generation instead of LLM
        return self._generate_chapters_from_file(total_seconds)

    def _gen_ctas(self) -> List[str]:
        # Return fixed CTAs (no LLM call)
        return [
            "👍 Like this episode if you loved the mystery!",
            "🔔 Subscribe for more classic-style mysteries, full-length audiobooks, and immersive soundscapes.",
            "💬 Tell us your theories in the comments below!",
        ]

    def _gen_hashtags(self, summary: str) -> str:
        # Return fixed hashtags (no LLM call)
        return (
            "#SherlockHolmes #AudioDrama #MysteryStory #VictorianLondon #DetectiveFiction #ClassicLiterature "
            "#Audiobook #CrimeMystery #221BBakerStreet #DrWatson #ConanDoyle #Suspense #FullAudiobook "
            "#VictorianMystery #Case"
        )

    def generate_and_save(self) -> str | None:
        print("🔍 Reading diffusion text...")
        diffusion_text = self._read_text(self.diffusion_file)
        if not diffusion_text:
            print(f"ERROR: No diffusion text found at {self.diffusion_file}")
            return None
        print(f"✅ Read {len(diffusion_text)} characters from diffusion file")

        print("📖 Reading title...")
        title = self._read_text(self.title_file) or "Untitled Story"
        print(f"✅ Title: {title}")

        print("⏱️ Calculating audio duration...")
        audio_seconds = self._get_audio_duration_seconds("../output/final.wav")
        # Determine total duration
        total_seconds = float(audio_seconds) if (audio_seconds and audio_seconds > 0) else self._estimate_total_seconds_from_text(diffusion_text)
        if not total_seconds or total_seconds <= 0:
            total_seconds = 8 * 60.0
        print(f"✅ Total duration: {self._format_chapter_time(total_seconds)} ({total_seconds:.1f}s)")

        # Multi-call generation for higher quality
        try:
            print("🎯 Generating title line...")
            title_line = self._gen_title_line(title, diffusion_text)
            print(f"✅ Title line: {title_line}")

            print("🪝 Generating hook...")
            hook = self._gen_hook(diffusion_text, self.hook_char_limit)
            print(f"✅ Hook: {hook}")

            print("📋 Generating bullet points...")
            bullets = self._gen_bullets(diffusion_text)
            print(f"✅ Generated {len(bullets)} bullet points")

            print("📚 Generating chapters from file...")
            chapters = self._gen_chapters(diffusion_text, total_seconds)
            print(f"✅ Generated {len(chapters)} chapters")
            for ch in chapters:
                print(f"   {ch['timestamp']} - {ch['title']}")

            print("📢 Generating CTAs...")
            ctas = self._gen_ctas()
            print(f"✅ Generated {len(ctas)} CTAs")

            print("🏷️ Generating hashtags...")
            hashtags = self._gen_hashtags(diffusion_text)
            print(f"✅ Hashtags: {hashtags}")

            parts = {
                "title_line": title_line,
                "hook": hook,
                "bullets": bullets,
                "chapters": chapters,
                "ctas": ctas,
                "hashtags": hashtags,
            }

            print("📝 Rendering final description...")
            out = self._render_description(parts)
            print(f"✅ Description rendered ({len(out)} characters)")
        except Exception as e:
            print(f"ERROR: LM Studio generation failed: {e}")
            return None

        print(f"💾 Saving description to {self.output_file}...")
        self._write_text(self.output_file, out)
        print("✅ Description saved")

        try:
            print("🏷️ Generating YouTube tags...")
            tags = self._gen_tags_initial(title, diffusion_text)
            print(f"✅ Generated {len(tags)} initial tags")
            
            print("🧹 Normalizing and trimming tags...")
            tags = self._normalize_and_trim_tags(tags)
            print(f"✅ After normalization: {len(tags)} tags")
            
            print("🔄 Deduplicating tags...")
            tags = self._deduplicate_by_words(tags)
            print(f"✅ After deduplication: {len(tags)} tags")
            
            print("✂️ Fitting to 500 character limit...")
            tags = self._fit_to_500_chars(tags)
            print(f"✅ Final tags: {len(tags)} tags ({len(', '.join(tags))} chars)")
            
            tags_line = self._render_tags_line(tags)
            print(f"💾 Saving tags to {self.tags_output_file}...")
            self._write_text(self.tags_output_file, tags_line)
            print("✅ Tags saved")
        except Exception as te:
            print(f"WARNING: Tag generation failed: {te}")
        return self.output_file


def main() -> int:
    start = time.time()
    
    # Generate thumbnail prompt first
    print("🖼️ Generating thumbnail prompt...")
    thumbnail_gen = DiffusionPromptGenerator()
    thumbnail_result = thumbnail_gen.generate_and_save()
    if thumbnail_result:
        print(f"✅ Thumbnail prompt generated: {thumbnail_result}")
    else:
        print("⚠️ Thumbnail prompt generation failed, continuing with YouTube description...")
    
    # Generate YouTube description
    print("📝 Generating YouTube description...")
    gen = YouTubeDescriptionGenerator()
    out = gen.generate_and_save()
    if not out:
        return 1
    print(f"Saved YouTube description to: {out} (took {time.time() - start:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


