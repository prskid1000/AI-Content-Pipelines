import os
import re
import json
import time
import requests
from functools import partial
import builtins as _builtins
print = partial(_builtins.print, flush=True)

# Model constants for easy switching
MODEL_DESCRIPTION_GENERATION = "qwen/qwen3-14b"  # Model for description generation

class DiffusionPromptGenerator:
    def __init__(self, lm_studio_url: str = "http://localhost:1234/v1", model: str = MODEL_DESCRIPTION_GENERATION):
        self.lm_studio_url = lm_studio_url
        self.model = model
        self.input_file = "../input/9.description.txt"
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

    def _build_system_prompt(self) -> str:
        return """You are a visual director CREATIVELY generating one Image Generation Model Prompt of 300-500 words for the following STORY TITLE/DESCRIPTION.
        
        CONSTRAINTS: 
         - highly specific spatial and material details, and technical quality flags. 
         - Include: main character(s) with detailed physical descriptions and clothing positioned specifically in the scene (center-left, background-right, etc.)
         - the central object or narrative focus placed precisely in the composition with detailed condition and appearance
         - the setting environment with exact spatial descriptions of furniture, walls, windows, and atmospheric elements
         - secondary characters positioned clearly with actions and props
         - background elements like weather, time period indicators, and contextual details
         - all object positions using directional terms (left wall, center focus, far background)
         - precise material descriptions for textures and surfaces (dark oak, brass fittings, weathered leather)

        Ensure every element supports the story and maintains spatial clarity and visual coherence. Output must be a CREATIVELY generated single continuous paragraph of 300-500 words without line breaks."""

    def _build_user_prompt(self, story_desc: str) -> str:
        return f"""STORY TITLE/DESCRIPTION: {story_desc}"""

    def _call_lm_studio(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt + "/no_think /no_think"},
                {"role": "user", "content": user_prompt + "/no_think /no_think"},
            ],
            "temperature": 1,
            "stream": False,
        }

        resp = requests.post(f"{self.lm_studio_url}/chat/completions", headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"LM Studio API error: {resp.status_code} {resp.text}")
        data = resp.json()
        if not data.get("choices"):
            raise RuntimeError("LM Studio returned no choices")
        content = data["choices"][0]["message"]["content"]
        return content

    def _sanitize_single_paragraph(self, text: str) -> str:
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

    def generate_and_save(self) -> str | None:
        story_desc = self._read_text(self.input_file)
        if not story_desc:
            print("ERROR: No story description found in 9.description.txt")
            return None

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(story_desc)

        try:
            raw = self._call_lm_studio(system_prompt, user_prompt)
            prompt = self._sanitize_single_paragraph(raw)
            if not prompt:
                raise RuntimeError("Empty prompt generated")
        except Exception as e:
            print(f"ERROR: LM Studio generation failed: {e}")
            return None

        # Ensure no line breaks
        prompt = self._sanitize_single_paragraph(prompt)

        print(f"Length: {len(prompt)}")
        print(f"Tokens: {len(prompt.split())}")

        out_path = self.output_file
        self._write_text(out_path, prompt)
        return out_path


def main() -> int:
    start = time.time()
    gen = DiffusionPromptGenerator()
    out = gen.generate_and_save()
    if not out:
        return 1
    print(f"Saved thumbnail prompt to: {out} (took {time.time() - start:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


