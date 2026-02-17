"""
Generate 2D assets from input.txt using ComfyUI Flux + Resize + Remove Background.

Input file: input/input.txt
Format per line: [asset_name]....description

Controllable: resize width, resize height, background-removal sensitivity.
"""
import os
import re
import json
import time
import shutil
import random
import argparse
from pathlib import Path

import requests

# Resize and sensitivity defaults (overridable via CLI)
DEFAULT_RESIZE_WIDTH = 100
DEFAULT_RESIZE_HEIGHT = 100
DEFAULT_SENSITIVITY = 0.2

# Paths relative to script dir (gen.2d/scripts)
INPUT_FILE = "../input/input.txt"
WORKFLOW_FILE = "../workflow/assets2d.json"
COMFYUI_OUTPUT_FOLDER = "../../ComfyUI/output"
FINAL_OUTPUT_DIR = "../output"


def _find_node_by_class(workflow: dict, class_types: str | list) -> list | None:
    """Return [node_id, output_index] for first node matching class_type(s)."""
    if isinstance(class_types, str):
        class_types = [class_types]
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") in class_types:
            return [node_id, 0]
    return None


def _update_node_input(workflow: dict, class_types: str | list, input_key: str, value) -> None:
    """Set input_key = value for all nodes matching class_types."""
    if isinstance(class_types, str):
        class_types = [class_types]
    for node_id, node in workflow.items():
        if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
            if node.get("class_type") in class_types and input_key in node["inputs"]:
                node["inputs"][input_key] = value


def _sanitize_filename(name: str) -> str:
    """Make asset name safe for filename prefix."""
    s = re.sub(r"[^\w\s\-.]", "", name)
    s = re.sub(r"\s+", "_", s).strip() or "asset"
    return s[:64]


def parse_input_file(path: str) -> list[tuple[str, str]]:
    """
    Parse input.txt. Format per line: [asset_name]....description
    Returns list of (asset_name, description).
    """
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "...." not in line:
                continue
            parts = line.split("....", 1)
            asset_name = parts[0].strip().strip("[]")
            description = parts[1].strip()
            if asset_name and description:
                entries.append((asset_name, description))
    return entries


class AssetGenerator:
    def __init__(
        self,
        comfyui_url: str = "http://127.0.0.1:8188/",
        resize_width: int = DEFAULT_RESIZE_WIDTH,
        resize_height: int = DEFAULT_RESIZE_HEIGHT,
        sensitivity: float = DEFAULT_SENSITIVITY,
    ):
        self.comfyui_url = comfyui_url.rstrip("/") + "/"
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.sensitivity = sensitivity
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_file = os.path.normpath(os.path.join(self.script_dir, INPUT_FILE))
        self.workflow_file = os.path.normpath(os.path.join(self.script_dir, WORKFLOW_FILE))
        self.comfyui_output_folder = os.path.normpath(
            os.path.join(self.script_dir, COMFYUI_OUTPUT_FOLDER)
        )
        self.final_output_dir = os.path.normpath(
            os.path.join(self.script_dir, FINAL_OUTPUT_DIR)
        )
        os.makedirs(self.final_output_dir, exist_ok=True)

    def _load_workflow(self) -> dict | None:
        try:
            with open(self.workflow_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"ERROR: Failed to load workflow: {e}")
            return None

    def _apply_workflow_settings(
        self,
        workflow: dict,
        asset_name: str,
        description: str,
    ) -> dict:
        """Set prompt, resize width/height, sensitivity, and output filename."""
        # Prompt
        _update_node_input(
            workflow,
            ["CLIPTextEncode", "CLIP Text Encode (Prompt)"],
            "text",
            description,
        )
        # Resize
        _update_node_input(workflow, "ImageResizeKJv2", "width", self.resize_width)
        _update_node_input(workflow, "ImageResizeKJv2", "height", self.resize_height)
        # Remove Background sensitivity
        _update_node_input(workflow, "RMBG", "sensitivity", self.sensitivity)
        # Output filename prefix (asset name, sanitized)
        prefix = _sanitize_filename(asset_name)
        _update_node_input(workflow, "SaveImage", "filename_prefix", prefix)
        # Random seed for variation
        seed = random.randint(0, 2**32 - 1)
        ksampler = _find_node_by_class(workflow, "KSampler")
        if ksampler:
            node_id = ksampler[0]
            if node_id in workflow and "inputs" in workflow[node_id]:
                workflow[node_id]["inputs"]["seed"] = seed
        return workflow

    def _find_newest_output_with_prefix(self, prefix: str) -> str | None:
        if not os.path.isdir(self.comfyui_output_folder):
            return None
        latest = None
        latest_mtime = -1.0
        for root, _dirs, files in os.walk(self.comfyui_output_folder):
            for name in files:
                if name.startswith(prefix) and Path(name).suffix.lower() in {
                    ".png", ".jpg", ".jpeg", ".webp",
                }:
                    full = os.path.join(root, name)
                    try:
                        mtime = os.path.getmtime(full)
                    except OSError:
                        continue
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                        latest = full
        return latest

    def generate_one(self, asset_name: str, description: str) -> str | None:
        workflow = self._load_workflow()
        if not workflow:
            return None
        workflow = self._apply_workflow_settings(workflow, asset_name, description)
        prefix = _sanitize_filename(asset_name)

        resp = requests.post(
            f"{self.comfyui_url}prompt",
            json={"prompt": workflow},
            timeout=60,
        )
        if resp.status_code != 200:
            print(f"ERROR: ComfyUI API: {resp.status_code} {resp.text}")
            return None
        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            print("ERROR: No prompt_id from ComfyUI")
            return None

        while True:
            h = requests.get(f"{self.comfyui_url}history/{prompt_id}", timeout=10)
            if h.status_code == 200:
                data = h.json()
                if prompt_id in data:
                    status = data[prompt_id].get("status", {})
                    if status.get("exec_info", {}).get("queue_remaining", 0) == 0:
                        time.sleep(2)
                        break
            time.sleep(2)

        generated = self._find_newest_output_with_prefix(prefix)
        if not generated:
            print(f"ERROR: No output found for prefix: {prefix}")
            return None
        final_path = os.path.join(
            self.final_output_dir,
            f"{prefix}.png",
        )
        shutil.copy2(generated, final_path)
        print(f"Saved: {final_path}")
        return final_path

    def generate_all(self) -> dict[str, str]:
        entries = parse_input_file(self.input_file)
        if not entries:
            print(f"No entries in {self.input_file} (format: [asset_name]....description)")
            return {}
        print(f"Resize: {self.resize_width}x{self.resize_height}, Sensitivity: {self.sensitivity}")
        print(f"Processing {len(entries)} assets from {self.input_file}")
        results = {}
        for asset_name, description in entries:
            print(f"Generating: {asset_name}")
            path = self.generate_one(asset_name, description)
            if path:
                results[asset_name] = path
        return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate 2D assets from input/input.txt (format: [asset_name]....description).",
    )
    parser.add_argument(
        "--width", "-W",
        type=int,
        default=DEFAULT_RESIZE_WIDTH,
        help=f"Resize width (default: {DEFAULT_RESIZE_WIDTH})",
    )
    parser.add_argument(
        "--height", "-H",
        type=int,
        default=DEFAULT_RESIZE_HEIGHT,
        help=f"Resize height (default: {DEFAULT_RESIZE_HEIGHT})",
    )
    parser.add_argument(
        "--sensitivity", "-s",
        type=float,
        default=DEFAULT_SENSITIVITY,
        help=f"Background removal sensitivity 0–1 (default: {DEFAULT_SENSITIVITY})",
    )
    parser.add_argument(
        "--comfyui-url",
        default="http://127.0.0.1:8188/",
        help="ComfyUI API base URL",
    )
    args = parser.parse_args()

    gen = AssetGenerator(
        comfyui_url=args.comfyui_url,
        resize_width=args.width,
        resize_height=args.height,
        sensitivity=args.sensitivity,
    )
    results = gen.generate_all()
    if results:
        print(f"\nGenerated {len(results)} assets:")
        for name, path in results.items():
            print(f"  {name}: {path}")
        return 0
    print("No assets generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
