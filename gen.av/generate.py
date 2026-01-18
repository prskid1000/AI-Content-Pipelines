import os
import sys
import time
import subprocess
import signal
import shlex
import shutil
import socket
import urllib.request
import urllib.error
from urllib.parse import urlparse, urljoin
import atexit
import hashlib
import json


# Simple global log timing stats
LOG_STATS = {
    "write_seconds": 0.0,
    "flush_seconds": 0.0,
}


class TimedLogWriter:
    """Wraps a file handle and tracks time spent in write/flush operations.

    Note: When used as stdout/stderr for subprocesses, the child process writes
    directly to the underlying OS file descriptor, so those writes are not
    attributed here. This measures only writes/flushes performed by this
    Python process plus log maintenance time tracked separately.
    """

    def __init__(self, file_handle):
        self._fh = file_handle

    def write(self, s):
        start = time.perf_counter()
        try:
            return self._fh.write(s)
        finally:
            LOG_STATS["write_seconds"] += time.perf_counter() - start

    def flush(self):
        start = time.perf_counter()
        try:
            return self._fh.flush()
        finally:
            LOG_STATS["flush_seconds"] += time.perf_counter() - start

    # Pass-throughs needed for compatibility
    def fileno(self):
        return self._fh.fileno()

    def seek(self, offset, whence=0):
        return self._fh.seek(offset, whence)

    def close(self):
        return self._fh.close()

    def __getattr__(self, name):
        return getattr(self._fh, name)


SCRIPTS = [
    #Story
    # "1.story.py",
    # "../gen.image/scripts/1.story.py",
    # "../gen.audio/scripts/1.character.py",

    #Images
    # "../gen.image/scripts/2.character.py",
    # "../gen.image/scripts/2.location.py",
    "../gen.image/scripts/3.scene.py",

    #Video
    # "../gen.video/scripts/2.motion.py",
    # "3.av.py",

    #Thumbnail
    # "../gen.audio/scripts/9.media.py",
    # "../gen.audio/scripts/10.thumbnail.py",

    # YouTube
    # "4.video.py",
    # "../gen.audio/scripts/11.video.py",
    # "../gen.audio/scripts/12.youtube.py"
]

SCRIPTS_DIR = "scripts"

NEEDS_COMFYUI = {"3.av.py", "2.story.py", "2.character.py", "3.scene.py", "7.sfx.py", "10.thumbnail.py", "3.animate.py", "2.location.py"}
NEEDS_LMSTUDIO = {"1.character.py", "1.story.py", "2.motion.py", "5.timeline.py", "6.timing.py", "9.media.py"}

# Centralized non-interactive defaults (only change this file)
SCRIPT_ARGS = {
    # "1.story.py": ["--bypass-validation"],
    "1.character.py": ["--auto-gender", "m", "--auto-confirm", "y", "--change-settings", "n"],
    "2.motion.py": ["--output", "../../gen.av/input/2.motion.txt"],  # Output path for AV motion prompts (relative to gen.video/scripts/)
    "11.video.py": ["--video-file", "../../gen.av/output/final.mp4"],
    "12.youtube.py": ["--video-file", "../../gen.av/output/final.mp4", "--upload-shorts", "--shorts-dir", "../../gen.audio/output"],
}


def resolve_comfyui_dir(base_dir: str) -> str:
    candidate = os.path.abspath(os.path.join(base_dir, "..", "ComfyUI"))
    if os.path.exists(os.path.join(candidate, "main.py")):
        return candidate
    alt = os.environ.get("COMFYUI_DIR")
    if alt and os.path.exists(os.path.join(alt, "main.py")):
        return alt
    return candidate


def calculate_file_hash(filepath: str) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as ex:
        raise RuntimeError(f"Failed to hash file {filepath}: {ex}")


def check_and_clean_tracking_if_files_changed(
    base_dir: str,
    log_handle,
    tracked_files: list[tuple[str, str]],  # List of (file_path, hash_key) tuples
    tracking_json_file: str,
    output_dirs: list[str] = None  # List of output directories to clear if files changed (defaults to tracking dir parent)
    ) -> bool:
    """
    Check if any tracked file has changed and delete output directory if it has.
    
    Args:
        base_dir: Base directory for relative paths
        log_handle: Log file handle
        tracked_files: List of tuples (relative_file_path, hash_key) to track
        tracking_json_file: Path to tracking JSON file
        output_dir: Output directory to clear if files changed (defaults to tracking dir parent)
    
    Returns:
        bool: True if successful, False on error
    """
    if not tracked_files:
        log_handle.write("No files to track. Skipping tracking check.\n")
        log_handle.flush()
        return True
    
    # Calculate current hashes for all tracked files
    current_hashes = {}
    for rel_path, hash_key in tracked_files:
        file_path = os.path.join(base_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
        
        # Skip if file doesn't exist
        if not os.path.exists(file_path):
            log_handle.write(f"Tracked file not found: {rel_path}. Skipping.\n")
            log_handle.flush()
            continue
        
        try:
            current_hashes[hash_key] = calculate_file_hash(file_path)
        except RuntimeError as ex:
            log_handle.write(f"ERROR: {ex}\n")
            log_handle.flush()
            return False
    
    # If no files were hashed successfully, nothing to do
    if not current_hashes:
        log_handle.write("No valid tracked files found. Skipping tracking check.\n")
        log_handle.flush()
        return True
    
    tracking_dir = os.path.dirname(tracking_json_file)
    
    # Determine output directories to clear
    if output_dirs is None:
        # Default to parent of tracking directory (e.g., output/)
        output_dirs = [os.path.dirname(tracking_dir)]
    
    # If tracking directory doesn't exist, create it with tracking JSON
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir, exist_ok=True)
        tracking_data = {
            "file_hashes": current_hashes,
            "last_checked": time.time(),
            "tracked_files": {key: rel_path for rel_path, key in tracked_files}
        }
        try:
            with open(tracking_json_file, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2)
            log_handle.write("Tracking directory created with file hashes.\n")
        except Exception as ex:
            log_handle.write(f"WARNING: Failed to save tracking JSON: {ex}\n")
        log_handle.flush()
        return True
    
    # Check if tracking JSON exists
    if not os.path.exists(tracking_json_file):
        # No tracking JSON means old tracking format, delete output and recreate
        log_handle.write("No tracking JSON found. Deleting old output format.\n")
        try:
            for out_dir in output_dirs:
                shutil.rmtree(out_dir, ignore_errors=True)
            os.makedirs(tracking_dir, exist_ok=True)
            tracking_data = {
                "file_hashes": current_hashes,
                "last_checked": time.time(),
                "tracked_files": {key: rel_path for rel_path, key in tracked_files}
            }
            with open(tracking_json_file, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2)
            log_handle.write("Output directories recreated with file hashes.\n")
        except Exception as ex:
            log_handle.write(f"WARNING: Failed to recreate output: {ex}\n")
            log_handle.flush()
            return False
        log_handle.flush()
        return True
    
    # Read stored tracking JSON
    try:
        with open(tracking_json_file, 'r', encoding='utf-8') as f:
            tracking_data = json.load(f)
        stored_hashes = tracking_data.get("file_hashes", {})
    except Exception as ex:
        log_handle.write(f"WARNING: Failed to read tracking JSON: {ex}. Deleting output.\n")
        log_handle.flush()
        try:
            for out_dir in output_dirs:
                shutil.rmtree(out_dir, ignore_errors=True)
            os.makedirs(tracking_dir, exist_ok=True)
            tracking_data = {
                "file_hashes": current_hashes,
                "last_checked": time.time(),
                "tracked_files": {key: rel_path for rel_path, key in tracked_files}
            }
            with open(tracking_json_file, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2)
        except Exception:
            pass
        return False
    
    # Compare hashes
    hashes_changed = False
    for hash_key, current_hash in current_hashes.items():
        stored_hash = stored_hashes.get(hash_key, "")
        if stored_hash != current_hash:
            hashes_changed = True
            log_handle.write(f"File hash changed for key '{hash_key}': {stored_hash[:16]}... -> {current_hash[:16]}...\n")
            break
    
    if hashes_changed:
        # File content has changed, delete all output directories
        try:
            for out_dir in output_dirs:
                shutil.rmtree(out_dir, ignore_errors=True)
            os.makedirs(tracking_dir, exist_ok=True)
            tracking_data = {
                "file_hashes": current_hashes,
                "last_checked": time.time(),
                "tracked_files": {key: rel_path for rel_path, key in tracked_files}
            }
            with open(tracking_json_file, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2)
            log_handle.write(
                f"Tracked file content changed. "
                f"Deleted and recreated output directories: {output_dirs}\n"
            )
        except Exception as ex:
            log_handle.write(f"WARNING: Failed to delete output directories: {ex}\n")
            log_handle.flush()
            return False
    else:
        log_handle.write("All tracked files unchanged. Preserving output directories.\n")
    
    log_handle.flush()
    return True


def check_and_clean_tracking_if_story_changed(base_dir: str, log_handle) -> bool:
    """Check if 1.story.txt has changed and delete output directory if it has."""
    tracking_dir = os.path.join(base_dir, "output", "tracking")
    tracking_json_file = os.path.join(tracking_dir, "generate.state.json")

    # Track the story file
    tracked_files = [
        ("input/1.story.txt", "story_file")
    ]

    # Output directories to clear: gen.video/output, gen.image/output, gen.audio/output
    gen_video_output = os.path.join(base_dir, "output")
    gen_image_output = os.path.abspath(os.path.join(base_dir, "..", "gen.image", "output"))
    gen_audio_output = os.path.abspath(os.path.join(base_dir, "..", "gen.audio", "output"))
    output_dirs = [gen_video_output, gen_image_output, gen_audio_output]

    return check_and_clean_tracking_if_files_changed(
        base_dir, log_handle, tracked_files, tracking_json_file, output_dirs=output_dirs
    )


def empty_comfyui_folders(base_dir: str, log_handle) -> bool:
    """Empty the ComfyUI input and output folders as a pre-step."""
    comfy_dir = resolve_comfyui_dir(base_dir)
    input_dir = os.path.join(comfy_dir, "input")
    output_dir = os.path.join(comfy_dir, "output")
    
    log_handle.write("Emptying ComfyUI input and output folders...\n")
    log_handle.flush()
    
    success = True
    
    # Empty input folder
    if os.path.exists(input_dir):
        try:
            for item in os.listdir(input_dir):
                item_path = os.path.join(input_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    log_handle.write(f"Removed file: {item_path}\n")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    log_handle.write(f"Removed directory: {item_path}\n")
        except Exception as ex:
            log_handle.write(f"WARNING: Failed to empty input folder: {ex}\n")
            success = False
    else:
        log_handle.write(f"Input folder does not exist: {input_dir}\n")
    
    # Empty output folder
    if os.path.exists(output_dir):
        try:
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    log_handle.write(f"Removed file: {item_path}\n")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    log_handle.write(f"Removed directory: {item_path}\n")
        except Exception as ex:
            log_handle.write(f"WARNING: Failed to empty output folder: {ex}\n")
            success = False
    else:
        log_handle.write(f"Output folder does not exist: {output_dir}\n")
    
    if success:
        log_handle.write("Successfully emptied ComfyUI input and output folders.\n")
    else:
        log_handle.write("Completed folder cleanup with warnings.\n")
    
    log_handle.flush()
    return success


def start_comfyui(working_dir: str, log_handle) -> subprocess.Popen:
    comfy_dir = resolve_comfyui_dir(working_dir)
    main_py = os.path.join(comfy_dir, "main.py")

    log_handle.write(f"Starting ComfyUI backend using Windows cmd style...\n")
    log_handle.flush()

    if not os.path.exists(main_py):
        log_handle.write(f"ERROR: ComfyUI main.py not found at: {main_py}\n")
        log_handle.flush()
        return None

    creation_flags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    if os.name == "nt":
        # Launch directly with cwd set to ComfyUI dir
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(
            [sys.executable, "main.py", "--disable-pinned-memory", "--async-offload", "16"],
            cwd=comfy_dir,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            creationflags=creation_flags,
            env=env,
        )
    else:
        # Non-Windows fallback
        proc = subprocess.Popen(
            [sys.executable, "main.py", "--disable-pinned-memory", "--async-offload", "16"],
            cwd=comfy_dir,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            creationflags=creation_flags,
        )

    return proc


def stop_comfyui(proc: subprocess.Popen, log_handle) -> None:
    if proc is None:
        return
    log_handle.write("Stopping ComfyUI backend...\n")
    log_handle.flush()

    try:
        # Use terminate/kill to avoid CTRL_BREAK abort messages from some runtimes on Windows
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)
    except Exception as ex:
        log_handle.write(f"WARNING: Failed to stop ComfyUI cleanly: {ex}\n")
        log_handle.flush()
    
    # Ensure the process is gone and the port has closed
    try:
        wait_for_comfyui_stopped(proc, log_handle)
    except Exception as ex:
        log_handle.write(f"WARNING: Error while waiting for ComfyUI to stop: {ex}\n")
        log_handle.flush()


def run_script(script_name: str, working_dir: str, log_handle) -> int:
    start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    start_perf = time.perf_counter()
    log_handle.write(f"\n===== START {script_name} @ {start_wall} =====\n")
    log_handle.flush()

    cmd = [sys.executable, script_name] + SCRIPT_ARGS.get(os.path.basename(script_name), [])

    # Ensure Python subprocess writes UTF-8 to stdout/stderr to avoid cp1252 errors on Windows
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    result = subprocess.run(
        cmd,
        cwd=os.path.dirname(script_name),
        stdout=log_handle,
        stderr=log_handle,
        text=True,
        env=env,
    )

    elapsed = time.perf_counter() - start_perf
    end_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    log_handle.write(
        f"\n===== END {script_name} @ {end_wall} (exit={result.returncode}, took={elapsed:.2f}s) =====\n"
    )
    log_handle.flush()
    return result.returncode


def start_lmstudio(log_handle) -> bool:
    cmd_env = os.environ.get("LM_STUDIO_CMD")
    if cmd_env:
        try:
            base_cmd = shlex.split(cmd_env, posix=False)
        except Exception:
            base_cmd = [cmd_env]
    else:
        if shutil.which("lms"):
            base_cmd = ["lms"]
        else:
            if os.name == "nt":
                userprofile = os.environ.get("USERPROFILE", "")
                candidate = os.path.join(userprofile, ".lmstudio", "bin", "lms.exe")
            else:
                candidate = os.path.expanduser(os.path.join("~", ".lmstudio", "bin", "lms"))
            base_cmd = [candidate]

    args = base_cmd + ["server", "start"]

    log_handle.write("Starting LM Studio backend via lms CLI...\n")
    log_handle.write("Command: " + " ".join(args) + "\n")
    log_handle.flush()

    creation_flags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        result = subprocess.run(
            args,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            creationflags=creation_flags,
        )
        if result.returncode == 0:
            return True
        else:
            log_handle.write(f"ERROR: lms server start exited with {result.returncode}\n")
            log_handle.flush()
            return False
    except FileNotFoundError as ex:
        log_handle.write(f"ERROR: Failed to start LM Studio. Command not found: {args[0]} ({ex})\n")
        log_handle.flush()
        return False
    except Exception as ex:
        log_handle.write(f"ERROR: Failed to start LM Studio: {ex}\n")
        log_handle.flush()
        return False


def unload_lmstudio_all_models(log_handle) -> None:
    # Build base command for lms
    cmd_env = os.environ.get("LM_STUDIO_CMD")
    if cmd_env:
        try:
            base_cmd = shlex.split(cmd_env, posix=False)
        except Exception:
            base_cmd = [cmd_env]
    else:
        if shutil.which("lms"):
            base_cmd = ["lms"]
        else:
            if os.name == "nt":
                userprofile = os.environ.get("USERPROFILE", "")
                candidate = os.path.join(userprofile, ".lmstudio", "bin", "lms.exe")
            else:
                candidate = os.path.expanduser(os.path.join("~", ".lmstudio", "bin", "lms"))
            base_cmd = [candidate]

    # Use default local server; no explicit host needed
    args = base_cmd + ["unload", "--all"]

    log_handle.write("Unloading all models from LM Studio...\n")
    log_handle.write("Command: " + " ".join(args) + "\n")
    log_handle.flush()

    creation_flags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        result = subprocess.run(
            args,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            creationflags=creation_flags,
        )
        if result.returncode != 0:
            log_handle.write(f"WARNING: 'lms unload --all' exited with code {result.returncode}.\n")
            log_handle.flush()
    except FileNotFoundError as ex:
        log_handle.write(f"WARNING: Could not run lms unload: {args[0]} not found ({ex}).\n")
        log_handle.flush()
    except Exception as ex:
        log_handle.write(f"WARNING: Failed to unload LM Studio models: {ex}\n")
        log_handle.flush()


def stop_lmstudio(log_handle) -> None:
    cmd_env = os.environ.get("LM_STUDIO_CMD")
    if cmd_env:
        try:
            base_cmd = shlex.split(cmd_env, posix=False)
        except Exception:
            base_cmd = [cmd_env]
    else:
        if shutil.which("lms"):
            base_cmd = ["lms"]
        else:
            if os.name == "nt":
                userprofile = os.environ.get("USERPROFILE", "")
                candidate = os.path.join(userprofile, ".lmstudio", "bin", "lms.exe")
            else:
                candidate = os.path.expanduser(os.path.join("~", ".lmstudio", "bin", "lms"))
            base_cmd = [candidate]

    args = base_cmd + ["server", "stop"]

    log_handle.write("Stopping LM Studio backend via lms CLI...\n")
    log_handle.write("Command: " + " ".join(args) + "\n")
    log_handle.flush()

    creation_flags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        # First try to unload all models to allow graceful shutdown
        unload_lmstudio_all_models(log_handle)

        result = subprocess.run(
            args,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            creationflags=creation_flags,
        )
        if result.returncode != 0:
            log_handle.write(f"WARNING: 'lms server stop' exited with code {result.returncode}.\n")
            log_handle.flush()
        # After requesting stop, wait until port is closed to ensure shutdown
        wait_for_lmstudio_stopped(log_handle)
    except Exception as ex:
        log_handle.write(f"WARNING: Failed to stop LM Studio cleanly: {ex}\n")
        log_handle.flush()


def _tcp_connect(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _http_probe(url: str, timeout: float = 3.0) -> bool:
    try:
        # Any HTTP response means the server socket is accepting connections
        with urllib.request.urlopen(url, timeout=timeout) as _:
            return True
    except urllib.error.HTTPError:
        # Server responded with an HTTP error => still indicates readiness
        return True
    except Exception:
        return False


def wait_for_comfyui_ready(proc: subprocess.Popen, log_handle, interval_seconds: int = 15) -> bool:
    base_url = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188")
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (8188 if (parsed.scheme or "http") == "http" else 8188)

    start_ts = time.perf_counter()
    attempt = 0
    while True:
        attempt += 1
        if proc is not None and proc.poll() is not None:
            log_handle.write("ERROR: ComfyUI process exited before readiness.\n")
            log_handle.flush()
            return False

        socket_ok = _tcp_connect(host, port, timeout=3.0)
        http_ok = _http_probe(urljoin(base_url if base_url.endswith('/') else base_url + '/', ''), timeout=3.0) if socket_ok else False

        if socket_ok and http_ok:
            elapsed = time.perf_counter() - start_ts
            log_handle.write(f"ComfyUI is ready after {elapsed:.1f}s (attempt {attempt}).\n")
            log_handle.flush()
            return True

        log_handle.write(f"ComfyUI not ready yet (attempt {attempt}). Retrying in {interval_seconds}s...\n")
        log_handle.flush()
        time.sleep(interval_seconds)


def wait_for_comfyui_stopped(proc: subprocess.Popen, log_handle, interval_seconds: int = 3) -> None:
    base_url = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188")
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (8188 if (parsed.scheme or "http") == "http" else 8188)

    # First, ensure the child process is gone
    if proc is not None:
        try:
            for _ in range(10):
                if proc.poll() is not None:
                    break
                time.sleep(interval_seconds)
        except Exception:
            pass

    # Then, wait until socket is closed to avoid port reuse hazards
    attempts = 0
    while True:
        attempts += 1
        if not _tcp_connect(host, port, timeout=1.0):
            log_handle.write(f"ComfyUI appears stopped (after {attempts} checks).\n")
            log_handle.flush()
            return
        log_handle.write("ComfyUI still responding; waiting for shutdown...\n")
        log_handle.flush()
        time.sleep(interval_seconds)


def wait_for_lmstudio_ready(log_handle, interval_seconds: int = 15) -> bool:
    base_url = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (1234 if (parsed.scheme or "http") == "http" else 1234)
    models_url = urljoin(base_url if base_url.endswith('/') else base_url + '/', 'models')

    start_ts = time.perf_counter()
    attempt = 0
    while True:
        attempt += 1
        socket_ok = _tcp_connect(host, port, timeout=3.0)
        http_ok = _http_probe(models_url, timeout=3.0) if socket_ok else False

        if socket_ok and http_ok:
            elapsed = time.perf_counter() - start_ts
            log_handle.write(f"LM Studio is ready after {elapsed:.1f}s (attempt {attempt}).\n")
            log_handle.flush()
            return True

        log_handle.write(f"LM Studio not ready yet (attempt {attempt}). Retrying in {interval_seconds}s...\n")
        log_handle.flush()
        time.sleep(interval_seconds)


def wait_for_lmstudio_stopped(log_handle, interval_seconds: int = 3) -> None:
    base_url = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (1234 if (parsed.scheme or "http") == "http" else 1234)
    attempts = 0
    # Poll until TCP connection fails, indicating server is down
    while True:
        attempts += 1
        if not _tcp_connect(host, port, timeout=1.0):
            log_handle.write(f"LM Studio appears stopped (after {attempts} checks).\n")
            log_handle.flush()
            return
        log_handle.write("LM Studio still responding; waiting for shutdown...\n")
        log_handle.flush()
        time.sleep(interval_seconds)


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "log.txt")
    workflow_start_perf = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as raw_log:
        log = TimedLogWriter(raw_log)
        log.write("Workflow runner started. Python executable: " + sys.executable + "\n")
        log.write("Working directory: " + base_dir + "\n")
        log.flush()

        # Empty ComfyUI folders as pre-step
        empty_comfyui_folders(base_dir, log)

        # Check if story file changed and clean tracking directory if needed
        # check_and_clean_tracking_if_story_changed(base_dir, log)

        # Manage services across scripts: keep running across consecutive needs
        comfy_proc = None
        lmstudio_active = False

        # Ensure services are shut down even on exceptions or Ctrl+C
        def _cleanup_services():
            nonlocal comfy_proc, lmstudio_active
            try:
                if comfy_proc is not None:
                    stop_comfyui(comfy_proc, log)
                    comfy_proc = None
            finally:
                if lmstudio_active:
                    stop_lmstudio(log)
                    lmstudio_active = False

        atexit.register(_cleanup_services)

        # Signal handlers to cleanup on Ctrl+C, Ctrl+Break, or termination
        cleanup_triggered = {"value": False}

        def _handle_signal(signum, _frame):
            if cleanup_triggered["value"]:
                return
            cleanup_triggered["value"] = True
            try:
                log.write(f"Received signal {signum}. Cleaning up services...\n")
                log.flush()
            except Exception:
                pass
            try:
                _cleanup_services()
            finally:
                try:
                    atexit.unregister(_cleanup_services)
                except Exception:
                    pass
            raise SystemExit(128 + (signum if isinstance(signum, int) else 0))

        try:
            signal.signal(signal.SIGINT, _handle_signal)
        except Exception:
            pass
        try:
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, _handle_signal)
        except Exception:
            pass
        try:
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, _handle_signal)
        except Exception:
            pass

        for idx, script in enumerate(SCRIPTS):
            if script.startswith("../"):
                # Audio scripts have full relative paths - resolve them properly
                script_path = os.path.abspath(os.path.join(base_dir, script))
            else:
                # Image scripts are in the scripts subdirectory
                script_path = os.path.join(base_dir, SCRIPTS_DIR, script)
            if not os.path.exists(script_path):
                log.write(f"ERROR: Script not found: {script_path}\n")
                log.flush()
                # Stop any running services before exiting
                if comfy_proc is not None:
                    stop_comfyui(comfy_proc, log)
                if lmstudio_active:
                    stop_lmstudio(log)
                return 1

            script_name = os.path.basename(script)
            needs_comfy = script_name in NEEDS_COMFYUI
            needs_lms = script_name in NEEDS_LMSTUDIO

            # Start services if required and not already running
            if needs_comfy and comfy_proc is None:
                comfy_proc = start_comfyui(base_dir, log)
                if comfy_proc is None:
                    log.write("ABORTING: Could not start ComfyUI backend.\n")
                    log.flush()
                    if lmstudio_active:
                        stop_lmstudio(log)
                    return 1
                log.write("Waiting for ComfyUI to become ready (polling every 15s)...\n")
                log.flush()
                if not wait_for_comfyui_ready(comfy_proc, log):
                    if lmstudio_active:
                        stop_lmstudio(log)
                    if comfy_proc is not None:
                        stop_comfyui(comfy_proc, log)
                    return 1

            if needs_lms and not lmstudio_active:
                lms_ok = start_lmstudio(log)
                if not lms_ok:
                    if comfy_proc is not None:
                        stop_comfyui(comfy_proc, log)
                    log.write("ABORTING: Could not start LM Studio backend.\n")
                    log.flush()
                    return 1
                lmstudio_active = True
                log.write("Waiting for LM Studio to become ready (polling every 15s)...\n")
                log.flush()
                if not wait_for_lmstudio_ready(log):
                    if comfy_proc is not None:
                        stop_comfyui(comfy_proc, log)
                    stop_lmstudio(log)
                    lmstudio_active = False
                    return 1

            code = run_script(script_path, base_dir, log)
 

            # Determine if the next script still needs services
            next_needs_comfy = False
            next_needs_lms = False
            if idx + 1 < len(SCRIPTS):
                next_script = SCRIPTS[idx + 1]
                next_script_name = os.path.basename(next_script)
                next_needs_comfy = next_script_name in NEEDS_COMFYUI
                next_needs_lms = next_script_name in NEEDS_LMSTUDIO

            # Stop services only if not needed by the next script
            if needs_comfy and not next_needs_comfy and comfy_proc is not None:
                stop_comfyui(comfy_proc, log)
                comfy_proc = None
            if needs_lms and not next_needs_lms and lmstudio_active:
                stop_lmstudio(log)
                lmstudio_active = False

            if code != 0:
                # On error, ensure services are stopped
                if comfy_proc is not None:
                    stop_comfyui(comfy_proc, log)
                    comfy_proc = None
                if lmstudio_active:
                    stop_lmstudio(log)
                    lmstudio_active = False
                log.write(f"ABORTING: {script} exited with code {code}.\n")
                log.flush()
                return code

        # After all scripts, ensure services are stopped
        if comfy_proc is not None:
            stop_comfyui(comfy_proc, log)
        if lmstudio_active:
            stop_lmstudio(log)

        # Summarize logging overhead before final message
        total_log_time = LOG_STATS["write_seconds"] + LOG_STATS["flush_seconds"]
        total_runtime = max(0.0, time.perf_counter() - workflow_start_perf)
        pct = (total_log_time / total_runtime * 100.0) if total_runtime > 0 else 0.0

        log.write(
            "\nLog time stats: "
            f"writes={LOG_STATS['write_seconds']:.3f}s, "
            f"flushes={LOG_STATS['flush_seconds']:.3f}s, "
            f"total={total_log_time:.3f}s (~{pct:.2f}% of runtime)\n"
        )
        log.flush()

        log.write("\nAll scripts completed successfully.\n")
        log.flush()

    print("All scripts completed. See log.txt for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())