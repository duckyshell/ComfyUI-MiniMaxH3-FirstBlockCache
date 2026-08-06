from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path


BASE_URL = "http://127.0.0.1:8195"
SEED = 20260807
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
COMFY_OUTPUT = Path(r"C:\Users\ama\Desktop\minimax-h3\ComfyUI_windows_portable\ComfyUI\output")

PROMPT = """integrated_multimodal_description: [Shot 1] Realistic live-action single continuous shot at dawn in a quiet misty pine forest. A red fox walks carefully through a shallow clear stream toward the camera, placing each paw naturally between smooth wet stones. Water ripples around its legs and droplets catch the soft golden light. The fox pauses briefly, looks toward a distant sound, then continues forward while its ears and tail move naturally. The camera performs a slow stable tracking move backward at the fox's eye level, maintaining a medium shot with gentle natural depth of field. Fine fur, water reflections, mist, and forest textures remain realistic and physically coherent throughout.

overall_soundscape: Clear flowing water, soft paw splashes, light morning wind through pine needles, and distant forest birds.

non_diegetic_music: A restrained cinematic ambient score with soft low strings and sparse warm tones, mixed quietly beneath the natural forest sounds."""

PRESETS = {
    "safe": "H3 Safe — 0.08 / max 2",
    "fast": "H3 Fast — 0.10 / max 2",
    "aggressive": "H3 Aggressive — 0.12 / max 2",
    "custom": "Custom — manual values",
}


def request_json(path: str, payload: dict | None = None, timeout: float = 30.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else None


def get_logs() -> list[dict]:
    return request_json("/internal/logs/raw")["entries"]


def unload_models():
    request_json("/free", {"unload_models": True, "free_memory": True})
    deadline = time.time() + 45
    while time.time() < deadline:
        queue = request_json("/queue")
        stats = request_json("/system_stats")
        free_vram = stats["devices"][0]["vram_free"]
        if not queue["queue_running"] and not queue["queue_pending"] and free_vram > 30 * 1024**3:
            time.sleep(1.0)
            return
        time.sleep(1.0)
    raise RuntimeError("Timed out while unloading models for a cold run")


def build_prompt(config: str, prefix: str, nonce: int, width: int, height: int, length: int) -> dict:
    model_link = ["1", 0]
    graph = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
        },
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "6": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "prompt": PROMPT,
                "width": width,
                "height": height,
                "length": ["16", 1],
            },
        },
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {
            "class_type": "BasicScheduler",
            "inputs": {"model": model_link, "scheduler": "simple", "steps": 20, "denoise": 1.0},
        },
        "10": {"class_type": "BasicGuider", "inputs": {"model": model_link, "conditioning": ["6", 0]}},
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["7", 0],
                "guider": ["10", 0],
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": ["6", 1],
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["5", 0]}},
        "14": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["12", 0], "audio": ["13", 0], "fps": 24.0, "bit_depth": 8},
        },
        "15": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["14", 0], "filename_prefix": prefix, "format": "mp4", "codec": "auto"},
        },
        "16": {
            "class_type": "ComfyMathExpression",
            "inputs": {"expression": f"{length} + 0 * a", "values.a": ["17", 0]},
        },
        "17": {"class_type": "PrimitiveFloat", "inputs": {"value": float(nonce)}},
    }
    if config != "baseline":
        graph["2"] = {
            "class_type": "ApplyMiniMaxH3FirstBlockCache",
            "inputs": {
                "model": ["1", 0],
                "mode": PRESETS[config],
                "threshold": 0.10,
                "start_percent": 0.10,
                "end_percent": 0.95,
                "max_consecutive_hits": 2,
                "temporal_guard": True,
            },
        }
        model_link = ["2", 0]
        graph["9"]["inputs"]["model"] = model_link
        graph["10"]["inputs"]["model"] = model_link
    return graph


def wait_for_prompt(prompt_id: str, timeout: float = 3600.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = request_json(f"/history/{prompt_id}")
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            if status.get("completed"):
                return item
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False, indent=2))
        time.sleep(1.0)
    raise TimeoutError(f"Prompt {prompt_id} did not finish within {timeout:.0f}s")


def output_files(history: dict) -> list[Path]:
    paths: list[Path] = []
    for node_output in history.get("outputs", {}).values():
        for items in node_output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                if item.get("type", "output") != "output":
                    continue
                paths.append(COMFY_OUTPUT / item.get("subfolder", "") / item["filename"])
    return paths


def backend_name() -> str:
    argv = request_json("/system_stats")["system"].get("argv", [])
    return "sage2" if "--use-sage-attention" in argv else "native"


def run_once(config: str, temperature: str, nonce: int, width: int, height: int, length: int) -> dict:
    if temperature == "cold":
        unload_models()

    backend = backend_name()
    run_name = f"{backend}_{nonce:02d}_{config}_{temperature}"
    prefix = f"video/FBCache_benchmark/{run_name}"
    graph = build_prompt(config, prefix, nonce, width, height, length)
    graph_path = RESULTS_DIR / f"{run_name}_api.json"
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    logs_before = len(get_logs())
    started = time.perf_counter()
    response = request_json("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
    prompt_id = response["prompt_id"]
    history = wait_for_prompt(prompt_id)
    wall_seconds = time.perf_counter() - started
    new_logs = get_logs()[logs_before:]
    log_text = "".join(entry.get("m", "") for entry in new_logs)
    (RESULTS_DIR / f"{run_name}.log.txt").write_text(log_text, encoding="utf-8")

    copied: list[str] = []
    for source in output_files(history):
        if source.is_file():
            target = RESULTS_DIR / f"{run_name}{source.suffix.lower()}"
            shutil.copy2(source, target)
            copied.append(str(target))

    cache_summary = None
    for line in log_text.splitlines():
        if "MiniMax H3 FBCache: cached" in line:
            cache_summary = line.strip()

    result = {
        "run": run_name,
        "config": config,
        "backend": backend,
        "temperature": temperature,
        "prompt_id": prompt_id,
        "wall_seconds": round(wall_seconds, 3),
        "cache_summary": cache_summary,
        "files": copied,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def save_results(results: list[dict], filename: str):
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sanity", "custom_sanity", "matrix", "control"])
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "sanity":
        results = [run_once("baseline", "cold", 0, 608, 352, 39)]
        save_results(results, f"sanity_results_{backend_name()}.json")
        return

    if args.mode == "custom_sanity":
        results = [run_once("custom", "warm", 0, 608, 352, 39)]
        save_results(results, f"custom_sanity_results_{backend_name()}.json")
        return

    if args.mode == "matrix":
        schedule = [
            ("baseline", "cold"),
            ("baseline", "warm"),
            ("safe", "cold"),
            ("safe", "warm"),
            ("fast", "cold"),
            ("fast", "warm"),
            ("aggressive", "cold"),
            ("aggressive", "warm"),
            ("baseline", "warm_confirm"),
        ]
    else:
        schedule = [
            ("baseline", "cold"),
            ("baseline", "warm"),
            ("fast", "cold"),
            ("fast", "warm"),
        ]
    results: list[dict] = []
    for nonce, (config, temperature) in enumerate(schedule, start=1):
        effective_temperature = "warm" if temperature == "warm_confirm" else temperature
        result = run_once(config, effective_temperature, nonce, 960, 544, 124)
        result["temperature"] = temperature
        results.append(result)
        save_results(results, f"{args.mode}_results_{backend_name()}.partial.json")
    save_results(results, f"{args.mode}_results_{backend_name()}.json")


if __name__ == "__main__":
    main()
