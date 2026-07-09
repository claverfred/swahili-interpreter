"""
Configuration loader — reads configs/config.yaml and exposes a typed dict.
"""
import os
import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Load YAML config. Searches for configs/config.yaml relative to project root.
    Auto-detects GPU and sets compute types accordingly.
    """
    if config_path is None:
        # Walk up from this file to find project root (contains configs/)
        here = Path(__file__).resolve()
        for parent in [here, *here.parents]:
            candidate = parent / "configs" / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.yaml not found. Expected at configs/config.yaml "
            f"relative to project root."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── Auto-detect GPU ───────────────────────────────────────────────────────
    try:
        import torch
        config["_gpu_available"] = torch.cuda.is_available()
        if config["_gpu_available"]:
            gpu = torch.cuda.get_device_properties(0)
            config["_gpu_name"] = gpu.name
            config["_gpu_vram_gb"] = round(gpu.total_memory / 1024**3, 1)
            config["asr"]["_compute_type"] = config["asr"]["compute_type_gpu"]
            config["mt"]["_device"] = "cuda"
        else:
            config["_gpu_name"] = "CPU"
            config["_gpu_vram_gb"] = 0
            config["asr"]["_compute_type"] = config["asr"]["compute_type_cpu"]
            config["mt"]["_device"] = "cpu"
    except ImportError:
        config["_gpu_available"] = False
        config["asr"]["_compute_type"] = config["asr"]["compute_type_cpu"]
        config["mt"]["_device"] = "cpu"

    # ── Create output directories ─────────────────────────────────────────────
    for key, path in config.get("paths", {}).items():
        os.makedirs(path, exist_ok=True)

    return config


def print_config_summary(config: dict):
    """Print a clean summary of the loaded configuration."""
    gpu  = config.get("_gpu_name", "Unknown")
    vram = config.get("_gpu_vram_gb", 0)
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"  Project  : {config['project']['name']}")
    print(f"  Version  : {config['project']['version']}")
    print(f"  GPU      : {gpu}  ({vram} GB VRAM)")
    print(f"  ASR      : Whisper {config['asr']['model']}  "
          f"({config['asr']['_compute_type']})")
    print(f"  MT       : {config['mt']['primary_model']}")
    print(f"  TTS      : {config['tts']['engine']} / "
          f"{config['tts']['coqui_model'] if config['tts']['engine'] == 'coqui' else config['tts']['edge_voice']}")
    print("=" * 60)
