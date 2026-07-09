"""
Audio utility functions: loading, saving, format conversion, chunking.
"""
import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    Supports: wav, mp3, m4a, flac, ogg.
    Returns mono float32 numpy array.
    """
    log.debug(f"Loading audio: {path}")
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    log.debug(f"Loaded {len(audio)/sr:.2f}s audio at {sr}Hz")
    return audio


def save_audio(audio: np.ndarray, path: str, sample_rate: int = 22050):
    """Save numpy array as wav file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sf.write(path, audio, sample_rate)
    log.debug(f"Saved audio: {path}")


def get_audio_duration(path: str) -> float:
    """Return duration of audio file in seconds."""
    info = sf.info(path)
    return info.duration


def chunk_audio(audio: np.ndarray, sample_rate: int,
                chunk_sec: float = 30.0) -> list:
    """
    Split audio into fixed-length chunks.
    Used for chunked ASR inference on long files.
    Returns list of (start_sec, chunk_array) tuples.
    """
    chunk_len = int(chunk_sec * sample_rate)
    chunks    = []
    for i in range(0, len(audio), chunk_len):
        start_sec = i / sample_rate
        chunk     = audio[i:i + chunk_len]
        chunks.append((start_sec, chunk))
    log.debug(f"Split audio into {len(chunks)} chunks of {chunk_sec}s")
    return chunks


def is_supported_format(path: str) -> bool:
    """Check if file extension is a supported audio format."""
    supported = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}
    return Path(path).suffix.lower() in supported
