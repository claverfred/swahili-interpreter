"""
TTS Module — Stage 3: English Text → English Speech
Primary  : Coqui XTTS-v2 (GPU, high quality, offline)
Fallback : Edge-TTS (Microsoft cloud, zero-setup)
"""
import asyncio
import os
import re
import time
import json
import tempfile
import torch
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import List, Optional
from src.utils.logger import get_logger

log = get_logger(__name__)

# XTTS-v2 hard-caps a single call at 400 GPT text tokens; its own internal
# sentence splitter doesn't guarantee every chunk stays under that (e.g. a
# long run-on segment with no punctuation), so we pre-chunk defensively.
_XTTS_MAX_CHARS = 200


def _chunk_text(text: str, max_chars: int = _XTTS_MAX_CHARS) -> List[str]:
    """Split into pieces safely under XTTS's per-call token cap: first on
    sentence boundaries, then hard-wrapping any punctuation-free run-on."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current.strip())
            current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            flush()
            piece = ""
            for word in sent.split():
                if piece and len(piece) + 1 + len(word) > max_chars:
                    chunks.append(piece)
                    piece = word
                else:
                    piece = f"{piece} {word}".strip()
            current = piece
            continue
        if current and len(current) + 1 + len(sent) > max_chars:
            flush()
        current = f"{current} {sent}".strip()
    flush()
    return chunks


def _run_async(coro):
    """Run a coroutine to completion, whether or not an event loop
    (e.g. Colab/Jupyter's kernel loop) is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


@dataclass
class TTSResult:
    engine:         str
    output_path:    str
    elapsed_sec:    float
    audio_duration: float
    real_time_factor: float
    char_count:     int

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class TTSEngine:
    """
    TTS engine with automatic Coqui XTTS-v2 → Edge-TTS fallback.
    Measures RTF (real-time factor) for thesis evaluation.
    """

    def __init__(self, config: dict):
        self.config  = config
        self.model   = None
        self._engine = config["tts"]["engine"]
        self._loaded = False

    def load(self) -> "TTSEngine":
        if self._engine == "coqui":
            self._load_coqui()
        else:
            log.info("Edge-TTS selected — no local model to load")
            self._loaded = True
        return self

    def _load_coqui(self):
        try:
            from TTS.api import TTS as CoquiTTS
            cfg = self.config["tts"]
            log.info(f"Loading Coqui {cfg['coqui_model']}...")
            t = time.perf_counter()
            self.model = CoquiTTS(
                cfg["coqui_model"],
                progress_bar = False,
                gpu          = torch.cuda.is_available()
            )
            self._loaded = True
            log.info(f"Coqui XTTS-v2 loaded in {time.perf_counter()-t:.2f}s")
        except Exception as e:
            log.warning(f"Coqui load failed: {e} — falling back to Edge-TTS")
            self._engine = "edge"
            self._loaded = True

    def synthesize(self, text: str, output_path: str,
                   save_meta: bool = False) -> TTSResult:
        if not self._loaded:
            self.load()

        if not text.strip():
            raise ValueError("Empty text passed to TTS synthesize()")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        log.info(f"Synthesizing {len(text)} chars via {self._engine}...")
        start = time.perf_counter()

        if self._engine == "coqui" and self.model is not None:
            self._synthesize_coqui(text, output_path)
        else:
            self._synthesize_edge(text, output_path)

        elapsed = time.perf_counter() - start

        # Measure output audio duration
        try:
            data, sr       = sf.read(output_path)
            audio_duration = len(data) / sr
        except Exception:
            audio_duration = 0.0

        rtf = elapsed / audio_duration if audio_duration > 0 else 0.0

        result = TTSResult(
            engine           = self._engine,
            output_path      = output_path,
            elapsed_sec      = round(elapsed, 2),
            audio_duration   = round(audio_duration, 2),
            real_time_factor = round(rtf, 4),
            char_count       = len(text),
        )

        log.info(
            f"TTS done | {elapsed:.2f}s | "
            f"Audio: {audio_duration:.2f}s | RTF: {rtf:.4f}"
        )

        if save_meta:
            meta_path = output_path.replace(".wav", "_meta.json")
            result.save(meta_path)

        return result

    def _synthesize_coqui(self, text: str, output_path: str):
        cfg    = self.config["tts"]
        chunks = _chunk_text(text)

        if len(chunks) == 1:
            self.model.tts_to_file(
                text      = chunks[0],
                file_path = output_path,
                language  = cfg["coqui_language"],
                speaker   = cfg["coqui_speaker"],
            )
            return

        log.info(f"Text exceeds XTTS's per-call limit — synthesizing in {len(chunks)} chunks")
        audio_parts, sr = [], None
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i, chunk in enumerate(chunks):
                part_path = os.path.join(tmp_dir, f"part_{i}.wav")
                self.model.tts_to_file(
                    text      = chunk,
                    file_path = part_path,
                    language  = cfg["coqui_language"],
                    speaker   = cfg["coqui_speaker"],
                )
                data, sr = sf.read(part_path)
                audio_parts.append(data)

        sf.write(output_path, np.concatenate(audio_parts), sr)

    def _synthesize_edge(self, text: str, output_path: str):
        import edge_tts
        cfg = self.config["tts"]

        async def _run():
            communicate = edge_tts.Communicate(
                text[:5000],
                cfg["edge_voice"],
                rate   = cfg.get("edge_rate",   "+0%"),
                volume = cfg.get("edge_volume", "+0%"),
            )
            await communicate.save(output_path)

        _run_async(_run())
