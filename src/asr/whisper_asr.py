"""
ASR Module — Stage 1: Swahili Speech → Swahili Text
Model: faster-whisper (Whisper large-v3)
GPU: float16 | CPU: int8
"""
import time
import json
import os
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Optional
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ASRSegment:
    start: float
    end:   float
    text:  str


@dataclass
class ASRResult:
    audio_file:        str
    language:          str
    lang_confidence:   float
    elapsed_sec:       float
    audio_duration:    float
    real_time_factor:  float
    num_segments:      int
    full_text:         str
    segments:          List[ASRSegment]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["segments"] = [asdict(s) for s in self.segments]
        return d

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        log.info(f"ASR results saved: {path}")


class WhisperASR:
    """
    Whisper large-v3 ASR with:
    - GPU float16 / CPU int8 auto-selection
    - VAD-based chunking (eliminates hallucination loops)
    - condition_on_previous_text=False (production stability)
    - Real-time factor (RTF) measurement
    - WER evaluation via jiwer
    """

    def __init__(self, config: dict):
        self.config = config
        self.model  = None
        self._loaded = False

    def load(self) -> "WhisperASR":
        from faster_whisper import WhisperModel

        model_size   = self.config["asr"]["model"]
        device       = "cuda" if self.config.get("_gpu_available") else "cpu"
        compute_type = self.config["asr"]["_compute_type"]

        log.info(f"Loading Whisper {model_size} on {device}/{compute_type}...")
        t = time.perf_counter()

        self.model = WhisperModel(
            model_size,
            device       = device,
            compute_type = compute_type,
        )
        self._loaded = True
        log.info(f"Whisper loaded in {time.perf_counter()-t:.2f}s")
        return self

    def transcribe(self, audio_path: str,
                   save_path: Optional[str] = None) -> ASRResult:
        if not self._loaded:
            self.load()

        cfg = self.config["asr"]
        log.info(f"Transcribing: {audio_path}")
        start = time.perf_counter()

        segments_gen, info = self.model.transcribe(
            audio_path,
            language                   = cfg["language"],
            beam_size                  = cfg["beam_size"],
            condition_on_previous_text = cfg["condition_on_previous_text"],
            vad_filter                 = cfg["vad_filter"],
            vad_parameters             = dict(
                min_silence_duration_ms = cfg["vad_min_silence_ms"]
            ),
        )

        raw_segments = list(segments_gen)
        elapsed      = time.perf_counter() - start

        # Deduplicate adjacent identical segments
        segments  = []
        prev_text = ""
        for s in raw_segments:
            text = s.text.strip()
            if text and text != prev_text:
                segments.append(ASRSegment(
                    start = round(s.start, 2),
                    end   = round(s.end,   2),
                    text  = text
                ))
                prev_text = text

        full_text      = " ".join(s.text for s in segments)
        audio_duration = segments[-1].end if segments else 0.0
        rtf            = elapsed / audio_duration if audio_duration > 0 else 0.0

        result = ASRResult(
            audio_file       = audio_path,
            language         = info.language,
            lang_confidence  = round(info.language_probability, 4),
            elapsed_sec      = round(elapsed, 2),
            audio_duration   = round(audio_duration, 2),
            real_time_factor = round(rtf, 4),
            num_segments     = len(segments),
            full_text        = full_text,
            segments         = segments,
        )

        log.info(
            f"ASR done | {elapsed:.2f}s | RTF: {rtf:.4f} | "
            f"Lang: {info.language} ({info.language_probability:.2f}) | "
            f"{len(segments)} segments"
        )

        if save_path:
            result.save(save_path)

        return result

    def compute_wer(self, hypothesis: str, reference: str) -> float:
        from jiwer import wer
        score = wer(reference, hypothesis)
        log.info(f"WER: {score:.4f} ({score*100:.2f}%)")
        return round(score, 4)

    def batch_wer(self, hypotheses: List[str],
                  references: List[str]) -> dict:
        from jiwer import wer, cer
        w = wer(references, hypotheses)
        c = cer(references, hypotheses)
        log.info(f"Batch WER: {w:.4f} | CER: {c:.4f}")
        return {"wer": round(w, 4), "cer": round(c, 4)}
