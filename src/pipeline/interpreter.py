"""
SwahiliInterpreter — Full Cascade Pipeline Orchestrator
ASR (Whisper) → MT (NLLB-200) → TTS (XTTS-v2)
"""
import os
import json
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional

from src.asr   import WhisperASR
from src.mt    import NLLBTranslator
from src.tts   import TTSEngine
from src.utils import get_logger

log = get_logger(__name__)


@dataclass
class PipelineResult:
    pipeline:        str
    input_audio:     str
    output_audio:    str
    timing: dict     = field(default_factory=dict)
    asr_meta: dict   = field(default_factory=dict)
    segments: list   = field(default_factory=list)
    full_swahili:    str = ""
    full_english:    str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        log.info(f"Pipeline results saved: {path}")

    def print_summary(self):
        t = self.timing
        total = t.get("total_sec", 0)
        log.info("=" * 60)
        log.info("PIPELINE SUMMARY")
        log.info("=" * 60)
        log.info(f"  Stage 1  ASR  : {t['asr_sec']:>8.2f}s  "
                 f"({t['asr_sec']/total*100:.1f}%)")
        log.info(f"  Stage 2  MT   : {t['mt_sec']:>8.2f}s  "
                 f"({t['mt_sec']/total*100:.1f}%)")
        log.info(f"  Stage 3  TTS  : {t['tts_sec']:>8.2f}s  "
                 f"({t['tts_sec']/total*100:.1f}%)")
        log.info(f"  {'─'*35}")
        log.info(f"  TOTAL         : {total:>8.2f}s")
        log.info(f"  Input audio   : {self.asr_meta.get('audio_duration', 0):>8.2f}s")
        log.info(f"  ASR RTF       : {self.asr_meta.get('rtf', 0):>8.4f}")
        log.info("=" * 60)


class SwahiliInterpreter:
    """
    Full cascade pipeline.
    Loads each component once; can run interpret() multiple times.

    Usage:
        from src.utils import load_config
        from src.pipeline import SwahiliInterpreter

        config      = load_config()
        interpreter = SwahiliInterpreter(config).load_all()
        result      = interpreter.interpret("audio.mp3")
    """

    def __init__(self, config: dict):
        self.config   = config
        self.asr      = WhisperASR(config)
        self.mt       = NLLBTranslator(config)
        self.tts      = TTSEngine(config)
        self._loaded  = False

    def load_all(self) -> "SwahiliInterpreter":
        log.info("=" * 60)
        log.info("LOADING ALL PIPELINE COMPONENTS")
        log.info("=" * 60)
        t = time.perf_counter()
        self.asr.load()
        self.mt.load()
        self.tts.load()
        self._loaded = True
        log.info(f"All components loaded in {time.perf_counter()-t:.2f}s")
        return self

    def interpret(self, audio_path: str,
                  output_dir: Optional[str] = None) -> PipelineResult:
        """
        Run the full pipeline on a Swahili audio file.

        Args:
            audio_path : Path to input Swahili audio (mp3/wav/m4a/flac)
            output_dir : Directory for output files (default: results/pipeline)

        Returns:
            PipelineResult with timing, transcripts, and output audio path
        """
        if not self._loaded:
            self.load_all()

        if output_dir is None:
            output_dir = self.config["paths"]["results_pipeline"]

        os.makedirs(output_dir, exist_ok=True)
        base         = os.path.splitext(os.path.basename(audio_path))[0]
        output_audio = os.path.join(output_dir, f"{base}_english.wav")
        output_json  = os.path.join(output_dir, f"{base}_results.json")
        asr_json     = os.path.join(self.config["paths"]["results_asr"],
                                    f"{base}_asr.json")
        mt_json      = os.path.join(self.config["paths"]["results_mt"],
                                    f"{base}_mt.json")

        log.info("=" * 60)
        log.info("SWAHILI → ENGLISH INTERPRETATION")
        log.info(f"Input  : {audio_path}")
        log.info("=" * 60)

        pipeline_start = time.perf_counter()

        # ── Stage 1: ASR ──────────────────────────────────────────────────────
        log.info("[1/3] ASR — Swahili Speech → Swahili Text")
        asr_result = self.asr.transcribe(audio_path, save_path=asr_json)
        sw_texts   = [s.text for s in asr_result.segments]

        # ── Stage 2: MT ───────────────────────────────────────────────────────
        log.info("[2/3] MT — Swahili Text → English Text")
        mt_result  = self.mt.translate(sw_texts, save_path=mt_json)
        en_texts   = mt_result.translations

        # ── Stage 3: TTS ──────────────────────────────────────────────────────
        log.info("[3/3] TTS — English Text → English Speech")
        tts_result = self.tts.synthesize(mt_result.full_translation, output_audio)

        total_elapsed = time.perf_counter() - pipeline_start

        # ── Build aligned segments ─────────────────────────────────────────────
        aligned = []
        for i, seg in enumerate(asr_result.segments):
            aligned.append({
                "start"   : seg.start,
                "end"     : seg.end,
                "swahili" : seg.text,
                "english" : en_texts[i] if i < len(en_texts) else "",
            })

        result = PipelineResult(
            pipeline     = "Whisper-large-v3 → NLLB-200-distilled-600M → XTTS-v2",
            input_audio  = audio_path,
            output_audio = output_audio,
            timing       = {
                "asr_sec"  : asr_result.elapsed_sec,
                "mt_sec"   : mt_result.elapsed_sec,
                "tts_sec"  : tts_result.elapsed_sec,
                "total_sec": round(total_elapsed, 2),
            },
            asr_meta     = {
                "language"        : asr_result.language,
                "lang_confidence" : asr_result.lang_confidence,
                "num_segments"    : asr_result.num_segments,
                "audio_duration"  : asr_result.audio_duration,
                "rtf"             : asr_result.real_time_factor,
            },
            segments     = aligned,
            full_swahili = asr_result.full_text,
            full_english = mt_result.full_translation,
        )

        result.print_summary()
        result.save(output_json)

        return result
