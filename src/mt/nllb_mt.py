"""
MT Module — Stage 2: Swahili Text → English Text
Primary model : NLLB-200-distilled-600M (GPU float16)
Baseline model: Helsinki-NLP/opus-mt-swc-en
"""
import time
import json
import os
import torch
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MTResult:
    model:            str
    src_lang:         str
    tgt_lang:         str
    elapsed_sec:      float
    segments_per_sec: float
    num_segments:     int
    full_translation: str
    translations:     List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        log.info(f"MT results saved: {path}")


class NLLBTranslator:
    """
    NLLB-200-distilled-600M translator with:
    - GPU float16 / CPU float32 auto-selection
    - Batched inference for throughput
    - BLEU + chrF evaluation
    """

    def __init__(self, config: dict):
        self.config    = config
        self.model     = None
        self.tokenizer = None
        self._loaded   = False
        self._model_name = config["mt"]["primary_model"]

    def load(self) -> "NLLBTranslator":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        cfg    = self.config["mt"]
        device = cfg["_device"]
        dtype  = torch.float16 if device == "cuda" else torch.float32

        log.info(f"Loading {self._model_name} on {device}...")
        t = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            src_lang = cfg["src_lang"]
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self._model_name,
            torch_dtype = dtype
        ).to(device)
        self.model.eval()

        self._loaded = True
        self._device = device
        log.info(f"{self._model_name} loaded in {time.perf_counter()-t:.2f}s")
        return self

    def translate(self, texts: List[str],
                  save_path: Optional[str] = None) -> MTResult:
        if not self._loaded:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        cfg        = self.config["mt"]
        tgt_id     = self.tokenizer.convert_tokens_to_ids(cfg["tgt_lang"])
        batch_size = cfg["batch_size_gpu"] if self._device == "cuda" \
                     else cfg["batch_size_cpu"]

        log.info(f"Translating {len(texts)} segments (batch={batch_size})...")
        translations = []
        start        = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch  = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors = "pt",
                padding        = True,
                truncation     = True,
                max_length     = cfg["max_length"],
            ).to(self._device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    forced_bos_token_id = tgt_id,
                    max_length          = cfg["max_length"],
                    num_beams           = cfg["num_beams"],
                    early_stopping      = True,
                )

            decoded = self.tokenizer.batch_decode(
                output, skip_special_tokens=True
            )
            translations.extend(decoded)
            log.debug(f"  {min(i+batch_size, len(texts))}/{len(texts)} translated")

        elapsed  = time.perf_counter() - start
        tps      = len(texts) / elapsed

        result = MTResult(
            model            = self._model_name,
            src_lang         = cfg["src_lang"],
            tgt_lang         = cfg["tgt_lang"],
            elapsed_sec      = round(elapsed, 2),
            segments_per_sec = round(tps, 3),
            num_segments     = len(texts),
            full_translation = " ".join(translations),
            translations     = translations,
        )

        log.info(
            f"MT done | {elapsed:.2f}s | {tps:.2f} seg/s | "
            f"{len(texts)} segments"
        )

        if save_path:
            result.save(save_path)

        return result

    def compute_bleu(self, hypotheses: List[str],
                     references: List[str]) -> dict:
        from sacrebleu.metrics import BLEU, CHRF
        bleu_score = BLEU().corpus_score(hypotheses, [references])
        chrf_score = CHRF().corpus_score(hypotheses, [references])
        scores = {
            "bleu": round(bleu_score.score, 2),
            "chrf": round(chrf_score.score, 2),
        }
        log.info(f"BLEU: {scores['bleu']} | chrF: {scores['chrf']}")
        return scores


class HelsinkiTranslator:
    """
    Helsinki-NLP/opus-mt-swc-en baseline translator.
    Lighter and faster than NLLB — used for comparison only.
    """

    def __init__(self, config: dict):
        self.config      = config
        self.model       = None
        self.tokenizer   = None
        self._loaded     = False
        self._model_name = config["mt"]["baseline_model"]

    def load(self) -> "HelsinkiTranslator":
        from transformers import MarianMTModel, MarianTokenizer

        log.info(f"Loading {self._model_name}...")
        t = time.perf_counter()

        device = self.config["mt"]["_device"]
        self.tokenizer = MarianTokenizer.from_pretrained(self._model_name)
        self.model     = MarianMTModel.from_pretrained(self._model_name).to(device)
        self.model.eval()

        self._loaded = True
        self._device = device
        log.info(f"Helsinki loaded in {time.perf_counter()-t:.2f}s")
        return self

    def translate(self, texts: List[str],
                  save_path: Optional[str] = None) -> MTResult:
        if not self._loaded:
            self.load()

        if isinstance(texts, str):
            texts = [texts]

        cfg        = self.config["mt"]
        batch_size = cfg["batch_size_gpu"] if self._device == "cuda" \
                     else cfg["batch_size_cpu"]

        translations = []
        start        = time.perf_counter()

        for i in range(0, len(texts), batch_size):
            batch  = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors = "pt",
                padding        = True,
                truncation     = True,
                max_length     = cfg["max_length"]
            ).to(self._device)

            with torch.no_grad():
                output = self.model.generate(**inputs)

            decoded = self.tokenizer.batch_decode(
                output, skip_special_tokens=True
            )
            translations.extend(decoded)

        elapsed = time.perf_counter() - start
        tps     = len(texts) / elapsed

        result = MTResult(
            model            = self._model_name,
            src_lang         = "sw",
            tgt_lang         = "en",
            elapsed_sec      = round(elapsed, 2),
            segments_per_sec = round(tps, 3),
            num_segments     = len(texts),
            full_translation = " ".join(translations),
            translations     = translations,
        )

        log.info(f"Helsinki MT done | {elapsed:.2f}s | {tps:.2f} seg/s")

        if save_path:
            result.save(save_path)

        return result
