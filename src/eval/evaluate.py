"""
Evaluation Module — WER, BLEU, chrF scoring on Common Voice Swahili.
"""
import os
import json
import time
import numpy as np
import soundfile as sf
from typing import List, Dict
from src.utils.logger import get_logger

log = get_logger(__name__)


class Evaluator:
    """
    Runs evaluation of ASR (WER/CER) and MT (BLEU/chrF)
    on Mozilla Common Voice Swahili test set.
    """

    def __init__(self, config: dict):
        self.config = config

    def load_common_voice(self) -> list:
        from datasets import load_dataset
        from itertools import islice
        cfg = self.config["eval"]
        log.info(f"Loading Common Voice {cfg['language']} / {cfg['split']} (streaming)...")
        # streaming=True avoids downloading every split's archive upfront —
        # Common Voice's loading script otherwise fetches train/dev/test/
        # other/invalidated (several GB) even when only `split` is requested.
        ds = load_dataset(
            cfg["dataset"],
            cfg["language"],
            split             = cfg["split"],
            streaming         = True,
            trust_remote_code = True,
        )
        samples = list(islice(ds, cfg["max_samples"]))
        log.info(f"Loaded {len(samples)} samples")
        return samples

    def evaluate_asr(self, asr_model, save_path: str = None) -> Dict:
        """
        Run ASR on Common Voice samples, compute WER and CER.
        """
        from jiwer import wer, cer

        ds          = self.load_common_voice()
        hypotheses  = []
        references  = []
        latencies   = []

        for i, sample in enumerate(ds):
            # Write temp wav
            tmp = f"data/processed/_eval_tmp_{i}.wav"
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr    = sample["audio"]["sampling_rate"]
            sf.write(tmp, audio, sr)

            # Transcribe
            t_start = time.perf_counter()
            segs, _ = asr_model.model.transcribe(
                tmp,
                language                   = "sw",
                condition_on_previous_text = False,
                vad_filter                 = True,
            )
            latencies.append(time.perf_counter() - t_start)

            hyp = " ".join(s.text.strip() for s in segs)
            hypotheses.append(hyp)
            references.append(sample["sentence"])

            os.remove(tmp)
            if (i + 1) % 10 == 0:
                log.info(f"  {i+1}/{len(ds)} evaluated...")

        wer_score = wer(references, hypotheses)
        cer_score = cer(references, hypotheses)
        avg_lat   = np.mean(latencies)

        results = {
            "model"          : f"whisper-{self.config['asr']['model']}",
            "dataset"        : self.config["eval"]["dataset"],
            "n_samples"      : len(ds),
            "wer"            : round(wer_score, 4),
            "cer"            : round(cer_score, 4),
            "avg_latency_sec": round(avg_lat,   4),
            "samples": [
                {"reference": references[i], "hypothesis": hypotheses[i],
                 "latency": round(latencies[i], 4)}
                for i in range(len(hypotheses))
            ]
        }

        log.info(f"ASR Eval | WER: {wer_score:.4f} | CER: {cer_score:.4f} | "
                 f"Avg latency: {avg_lat:.3f}s")

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            log.info(f"ASR eval saved: {save_path}")

        return results

    def evaluate_mt(self, mt_model, source_texts: List[str],
                    reference_texts: List[str],
                    save_path: str = None) -> Dict:
        """
        Translate source texts and score against references.
        """
        from sacrebleu.metrics import BLEU, CHRF

        mt_result  = mt_model.translate(source_texts)
        hypotheses = mt_result.translations

        bleu = BLEU().corpus_score(hypotheses, [reference_texts])
        chrf = CHRF().corpus_score(hypotheses, [reference_texts])

        results = {
            "model"    : mt_result.model,
            "n_samples": len(source_texts),
            "bleu"     : round(bleu.score, 2),
            "chrf"     : round(chrf.score, 2),
            "elapsed"  : mt_result.elapsed_sec,
            "seg_per_sec": mt_result.segments_per_sec,
            "samples"  : [
                {"source": source_texts[i],
                 "hypothesis": hypotheses[i],
                 "reference": reference_texts[i]}
                for i in range(len(source_texts))
            ]
        }

        log.info(f"MT Eval | BLEU: {bleu.score:.2f} | chrF: {chrf.score:.2f}")

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            log.info(f"MT eval saved: {save_path}")

        return results

    def compare_mt_models(self, nllb_model, helsinki_model,
                          source_texts: List[str],
                          reference_texts: List[str]) -> Dict:
        """
        Side-by-side NLLB vs Helsinki comparison.
        """
        log.info("Comparing NLLB-200 vs Helsinki-NLP...")
        nllb_results = self.evaluate_mt(
            nllb_model, source_texts, reference_texts,
            save_path=f"{self.config['paths']['results_eval']}/mt_nllb_eval.json"
        )
        helsinki_results = self.evaluate_mt(
            helsinki_model, source_texts, reference_texts,
            save_path=f"{self.config['paths']['results_eval']}/mt_helsinki_eval.json"
        )

        comparison = {
            "nllb"    : nllb_results,
            "helsinki": helsinki_results,
            "winner"  : "nllb" if nllb_results["bleu"] >= helsinki_results["bleu"]
                        else "helsinki",
        }

        path = f"{self.config['paths']['results_eval']}/mt_comparison.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        log.info(
            f"MT Comparison | NLLB BLEU: {nllb_results['bleu']} | "
            f"Helsinki BLEU: {helsinki_results['bleu']} | "
            f"Winner: {comparison['winner'].upper()}"
        )
        return comparison
