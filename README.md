# 🎙️ Real-Time Swahili–English AI Interpretation System

**NM-AIST DSAI Capstone Project**
**Author:** Fred | ICT & Statistics Unit, AICC | NM-AIST DSAI Program

---

## Overview

A cascade neural pipeline that interprets spoken Swahili into English speech in near real-time.

```
Audio (Swahili) ──► ASR ──► Text (Swahili) ──► MT ──► Text (English) ──► TTS ──► Audio (English)
                  Whisper               NLLB-200                        XTTS-v2
                  large-v3              distilled-600M
```

---

## Project Structure

```
swahili-interpreter/
├── src/
│   ├── asr/            # ASR module (Whisper)
│   ├── mt/             # MT module (NLLB-200 + Helsinki-NLP)
│   ├── tts/            # TTS module (Coqui XTTS-v2 + Edge-TTS)
│   ├── pipeline/       # Full cascade pipeline
│   ├── eval/           # WER, BLEU, chrF evaluation
│   └── utils/          # Shared utilities
├── configs/            # YAML configuration files
├── notebooks/          # Google Colab notebooks
├── data/               # Audio data (gitignored)
├── results/            # Output files (gitignored)
├── tests/              # Unit tests
└── docs/               # Documentation
```

---

## Quick Start (Google Colab)

Open the main notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/swahili-interpreter/blob/main/notebooks/main_pipeline.ipynb)

> Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username.

---

## Local Setup (Windows)

```powershell
git clone https://github.com/YOUR_GITHUB_USERNAME/swahili-interpreter.git
cd swahili-interpreter
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Pipeline Components

| Stage | Model | Device | Metric |
|---|---|---|---|
| ASR | Whisper large-v3 | GPU float16 | WER |
| MT  | NLLB-200-distilled-600M | GPU float16 | BLEU / chrF |
| TTS | Coqui XTTS-v2 | GPU | RTF / MOS |

---

## Phase 1 Baseline Results

| Component | Model | Latency (CPU) | Latency (GPU) |
|---|---|---|---|
| ASR | Whisper large-v3 | ~103 min | ~3–5 min |
| MT  | NLLB-200-distilled-600M | 88.05s | ~5–8s |
| MT  | Helsinki-NLP opus-swc-en | 58.71s | ~3–5s |
| TTS | Edge-TTS / Coqui XTTS-v2 | <5s | <2s |

---

## License

MIT License — see [LICENSE](LICENSE)
