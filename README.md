# NPTEL Lecture Translation Pipeline

Translate English NPTEL lecture videos into **Hindi**, **Telugu**, and **Odia** — with subtitles and dubbed audio.

## Pipeline Flow

```
Input Video (.mp4)
    │
    ▼
┌──────────────────┐
│  Audio Extraction │  (ffmpeg)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  English STT     │  (Sarvam / Gemini)
│  + timestamps    │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Translation     │  (Sarvam / Gemini)
│  → hi / te / od  │
└────────┬─────────┘
         ▼
┌──────────────────────────────────────┐
│  Subtitle Generation (.srt)          │
│  en + hi + te + od with timestamps   │
└────────┬─────────────────────────────┘
         ▼
┌──────────────────┐
│  TTS Generation  │  (Sarvam)
│  → hi / te / od  │
└──────────────────┘
```

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install ffmpeg (must be on PATH)
# Windows: winget install ffmpeg
# Or download from https://ffmpeg.org/download.html

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your Sarvam & Gemini API keys
```

## Usage

### GUI (recommended for most users)

```bash
streamlit run app.py
```

This opens a web interface where you can upload a video, pick languages, and download results.

### CLI

```bash
python main.py input/lecture.mp4
python main.py input/lecture.mp4 --stt gemini --translate sarvam --no-tts
```

## Project Structure

```
BTP/
├── config/
│   └── settings.py            # API keys, language configs, paths
├── src/
│   ├── pipeline.py            # Main orchestrator
│   ├── audio_extractor.py     # ffmpeg audio extraction
│   ├── transcriber.py         # English STT (Sarvam / Gemini)
│   ├── translator.py          # Text translation (Sarvam / Gemini)
│   ├── subtitle_generator.py  # .srt generation with timestamps
│   └── tts_generator.py       # Text-to-Speech (Sarvam)
├── app.py                     # Streamlit GUI
├── input/                     # Place lecture videos here
├── output/                    # Generated subtitles + audio
├── main.py                    # CLI entry point
├── requirements.txt
└── .env                       # API keys (not committed)
```

## APIs Used

| API | Purpose |
|-----|---------|
| **Sarvam AI** | STT, Translation, TTS for Indian languages |
| **Google Gemini** | Fallback/alternative for STT & Translation |

## Languages

| Code | Language |
|------|----------|
| `en` | English (source) |
| `hi` | Hindi |
| `te` | Telugu |
| `od` | Odia |
