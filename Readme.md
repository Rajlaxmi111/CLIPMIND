# CLIPMIND — Web Edition

## One-time setup

1. `python3 -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `brew install ffmpeg` (if not already installed — required by both Whisper and yt-dlp)
4. `cp .env.example .env` and put your OpenAI API key inside

## Option A — Bulk-load existing videos (like before)

1. Drop video files into `videos/`
2. `python video_to_mp3.py`
3. `python mp3_to_json.py`
4. `python re_embed.py` — embeds everything in `jsons/` using OpenAI and writes `embeddings.joblib`

## Option B — Add videos from the website (new)

Just run the server (see below) and paste a YouTube link into the "Add video" box. It downloads the audio, transcribes it, embeds it, and appends it to `embeddings.joblib` automatically — no manual steps.

## Running the site

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`. Ask questions in the chat box, or add new videos via the YouTube link box — both use the same underlying dataset.

## Files

- `main.py` — the FastAPI server: `/ask` (answers questions) and `/ingest` (adds a YouTube video)
- `ingest.py` — YouTube download → transcribe → embed → append logic, used by `/ingest`
- `re_embed.py` — one-time bulk embedding of everything already in `jsons/`
- `static/index.html` — the web page
- `video_to_mp3.py`, `mp3_to_json.py` — still used for Option A (bulk local video processing)

## Deprecated

`pre_process_json.py` and `process_incoming.py` used local Ollama and are replaced by `re_embed.py` + `main.py` (OpenAI-based). Safe to delete, along with the `whisper/` folder (unused clone of the Whisper repo — the pip package `openai-whisper` is what's actually used).