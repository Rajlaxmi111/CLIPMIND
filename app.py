import os
import re
import shutil
import subprocess
import threading
import uuid
from typing import Optional

import chromadb
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

OLLAMA = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "llama3.1"
WHISPER_MODEL = "medium"

# Homebrew yt-dlp (always up to date, works with any Python version)
YTDLP = "/opt/homebrew/bin/yt-dlp" if os.path.exists("/opt/homebrew/bin/yt-dlp") else shutil.which("yt-dlp")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("clipmind")

_whisper = None


def get_whisper():
    global _whisper
    if _whisper is None:
        import whisper
        _whisper = whisper.load_model(WHISPER_MODEL)
    return _whisper


def get_embedding(text):
    r = requests.post(f"{OLLAMA}/api/embeddings", json={"model": EMBED_MODEL, "prompt": text})
    return r.json()["embedding"]


def cosine(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def relevance(cos):
    """Map cosine similarity to a 0-100 match score (tuned for bge-m3: ~0.35 unrelated, ~0.75 strong)."""
    return round(max(0.0, min(1.0, (cos - 0.35) / 0.40)) * 100)


def level_of(score):
    return "high" if score >= 70 else "medium" if score >= 45 else "low"


# ---------- models ----------
class Question(BaseModel):
    question: str
    video: Optional[str] = None  # video number to restrict search to


class IngestReq(BaseModel):
    url: str


# ---------- health ----------
@app.get("/api/health")
def health():
    try:
        requests.get(OLLAMA, timeout=2)
        ollama = True
    except Exception:
        ollama = False
    return {"ollama": ollama, "chunks": collection.count()}


# ---------- ask ----------
BEFORE, AFTER = 25, 55   # seconds of transcript pulled in around each matching clip

PROMPT = """You answer questions about a student's course lecture recordings, using ONLY the lecture excerpts below. Each excerpt shows the video title and the time it starts.

Rules:
- Use only what the excerpts say. Do not use outside knowledge and do not guess.
- If the question asks which video covers a topic, name the video title and the timestamp.
- For a worked example or calculation, explain it step by step the way the lecturer does, and give the final result the lecturer reaches.
- If the excerpts do not contain the answer, reply with exactly: NOT_FOUND

Lecture excerpts:
{context}

Question: {question}
Answer:"""

NOT_FOUND_MSG = "I can't find that in your lectures. Try rephrasing, or add a video that covers it."


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"


def keywords(question):
    """Distinctive tokens that must match literally, e.g. the number in '1341/9'."""
    return list(dict.fromkeys(re.findall(r"\d{3,}", question)))


def find_hits(question, q_emb, video):
    """Vector search plus exact keyword search, so numbers and rare words are not missed."""
    base_where = {"number": video} if video else None
    fields = ["documents", "metadatas", "embeddings"]
    hits = {}

    kw = {"query_embeddings": [q_emb], "n_results": 8, "include": fields}
    if base_where:
        kw["where"] = base_where
    r = collection.query(**kw)
    for i, d, m, e in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["embeddings"][0]):
        hits[i] = {"doc": d, "meta": m, "score": relevance(cosine(q_emb, e))}

    for tok in keywords(question):
        g = collection.get(where_document={"$contains": tok}, where=base_where, include=fields, limit=8)
        for i, d, m, e in zip(g["ids"], g["documents"], g["metadatas"], g["embeddings"]):
            s = max(relevance(cosine(q_emb, e)), 70)  # exact match on a number from the question
            if i not in hits or hits[i]["score"] < s:
                hits[i] = {"doc": d, "meta": m, "score": s}
    return list(hits.values())


def build_passages(hits):
    """A single clip is one short sentence. Pull in the transcript around it so the
    model sees the whole explanation, and merge clips that sit close together."""
    by_video = {}
    for h in hits:
        by_video.setdefault(h["meta"]["number"], []).append(h)

    passages = []
    for number, group in by_video.items():
        group.sort(key=lambda h: h["meta"]["start"])
        windows = []
        for h in group:
            lo, hi = h["meta"]["start"] - BEFORE, h["meta"]["start"] + AFTER
            if windows and lo <= windows[-1]["hi"]:
                windows[-1]["hi"] = max(windows[-1]["hi"], hi)
                windows[-1]["score"] = max(windows[-1]["score"], h["score"])
            else:
                windows.append({"lo": lo, "hi": hi, "score": h["score"], "meta": h["meta"]})
        for w in windows:
            segs = collection.get(
                where={"$and": [{"number": number}, {"start": {"$gte": w["lo"]}}, {"start": {"$lte": w["hi"]}}]},
                include=["documents", "metadatas"],
            )
            rows = sorted(zip(segs["metadatas"], segs["documents"]), key=lambda r: r[0]["start"])
            if not rows:
                continue
            passages.append({
                "title": w["meta"]["title"], "number": number,
                "start": rows[0][0]["start"], "source_url": w["meta"].get("source_url", ""),
                "text": " ".join(d.strip() for _, d in rows), "relevance": w["score"],
            })
    passages.sort(key=lambda p: p["relevance"], reverse=True)
    return passages[:4]


@app.post("/api/ask")
def ask(q: Question):
    if collection.count() == 0:
        return {"answer": "No lectures are indexed yet. Add a video first.", "sources": [],
                "confidence": 0, "level": "none", "found": False}

    q_embedding = get_embedding(q.question)
    hits = find_hits(q.question, q_embedding, q.video)
    passages = build_passages(hits) if hits else []
    if not passages:
        return {"answer": NOT_FOUND_MSG, "sources": [], "confidence": 0, "level": "none", "found": False}

    context = "\n\n".join(f"[{p['title']} - from {fmt_time(p['start'])}]\n{p['text']}" for p in passages)
    r = requests.post(f"{OLLAMA}/api/generate", json={
        "model": LLM_MODEL,
        "prompt": PROMPT.format(context=context, question=q.question),
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    })
    answer = r.json()["response"].strip()

    # The model says NOT_FOUND when the excerpts don't contain the answer.
    if re.search(r"\bNOT[_ ]FOUND\b", answer):
        return {"answer": NOT_FOUND_MSG, "sources": passages, "confidence": 0, "level": "none", "found": False}

    scores = [p["relevance"] for p in passages]
    top3 = scores[:3]
    confidence = round(0.6 * scores[0] + 0.4 * (sum(top3) / len(top3)))
    return {"answer": answer, "sources": passages, "confidence": confidence,
            "level": level_of(confidence), "found": True}


# ---------- library ----------
@app.get("/api/videos")
def videos():
    data = collection.get(include=["metadatas"])
    out = {}
    for m in data["metadatas"]:
        n = m["number"]
        v = out.setdefault(n, {
            "number": n, "title": m["title"], "chunks": 0,
            "duration": 0, "source_url": m.get("source_url", ""),
        })
        v["chunks"] += 1
        v["duration"] = max(v["duration"], m.get("end", m["start"]))
    return sorted(out.values(), key=lambda v: int(v["number"]) if str(v["number"]).isdigit() else 0)


@app.delete("/api/videos/{number}")
def delete_video(number: str):
    collection.delete(where={"number": number})
    return {"ok": True}


# ---------- ingest a YouTube video ----------
jobs = {}


def next_number():
    data = collection.get(include=["metadatas"])
    nums = [int(m["number"]) for m in data["metadatas"] if str(m["number"]).isdigit()]
    return max(nums, default=0) + 1


def clean(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text).strip()


def run_ingest(job_id, url):
    job = jobs[job_id]
    try:
        if not YTDLP:
            raise RuntimeError("yt-dlp not found. Run: brew install yt-dlp deno")

        number = next_number()
        job.update(status="running", step="Downloading audio…", progress=5)
        os.makedirs("audios", exist_ok=True)

        # 1. get the title
        t = subprocess.run([YTDLP, "--no-playlist", "--print", "title", url],
                           capture_output=True, text=True)
        if t.returncode != 0:
            raise RuntimeError(clean(t.stderr).splitlines()[-1] if t.stderr else "Could not read video info")
        title = t.stdout.strip().splitlines()[0] if t.stdout.strip() else "untitled"
        safe = re.sub(r'[\\/:"*?<>|]+', "", title).strip()

        # 2. download audio as mp3
        d = subprocess.run([
            YTDLP, "--no-playlist", "-x", "--audio-format", "mp3", "--audio-quality", "192K",
            "-o", f"audios/{number}_{safe}.%(ext)s", url,
        ], capture_output=True, text=True)
        audio_path = f"audios/{number}_{safe}.mp3"
        if d.returncode != 0 or not os.path.exists(audio_path):
            raise RuntimeError(clean(d.stderr).splitlines()[-1] if d.stderr else "Download failed")

        # 3. transcribe
        job.update(step="Transcribing (this can take a few minutes)…", progress=20)
        result = get_whisper().transcribe(audio=audio_path, task="translate", word_timestamps=False)
        segments = result["segments"]

        # 4. embed + store
        ids, embs, docs, metas = [], [], [], []
        for i, seg in enumerate(segments):
            job.update(step=f"Embedding {i + 1}/{len(segments)}…", progress=50 + int(45 * (i + 1) / len(segments)))
            ids.append(f"yt{number}_{i}")
            embs.append(get_embedding(seg["text"]))
            docs.append(seg["text"])
            metas.append({
                "number": str(number), "title": title,
                "start": seg["start"], "end": seg["end"], "source_url": url,
            })
        collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        job.update(status="done", step="Done", progress=100, title=title)
    except Exception as e:
        job.update(status="error", step=clean(str(e)), progress=0)


@app.post("/api/ingest")
def ingest(req: IngestReq):
    if "youtu" not in req.url:
        raise HTTPException(400, "Please paste a YouTube link.")
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "step": "Queued…", "progress": 0}
    threading.Thread(target=run_ingest, args=(job_id, req.url), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/ingest/{job_id}")
def ingest_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[job_id]


# ---------- frontend (keep last) ----------
app.mount("/", StaticFiles(directory="static", html=True), name="static")