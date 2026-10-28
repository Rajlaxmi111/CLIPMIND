"""
Regenerates embeddings.joblib using OpenAI's embedding model instead of
local Ollama/bge-m3, so query-time embeddings (also OpenAI) live in the
same vector space as the stored ones. Run this once whenever your
jsons/ transcripts change.

Requires OPENAI_API_KEY set in a .env file (see .env.example).
"""
import os
import json
import pandas as pd
import joblib
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"  # cheap, fast, 1536-dim
BATCH_SIZE = 100


def embed_batch(text_list):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text_list)
    return [d.embedding for d in resp.data]


def create_embedding(text_list, batch_size=BATCH_SIZE):
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        all_embeddings.extend(embed_batch(batch))
    return all_embeddings


def main():
    jsons_dir = "jsons"
    my_dicts = []
    chunk_id = 0

    for json_file in os.listdir(jsons_dir):
        with open(f"{jsons_dir}/{json_file}") as f:
            content = json.load(f)
        print(f"Embedding {json_file} ({len(content['chunks'])} chunks)...")
        texts = [c["text"] for c in content["chunks"]]
        embeddings = create_embedding(texts)

        for i, chunk in enumerate(content["chunks"]):
            chunk.setdefault("source_url", None)  # local video files have no YouTube link
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embeddings[i]
            chunk_id += 1
            my_dicts.append(chunk)

    df = pd.DataFrame.from_records(my_dicts)
    joblib.dump(df, "embeddings.joblib")
    print(f"Saved embeddings.joblib with {len(df)} chunks.")


if __name__ == "__main__":
    main()