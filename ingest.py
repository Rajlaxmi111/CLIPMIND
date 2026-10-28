import json
import os
import requests
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("clipmind")

def get_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings", json={
        "model": "bge-m3",
        "prompt": text
    })
    return r.json()["embedding"]

for filename in os.listdir("jsons"):
    with open(f"jsons/{filename}") as f:
        data = json.load(f)

    for i, chunk in enumerate(data["chunks"]):
        embedding = get_embedding(chunk["text"])
        collection.add(
            ids=[f"{filename}_{i}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "number": chunk["number"],
                "title": chunk["title"],
                "start": chunk["start"]
            }]
        )

print("Ingestion complete")