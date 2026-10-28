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

def ask(question):
    q_embedding = get_embedding(question)
    results = collection.query(query_embeddings=[q_embedding], n_results=3)
    context = "\n".join(results["documents"][0])

    prompt = f"Answer the question using only this context:\n\n{context}\n\nQuestion: {question}"

    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False
    })
    return r.json()["response"]

if __name__ == "__main__":
    question = input("Ask: ")
    print(ask(question))