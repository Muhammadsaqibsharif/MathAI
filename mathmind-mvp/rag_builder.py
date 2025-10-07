# rag_builder.py
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

MODEL_NAME = "all-MiniLM-L6-v2"

# Fallback imports for when FAISS isn't available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    from sklearn.metrics.pairwise import cosine_similarity

def load_text_chunks(path):
    chunks = []
    for fname in os.listdir(path):
        if fname.endswith(".txt"):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                text = f.read()
                # naive split by paragraphs
                for block in text.split("\n\n"):
                    block = block.strip()
                    if len(block) > 40:
                        chunks.append(block)
    return chunks

def build_index(text_chunks, model_name=MODEL_NAME, index_path="faiss_index.npy"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks, show_progress_bar=True)
    
    if FAISS_AVAILABLE:
        # Use FAISS if available
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, "faiss.index")
        print("FAISS index created successfully.")
    else:
        # Fallback to numpy for similarity search
        np.save("embeddings.npy", embeddings)
        print("Embeddings saved for in-memory similarity search.")
    
    # save chunks
    np.save("chunks.npy", np.array(text_chunks, dtype=object))
    print("Text chunks saved.")

def search_similar(query, model_name=MODEL_NAME, top_k=3):
    """Search for similar text chunks to the query"""
    if not os.path.exists("chunks.npy"):
        return ["No indexed content available."]
    
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    chunks = np.load("chunks.npy", allow_pickle=True)
    
    if FAISS_AVAILABLE and os.path.exists("faiss.index"):
        # Use FAISS search
        index = faiss.read_index("faiss.index")
        distances, indices = index.search(query_embedding, top_k)
        return [chunks[i] for i in indices[0]]
    elif os.path.exists("embeddings.npy"):
        # Use cosine similarity fallback
        embeddings = np.load("embeddings.npy")
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [chunks[i] for i in top_indices]
    else:
        return ["No search index available."]