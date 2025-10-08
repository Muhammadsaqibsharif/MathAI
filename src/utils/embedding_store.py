import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingStore:
    """Simple local embedding store backed by numpy files.

    - Uses SentenceTransformer to encode texts.
    - Persists embeddings and metadata under the given directory as .npz and .json files.
    - Retrieval uses cosine similarity computed with numpy (no external vector DB required).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./vector_store"):
        self.model = SentenceTransformer(model_name)
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory

        # Files for storing ids, embeddings and metadata
        self._idx_path = os.path.join(persist_directory, "simple_store.npz")
        self._meta_path = os.path.join(persist_directory, "simple_store_meta.json")

        # load existing store if present
        if os.path.exists(self._idx_path) and os.path.exists(self._meta_path):
            arr = np.load(self._idx_path, allow_pickle=True)
            self._ids = arr["ids"].tolist()
            self._embeddings = arr["embeddings"].tolist()
            with open(self._meta_path, "r", encoding="utf8") as f:
                self._metadatas = json.load(f)
        else:
            self._ids = []
            self._embeddings = []
            self._metadatas = []

    def index_documents(self, ids: List[str], documents: List[str], metadatas: List[dict]):
        embeddings = self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)

        # append to simple numpy store
        for i, _id in enumerate(ids):
            self._ids.append(_id)
            self._embeddings.append(embeddings[i].tolist())
            # store doc text inside metadata for retrieval
            md = metadatas[i] if i < len(metadatas) else {}
            md["document"] = documents[i]
            self._metadatas.append(md)

        # persist
        np.savez(self._idx_path, ids=np.array(self._ids, dtype=object), embeddings=np.array(self._embeddings, dtype=object))
        with open(self._meta_path, "w", encoding="utf8") as f:
            json.dump(self._metadatas, f, ensure_ascii=False)

    def retrieve(self, query: str, k: int = 4):
        q_emb = self.model.encode([query], convert_to_numpy=True)[0].tolist()

        if len(self._embeddings) == 0:
            return []

        # compute cosine similarity (normalized dot product)
        emb_mat = np.array(self._embeddings, dtype=float)
        q = np.array(q_emb, dtype=float)

        def norm(a):
            denom = np.linalg.norm(a, axis=1, keepdims=True)
            denom[denom == 0] = 1e-9
            return a / denom

        emb_norm = norm(emb_mat)
        q_norm = q / (np.linalg.norm(q) + 1e-9)
        sims = emb_norm.dot(q_norm)
        top_idx = np.argsort(-sims)[:k]
        docs = []
        for idx in top_idx:
            meta = self._metadatas[int(idx)]
            doc_text = meta.get("document")
            docs.append({"document": doc_text, "metadata": meta})
        return docs

    def list_top_sources(self, limit: int = 10):
        # return list of top source filenames present in the collection metadata
        try:
            sources = [m.get('source', 'unknown') for m in self._metadatas]
            uniq = []
            for s in sources:
                if s not in uniq:
                    uniq.append(s)
            return uniq[:limit]
        except Exception:
            return []

    def delete_collection(self):
        try:
            # remove simple store files
            try:
                if os.path.exists(self._idx_path):
                    os.remove(self._idx_path)
                if os.path.exists(self._meta_path):
                    os.remove(self._meta_path)
            except Exception:
                pass
        except Exception:
            pass
