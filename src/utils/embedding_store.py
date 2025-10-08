import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMA = True
except Exception:
    chromadb = None
    Settings = None
    _HAS_CHROMA = False


class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        self.model = SentenceTransformer(model_name)
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self.collection_name = "mathai_collection"

        # Try to initialize Chroma client; if it fails due to deprecated settings or import, fall back to a simple numpy store
        self.use_chroma = False
        if _HAS_CHROMA:
            try:
                # Use the new Chroma Settings API. Avoid passing deprecated options.
                self.client = chromadb.Client(Settings(persist_directory=persist_directory))
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                except Exception:
                    try:
                        self.collection = self.client.create_collection(name=self.collection_name)
                    except TypeError:
                        self.collection = self.client.create_collection(self.collection_name)
                self.use_chroma = True
            except ValueError as e:
                # Known migration/deprecation error from chromadb config; fall back
                print("Chroma initialization error, falling back to numpy store:", e)
                self.use_chroma = False
            except Exception as e:
                print("Chroma unavailable or failed to initialize, falling back to numpy store:", e)
                self.use_chroma = False

        if not self.use_chroma:
            # Fallback simple vector store using numpy for embeddings and a metadata JSON
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
        if self.use_chroma:
            try:
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist())
            except Exception:
                # try recreate
                try:
                    self.client.delete_collection(self.collection_name)
                except Exception:
                    pass
                self.collection = self.client.create_collection(name=self.collection_name)
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist())
        else:
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
        if self.use_chroma:
            res = self.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
            docs = []
            for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
                docs.append({"document": doc, "metadata": meta})
            return docs
        else:
            if len(self._embeddings) == 0:
                return []
            # compute cosine similarity
            emb_mat = np.array(self._embeddings, dtype=float)
            q = np.array(q_emb, dtype=float)
            # normalize
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
            if self.use_chroma:
                metas = self.collection.get(include=["metadatas"])['metadatas']
                sources = [m.get('source', 'unknown') for m in metas]
            else:
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
            if self.use_chroma:
                self.client.delete_collection(name=self.collection_name)
            else:
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
