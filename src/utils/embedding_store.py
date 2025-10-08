import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List


class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_directory: str = "./chroma_db"):
        self.model = SentenceTransformer(model_name)
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
        self.collection_name = "mathai_collection"
        # create or get collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=self.collection_name)

    def index_documents(self, ids: List[str], documents: List[str], metadatas: List[dict]):
        embeddings = self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        # if collection exists, add; otherwise create
        try:
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist())
        except Exception:
            # try recreate
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist())

    def retrieve(self, query: str, k: int = 4):
        q_emb = self.model.encode([query], convert_to_numpy=True)[0].tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
        docs = []
        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
            docs.append({"document": doc, "metadata": meta})
        return docs

    def list_top_sources(self, limit: int = 10):
        # return list of top source filenames present in the collection metadata
        try:
            metas = self.collection.get(include=["metadatas"])['metadatas']
            sources = [m.get('source', 'unknown') for m in metas]
            uniq = []
            for s in sources:
                if s not in uniq:
                    uniq.append(s)
            return uniq[:limit]
        except Exception:
            return []

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
