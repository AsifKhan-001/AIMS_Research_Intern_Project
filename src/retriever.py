import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

class HybridRetriever:
    """
    Combines Keyword search (BM25) with meaning-based Vector AI search (FAISS).
    """
    def __init__(self, corpus_chunks):
        self.chunks = corpus_chunks
        self.corpus_texts = [c["text"] for c in corpus_chunks]
        
        # 1. Initialize BM25 (Keyword Engine)
        tokenized_corpus = [text.lower().split(" ") for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Initialize FAISS Vector Storage (Semantic AI Engine)
        # We use a lightweight model perfect for laptops and free tiers
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.encoder.encode(self.corpus_texts, show_progress_bar=False)
        dimension = embeddings.shape[1]
        
        # Create the index and load the number vectors into it
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query, top_k=3, disable_hybrid=False):
        """
        Executes query matching. Supports toggle adjustments to handle ablation study testing!
        """
        # If the ablation study turns off hybrid retrieval, run FAISS vector search only
        if disable_hybrid:
            query_vector = self.encoder.encode([query]).astype("float32")
            _, faiss_indices = self.faiss_index.search(query_vector, top_k)
            return [self.chunks[idx] for idx in faiss_indices[0] if idx < len(self.chunks)]

        # Run full Reciprocal Rank Fusion (RRF) Hybrid Search
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = np.argsort(bm25_scores)[::-1]
        
        query_vector = self.encoder.encode([query]).astype("float32")
        _, faiss_ranked = self.faiss_index.search(query_vector, len(self.chunks))
        faiss_ranked = faiss_ranked[0]
        
        # Combine the ranking scores
        rrf_scores = {}
        constant = 60
        for rank, index in enumerate(bm25_ranked):
            rrf_scores[index] = rrf_scores.get(index, 0.0) + (1.0 / (constant + rank + 1))
        for rank, index in enumerate(faiss_ranked):
            rrf_scores[index] = rrf_scores.get(index, 0.0) + (1.0 / (constant + rank + 1))
            
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for index, score in sorted_indices[:top_k]:
            results.append(self.chunks[index])
        return results