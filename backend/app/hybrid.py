from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        self.docs = documents or []

        # ✅ Handle empty documents
        if not self.docs:
            self.corpus = []
            self.bm25 = None
            return

        self.corpus = [
            (getattr(doc, "page_content", "") or "").lower().split()
            for doc in self.docs
        ]

        # ✅ Avoid BM25 init on empty corpus
        if not any(self.corpus):
            self.bm25 = None
        else:
            self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, k=5):
        # ✅ Guard all edge cases
        if not self.docs or not query or not self.bm25:
            return []

        tokenized_query = query.lower().split()
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:k]]