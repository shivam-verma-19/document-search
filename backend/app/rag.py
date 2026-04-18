import time
from backend.app.utils import build_prompt
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from .cache import get_cache, set_cache
from .metrics import log_metrics
from .utils import build_prompt
from .utils import get_secrets
from .monitoring import push_metric

# Cache hit
push_metric("CacheHit", 1)

# LLM fallback
push_metric("LLMFallback", 1)

secrets = get_secrets()

OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = secrets["PINECONE_API_KEY"]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

vector_db = PineconeVectorStore(index_name="rag-index", embedding=embeddings)

def ask_question(query):
    start = time.time()

    cached = get_cache(query)
    if cached:
        log_metrics(query, time.time()-start, "cache")
        return cached

    docs = vector_db.similarity_search(query, k=5)

    if not docs:
        ans = llm.invoke(query).content
        set_cache(query, ans)
        return ans

    context = "\n".join([d.page_content for d in docs])

    prompt = build_prompt(context, query)

    ans = llm.invoke(prompt).content

    set_cache(query, ans)
    log_metrics(query, time.time()-start, "rag")

    return ans


def summarize_doc(doc_id):
    docs = vector_db.similarity_search(doc_id, k=10)

    context = "\n".join([d.page_content for d in docs])

    return llm.invoke(f"Summarize:\n{context}").content
