import sys
import types
import unittest.mock as mock


def install_all_stubs():
    _stub_unstructured()
    _stub_sentence_transformers()
    _stub_pinecone_and_openai()


def _stub_unstructured():
    if "unstructured" in sys.modules:
        return
    pkg = types.ModuleType("unstructured")
    pm = types.ModuleType("unstructured.partition")
    am = types.ModuleType("unstructured.partition.auto")
    am.partition = lambda **kw: [types.SimpleNamespace(text="extracted text")]
    sys.modules.update(
        {
            "unstructured": pkg,
            "unstructured.partition": pm,
            "unstructured.partition.auto": am,
        }
    )


def _stub_sentence_transformers():
    if "sentence_transformers" in sys.modules:
        return
    st = types.ModuleType("sentence_transformers")

    class FakeCrossEncoder:
        def __init__(self, *a, **kw):
            pass

        def predict(self, pairs):
            # Return equal scores so order is stable
            return [0.5] * len(pairs)

    st.CrossEncoder = FakeCrossEncoder
    sys.modules["sentence_transformers"] = st


def _stub_pinecone_and_openai():
    """Provide minimal stubs so module-level code in rag.py doesn't fail."""
    # langchain_pinecone
    if "langchain_pinecone" not in sys.modules:
        lp = types.ModuleType("langchain_pinecone")
        fake_vdb = mock.MagicMock()
        fake_vdb.similarity_search.return_value = []
        lp.PineconeVectorStore = mock.MagicMock(return_value=fake_vdb)
        sys.modules["langchain_pinecone"] = lp

    # langchain_openai
    if "langchain_openai" not in sys.modules:
        lo = types.ModuleType("langchain_openai")
        lo.ChatOpenAI = mock.MagicMock(return_value=mock.MagicMock())
        lo.OpenAIEmbeddings = mock.MagicMock(return_value=mock.MagicMock())
        sys.modules["langchain_openai"] = lo
