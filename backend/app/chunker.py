from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]
