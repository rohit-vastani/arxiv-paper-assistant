from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from paper_agent.config import ResourcePath

PATHS = ResourcePath()

class NomicEmbeddings(OllamaEmbeddings):
    def embed_documents(self, texts):
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text):
        return super().embed_query(f"search_query: {text}")

llm = NomicEmbeddings(model="nomic-embed-text:latest")

def get_vectorstore():
    vectorstore = Chroma(
        persist_directory=PATHS.chroma_dir,
        collection_name="research_paper_data",
        collection_metadata={"hnsw:space": "cosine"},
        embedding_function=llm,
    )
    return vectorstore