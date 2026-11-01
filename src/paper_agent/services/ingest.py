import hashlib
from pathlib import Path
from paper_agent.services.vectorstore import get_vectorstore
from paper_agent.config import ResourcePath
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PATHS = ResourcePath()

def ingest_paper(arxiv_id: str, title: str) -> str:
    """
    Load the research papers from the directory and split using text splitter.
    After splitting the text store it into the chroma vector database

    Args:
        arxiv_id (str): Research paper Arxiv id
        title (str): Title of the research paper

    Returns: if loading, splitting and vector storing happen successfully then
             return data successfully stored into vector DB else error in
             particualr part
    """
    try:
        try:

            vs = get_vectorstore()
            PATHS = ResourcePath()
            docs = PyMuPDFLoader(str(Path(PATHS.pdf_dir) / f"{arxiv_id}.pdf")).load()

            for d in docs:
                d.metadata.update({"arxiv_id": arxiv_id, "title": title})

            chunks = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150).split_documents(docs)

        except Exception as e:
            raise Exception(f"Error while loading or splitting the data \nError:\n{e}")

        try:

            ids = [hashlib.md5(f"{arxiv_id}|{n}|{c.metadata.get("page")}|{c.page_content}".encode()).hexdigest()
                for n, c in enumerate(chunks)]

            done = set(vs.get()["ids"])
            todo = [(c, i) for c, i in zip(chunks, ids) if i not in done]
            for s in range(0, len(todo), 50):
                part = todo[s: s + 50]
                vs.add_documents([c for c, _ in part], ids=[i for _, i in part])

        except Exception as e:
            raise Exception(f"Error while storing the vectors into DB. \nError:\n{e}")

        return len(todo)
    
    except Exception as e:
        raise Exception(f"Something unusual happened. {e}")
