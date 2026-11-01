import io
import arxiv
import httpx
import aiofiles
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from paper_agent.config import ResourcePath
from paper_agent.services.catalog import _load, _save, CATALOG
from paper_agent.services.ingest import ingest_paper

PATHS = ResourcePath()

def search_papers(title: str) -> list[dict]:
    """Search arXiv and list candidate papers (id, title, authors, year). Does NOT download."""
    client = arxiv.Client()

    query = title.replace('"', '').replace("'", "").strip()
    api_query = f'all:"{query}"'

    search = arxiv.Search(
        query=api_query,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = list(client.results(search))

    catalog, out = _load(), []
    for r in results:
        entry = {"arxiv_id": r.get_short_id(), "title": r.title,
                "authors": [a.name for a in r.authors][:3],
                "year": r.published.year, "pdf_url": r.pdf_url}

        catalog.setdefault(entry["arxiv_id"], {}).update(entry)
        out.append({k: v for k, v in entry.items() if k != "pdf_url"})
    _save(catalog)
    return out

async def add_paper(arxiv_id: str) -> str:
    """Download ONE chosen paper by arXiv id and embed it."""

    catalog = _load()
    entry = catalog.get(arxiv_id)

    if not entry:
        return "Unknown paper id. Call search_paper first and use an id from its results."

    pdf_path = Path(PATHS.pdf_dir) / f"{arxiv_id}.pdf"
    if not pdf_path.exists():
        try:
            await download_pdf(entry["pdf_url"], pdf_path, {"/Title": entry["title"]})
        except Exception as e:
            return f"Error while donwload the paper {e}"
    
    try:
        n = ingest_paper(arxiv_id, title=entry["title"])
        catalog[arxiv_id]["embedded"] = True
        _save(catalog)
        return "Already embedded." if n == 0 else f"Embedded {n} chunks."
    except Exception as e:
        return f"Embedding are not stored. {e}"

def list_papers() -> list[dict]:
    """List papers already downloaded and stored in vector database, with their arxiv_id and title."""
    
    if CATALOG.exists():
        return [{"arxiv_id": v["arxiv_id"], "title": v["title"]} for k, v in _load().items() if v.get("embedded")]

    return []

async def download_pdf(pdf_url: str, pdf_path: str, metadata: dict):

    buffer = io.BytesIO() # create buffere in memory

    # Create client
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        # stream the response
        async with client.stream("GET", pdf_url, follow_redirects=True) as response:
            response.raise_for_status()

            async for chunk in response.aiter_bytes():
                buffer.write(chunk)

    buffer.seek(0) # set the pointer at the begining

    reader = PdfReader(buffer) # read the pdf from buffer
    writer = PdfWriter() # write the pdf

    for page in reader.pages:
        writer.add_page(page) # add pages to the writer

    writer.add_metadata(metadata) # add metadata

    output_buffer = io.BytesIO()
    writer.write(output_buffer) # write the bytes to the output buffer
    output_buffer.seek(0)

    async with aiofiles.open(pdf_path, "wb") as file:
        await file.write(output_buffer.read()) # save pdf

    # close buffers
    buffer.close()
    output_buffer.close()