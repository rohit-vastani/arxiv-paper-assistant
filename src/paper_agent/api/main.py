import asyncio
import logging
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
 
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from langchain.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
 
from paper_agent.agent.builder import build_agent
from paper_agent.services.papers import add_paper, list_papers, search_papers
from paper_agent.schemas import ChatRequest, ChatResponse
from paper_agent.config import ResourcePath
 
load_dotenv()
log = logging.getLogger("uvicorn.error")
PATHS = ResourcePath()
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent() # create one time agent at the starting
    yield
 
app = FastAPI(title="Research Paper Assistant API", lifespan=lifespan)

 
def _text(content) -> str:
    """Gemini returns a list of content parts; other models return a plain string."""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    )
 
 
def _passages_this_turn(messages) -> list[dict]:
    """Collect passages from retrieve_tool calls made after the latest user message."""
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    seen: dict[str, dict] = {}
    for m in messages[last_human + 1:]:
        if isinstance(m, ToolMessage) and m.name == "retrieve_tool" and isinstance(m.artifact, list):
            for p in m.artifact:
                seen.setdefault(p["text"], p)
    return list(seen.values())
 
 
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/view-pdf/{filename}")
def view_pdf(filename: str):
    pdf_dir = Path(PATHS.pdf_dir).resolve()
    file_path = (pdf_dir / filename).resolve()

    if file_path.parent != pdf_dir or file_path.suffix != ".pdf" or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Pdf not found!")

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename,
        content_disposition_type="inline"
    )
    
 
@app.get("/search")
def search(title: str = Query(..., min_length=2)) -> list[dict]:
    """Candidate papers from arXiv. Nothing is downloaded."""
    try:
        return search_papers(title)
    except Exception as e:
        log.exception("arXiv search failed")
        raise HTTPException(status_code=502, detail=f"arXiv search failed: {e}")
 
 
@app.get("/papers")
def papers() -> list[dict]:
    """Papers that are downloaded and embedded."""
    return list_papers()
 
 
@app.post("/papers/{arxiv_id:path}")
async def add(arxiv_id: str) -> dict:
    """Download and embed one paper. Takes a while on CPU the first time."""
    # add_paper blocks while embedding, so run it in a worker thread with its own event loop
    # to keep the API responsive for other requests.
    message = await asyncio.to_thread(asyncio.run, add_paper(arxiv_id))
    listed = list_papers()
    if not (isinstance(listed, list) and any(p["arxiv_id"] == arxiv_id for p in listed)):
        raise HTTPException(status_code=400, detail=str(message))
    return {"arxiv_id": arxiv_id, "message": message}
 
 
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    text = f"[Selected paper: {req.arxiv_id}] {req.message}" if req.arxiv_id else req.message
    try:
        result = await request.app.state.agent.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:
        log.exception("agent call failed")
        raise HTTPException(status_code=502, detail=f"Agent error: {e}")
 
    messages = result["messages"]
    return ChatResponse(
        thread_id=thread_id,
        answer=_text(messages[-1].content) or "I couldn't produce an answer.",
        passages=_passages_this_turn(messages),
    )
 