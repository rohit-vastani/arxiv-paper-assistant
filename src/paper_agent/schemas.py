from pydantic import BaseModel, Field

class Passage(BaseModel):
    text: str
    page: int
    source: str
    score: float

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None  # omit to start a new conversation
    arxiv_id: str | None = None   # paper selected in the UI
 
 
class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    passages: list[dict] = []     # passages retrieved for THIS answer (page, score, text, source)