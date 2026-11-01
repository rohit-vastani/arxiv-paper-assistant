from pathlib import Path
from paper_agent.services.vectorstore import get_vectorstore
from paper_agent.schemas import Passage

def retrieve(query: str, arxiv_id: str):
    """
    Based on the user query find the most relevant chunks from the vector database.
    
    Args:
        query (str): User query
        arxiv_id (str): arxiv_id of the paper from which user asked question

    Returns:
        Most relevant chunk to user query
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k=5,
        filter={"arxiv_id": arxiv_id}
    )

    passages, seen = [], set()
    for doc, score in results:
        text = doc.page_content.strip()
        if text in seen or score < 0.65:
            continue

        seen.add(text)
        passages.append(Passage(
            text=text,
            page=doc.metadata.get("page", -1) + 1,
            source=Path(doc.metadata.get("source", "unknown")).name,
            score=round(score, 2),
        ))


    if not passages:
        summary = "NO_RELEVANT_CONTEXT: the paper does not appear to cover this query."
    else:
        summary = "\n\n---\n\n".join(
            f"[{i+1}] (page {p.page})\n{p.text}" for i, p in enumerate(passages)
        )

    return summary, [p.model_dump() for p in passages]