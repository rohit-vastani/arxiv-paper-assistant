import os
import uuid
 
import httpx
import streamlit as st
from dotenv import load_dotenv
 
load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")
 
st.set_page_config(page_title="Research Paper Assistant", page_icon="📄", layout="wide")
 
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("results", [])
 
 
def call(method: str, path: str, timeout: float = 60, **kwargs):
    """Call the API. Returns parsed JSON, or None after showing an error."""
    try:
        r = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except ValueError:
            detail = e.response.text
        st.error(str(detail))
    except httpx.HTTPError as e:
        st.error(f"Could not reach the API at {API_URL} ({e.__class__.__name__}). Is it running?")
    return None
 
 
def short(title: str, n: int = 70) -> str:
    return title if len(title) <= n else title[: n - 1] + "…"
 
 
def show_sources(passages: list[dict]) -> None:
    if not passages:
        return
    with st.expander(f"Sources ({len(passages)} passages)"):
        for p in passages:
            pdf_url = f"{API_URL}/view-pdf/{p['source']}#page={p['page']}"
            st.markdown(f"**Page {p['page']}** · similarity {p['score']} · [Open in PDF]({pdf_url})")
            st.text(p["text"])
 
 
# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Find a paper")
    with st.form("search_form"):
        query = st.text_input("Title or topic", placeholder="Attention Is All You Need")
        submitted = st.form_submit_button("Search arXiv")
    if submitted and query.strip():
        with st.spinner("Searching arXiv…"):
            found = call("GET", "/search", params={"title": query.strip()})
        if found is not None:
            ss.results = found
            if not found:
                st.info("No papers found.")
 
    if ss.results:
        by_id = {r["arxiv_id"]: r for r in ss.results}
        pick = st.selectbox(
            "Results",
            list(by_id),
            format_func=lambda i: f"{short(by_id[i]['title'])} ({by_id[i]['year']})",
        )
        chosen = by_id[pick]
        st.caption(f"{', '.join(chosen['authors'])} · arXiv:{chosen['arxiv_id']}")
        if st.button("Add this paper", type="primary"):
            with st.spinner("Downloading and embedding… the first time can take a minute or two."):
                done = call("POST", f"/papers/{chosen['arxiv_id']}", timeout=900)
            if done:
                st.success(done["message"])

 
    st.header("2. Ask about")
    papers = call("GET", "/papers") or []
    titles = {p["arxiv_id"]: p["title"] for p in papers}
    if titles:
        selected = st.selectbox("Paper", list(titles), format_func=lambda i: short(titles[i]))
    else:
        selected = None
        st.caption("No papers added yet.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("New chat"):
            ss.messages = []
            ss.thread_id = str(uuid.uuid4())
            st.rerun()
    with col2:
        st.link_button(
            "View PDF",
            f"{API_URL}/view-pdf/{selected}.pdf" if selected else API_URL,
            disabled=selected is None,
        )
        
 
# ---------------------------------------------------------------- main area
st.title("Research Paper Assistant")
st.caption("Search arXiv, add a paper, then ask questions. Answers use only the paper's text, with page citations.")
 
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("passages", []))
 
if prompt := st.chat_input("Ask a question about the selected paper"):
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            reply = call(
                "POST", "/chat", timeout=180,
                json={"message": prompt, "thread_id": ss.thread_id, "arxiv_id": selected},
            )
        if reply:
            st.markdown(reply["answer"])
            show_sources(reply["passages"])
            ss.messages.append(
                {"role": "assistant", "content": reply["answer"], "passages": reply["passages"]}
            )
