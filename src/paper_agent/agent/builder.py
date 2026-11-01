from dotenv import load_dotenv; load_dotenv()
from langchain.tools import tool
from langchain.agents import create_agent
from paper_agent.services.papers import search_papers, add_paper, list_papers
from paper_agent.services.retrieval import retrieve
from langgraph.checkpoint.memory import InMemorySaver
from paper_agent.agent.prompts import SYSTEM_PROMPT

@tool(response_format="content_and_artifact")
def retrieve_tool(query: str, arxiv_id: str) -> tuple:
    """Find the most relevant passages from an added paper. Use list_papers to get arxiv_id"""
    return retrieve(query, arxiv_id)


def build_agent():
    tools = [tool(search_papers), tool(add_paper), tool(list_papers), retrieve_tool]

    return create_agent(
        model="google_genai:gemini-3.1-flash-lite",
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=SYSTEM_PROMPT
    )