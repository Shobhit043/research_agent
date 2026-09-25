"""Tool registry: builds the agent's tools and the matching prompt guidance.

Guidance lists only the tools actually bound for a call, because naming a tool the model
can't use makes gpt-oss try to call it anyway (see prompts.py).
"""

from langchain_core.tools import BaseTool

from agent.config import Settings
from agent.documents import DocumentStore
from agent.tools.document_tools import build_document_search_tool, build_read_document_tool
from agent.tools.knowledge_tools import build_arxiv_tool, build_wikipedia_tool
from agent.tools.place_tools import build_get_time_tool, build_place_info_tool
from agent.tools.web_tools import build_fetch_url_tool, build_web_search_tool

# Tools that only make sense once the user has uploaded something.
DOCUMENT_TOOLS = frozenset({"search_documents", "read_document"})

GUIDANCE = {
    "search_documents": "Use `search_documents` for questions about the user's uploaded documents.",
    "read_document": "Use `read_document` to read a whole document or page, e.g. to summarise it.",
    "web_search": "Use `web_search` for current, changing, or external information.",
    "fetch_url": "Use `fetch_url` to read a specific URL the user gives or a promising search result.",
    "wikipedia_search": "Use `wikipedia_search` for well-established background facts.",
    "arxiv_search": "Use `arxiv_search` for academic papers and research literature.",
    "get_time": "Use `get_time` for the current date or time anywhere; never assume what time it is.",
    "place_info": "Use `place_info` for facts about a place: country, coordinates, population, timezone.",
}
ALL_TOOLS = tuple(GUIDANCE)


def build_tools(store: DocumentStore, settings: Settings) -> dict[str, BaseTool]:
    enabled = set(settings.enabled_tools or ALL_TOOLS)
    unknown = enabled - set(ALL_TOOLS)
    if unknown:
        raise ValueError(f"Unknown tools in ENABLED_TOOLS: {', '.join(sorted(unknown))}. Known: {', '.join(ALL_TOOLS)}")
    builders = {
        "search_documents": lambda: build_document_search_tool(store, settings),
        "read_document": lambda: build_read_document_tool(store),
        "web_search": lambda: build_web_search_tool(settings),
        "fetch_url": lambda: build_fetch_url_tool(settings),
        "wikipedia_search": lambda: build_wikipedia_tool(settings),
        "arxiv_search": lambda: build_arxiv_tool(settings),
        "get_time": lambda: build_get_time_tool(settings),
        "place_info": lambda: build_place_info_tool(settings),
    }
    return {name: builders[name]() for name in ALL_TOOLS if name in enabled}


def tool_guidance(tools: list[BaseTool]) -> str:
    lines = ["- Use only the tools bound to you."]
    lines += [f"- {GUIDANCE[t.name]}" for t in tools if t.name in GUIDANCE]
    lines.append("- If the tools return nothing useful, say so plainly instead of guessing.")
    return "\n".join(lines)


__all__ = ["ALL_TOOLS", "DOCUMENT_TOOLS", "build_tools", "tool_guidance"]
