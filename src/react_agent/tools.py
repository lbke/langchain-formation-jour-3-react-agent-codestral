"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from react_agent.configuration import Configuration

# Test version with 2 dummy docs about LangChain and LangGraph
from react_agent.langchain_doc_retriever import get_langchain_doc_retriever
# Version that actually loads LangChain documentation
from react_agent.langsmith_doc_retriever_complete import get_langsmith_doc_retriever

# Uncomment to ingest actual LangChain doc,
# it will take some time to run on the first iteration (in minutes)
# result = langsmith_doc_retriever.invoke(query)
langchain_doc_retriever = get_langchain_doc_retriever()
langsmith_doc_retriever = get_langsmith_doc_retriever()


async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


def search_langchain(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search in LangChain and LangGraph official documentation.

    This tool should be used in priority over a generic web search. 

    It is is designed to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about LangChain and LangGraph.
    """
    # result = retriever.invoke(query)
    result = langchain_doc_retriever.invoke(query)
    return cast(list[dict[str, Any]], result)


def search_langsmith(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Search in LangSmith official documentation.

    This tool should be used in priority over a generic web search. 

    It is is designed to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about LangSmith
    """
    # result = retriever.invoke(query)
    result = langsmith_doc_retriever.invoke(query)
    return cast(list[dict[str, Any]], result)


TOOLS: List[Callable[..., Any]] = [search, search_langsmith, search_langchain]
