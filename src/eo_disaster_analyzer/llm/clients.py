from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from loguru import logger

from eo_disaster_analyzer.config import get_settings
from eo_disaster_analyzer.data.providers import fetch_disaster_news
from eo_disaster_analyzer.llm.prompts import (ANALYZE_ARTICLE_PROMPT,
                                               CLASSIFY_RElevance_PROMPT)
from eo_disaster_analyzer.llm.schemas import DisasterEvent
from eo_disaster_analyzer.data.extractors import extract_entities_from_text


# ------------------------------------------------------------------------------
# LLM INITIALIZATION
# ------------------------------------------------------------------------------

_LLM_CACHE: Optional[ChatOpenAI] = None


def get_llm_client(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    """
    Initializes and returns a shared LangChain ChatOpenAI client.
    Ensures that only one instance is created (acts like a lightweight singleton).
    """
    global _LLM_CACHE
    if _LLM_CACHE is not None:
        return _LLM_CACHE

    settings = get_settings()
    api_key = settings.openai_api_key.get_secret_value()

    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment.")

    logger.info(f"Initializing LLM client (model={model_name})")

    _LLM_CACHE = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        temperature=0.0,
    )

    return _LLM_CACHE


# ------------------------------------------------------------------------------
# ARTICLE ANALYSIS
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# NEWS ANALYSIS PIPELINE
# ------------------------------------------------------------------------------

def run_news_analysis_pipeline(
    query: str = "flood OR flood disaster OR flooding",
    period: str = "7d",
    max_results: int = 10,
) -> List[tuple[DisasterEvent, dict[str, Any]]]:
    """
    Orchestrates a full pipeline:
    1. Fetch a broad set of news articles from Google News.
    2. Perform a fast, cheap LLM-based pre-filtering to find relevant articles.
    3. Run a detailed, structured LLM extraction only on the relevant articles.

    Args:
        query: Search query for Google News.
        period: Recency filter ('24h', '7d', etc.).
        max_results: Number of articles to fetch.

    Returns:
        A list of tuples, where each tuple contains the confirmed
        DisasterEvent and its original source article dictionary.
    """
    # 1. Fetch a broad set of articles
    articles = fetch_disaster_news(
        query=query,
        period=period,
        max_results=max_results,
    )

    if not articles:
        logger.warning("No news articles found.")
        return []

    # 2. Perform fast pre-filtering
    logger.info(f"Fetched {len(articles)} articles. Starting cheap pre-filtering...")
    llm = get_llm_client()
    relevance_chain = (
        CLASSIFY_RELEVANCE_PROMPT | llm | StrOutputParser()
    )

    try:
        relevance_results = relevance_chain.batch(
            articles, config={"max_concurrency": 10}
        )
    except Exception as e:
        logger.error(f"An error occurred during relevance filtering: {e}")
        return []

    relevant_articles = [
        article
        for article, result in zip(articles, relevance_results)
        if "relevant" in result.lower()
    ]

    if not relevant_articles:
        logger.warning("No relevant articles found after pre-filtering.")
        return []

    logger.info(
        f"{len(relevant_articles)} relevant articles found. "
        "Starting detailed extraction..."
    )

    # 3. Run detailed extraction only on the filtered list
    structured_llm = llm.with_structured_output(DisasterEvent)
    analysis_chain: Runnable = ANALYZE_ARTICLE_PROMPT | structured_llm

    # Prepare inputs for the final batch run
    batch_inputs = []
    for article in relevant_articles:
        text_to_analyze = f"{article.get('title', '')}. {article.get('summary', '')}"
        entities = extract_entities_from_text(text_to_analyze)
        batch_inputs.append({**article, **entities})

    try:
        disaster_events: List[DisasterEvent] = analysis_chain.batch(
            batch_inputs, config={"max_concurrency": 5}
        )
    except Exception as e:
        logger.error(f"An error occurred during detailed analysis: {e}")
        return []

    # Final confirmation (the LLM might still decide it's not a disaster)
    # Pair events with their original articles before the final filter.
    # The order is preserved from the batch call.
    paired_results = zip(disaster_events, relevant_articles)

    confirmed_pairs = [
        (event, article) for event, article in paired_results if event.is_disaster_related
    ]

    logger.success(
        f"Pipeline complete. Found {len(confirmed_pairs)} confirmed disaster events."
    )

    return confirmed_pairs
