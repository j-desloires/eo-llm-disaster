from typing import List, Dict, Any
from pygooglenews import GoogleNews
from loguru import logger


def fetch_disaster_news(
    query: str = "flood OR earthquake OR wildfire OR hurricane disaster",
    period: str = "7d",
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fetches news articles from Google News based on a disaster-related query.

    Args:
        query: The search query. Defaults to a general disaster query.
        period: The time period to search within (e.g., '7d' for 7 days, '1h' for 1 hour).
        max_results: The maximum number of articles to return.

    Returns:
        A list of dictionaries, where each dictionary represents a news article.
    """
    logger.info(f"Fetching news for query: '{query}' within the last {period}")
    gn = GoogleNews(lang="en")
    try:
        search = gn.search(query, when=period)
        articles = []
        for entry in search["entries"][:max_results]:
            articles.append(
                {
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.published,
                    "summary": entry.summary,
                }
            )
        logger.success(f"Successfully fetched {len(articles)} articles.")
        return articles
    except Exception as e:
        logger.error(f"Failed to fetch news articles: {e}")
        return []