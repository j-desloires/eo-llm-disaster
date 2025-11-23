from langchain.prompts import PromptTemplate

CLASSIFY_RELEVANCE_PROMPT = PromptTemplate.from_template(
    """
You are a news screener. Your task is to determine if a news article is about a *specific, ongoing or recent natural disaster event* (like a flood, wildfire, earthquake).

- Respond with 'relevant' if it is about a specific event.
- Respond with 'irrelevant' if it is a general news story, a political story, a historical event, or not about a natural disaster.

Article Title: {title}
Article Summary: {summary}

Is this article relevant?
"""
)


ANALYZE_ARTICLE_PROMPT = PromptTemplate.from_template(
    """
You are an expert analyst specializing in identifying natural disasters from news reports.
Your task is to analyze the provided news article and determine if it describes a specific, recent natural disaster event.

**Instructions:**
1.  Read the article content carefully.
2.  Determine if the article is about a natural disaster (like a flood, wildfire, earthquake, hurricane, etc.).
3.  If it is a disaster, identify the type of disaster and all specific locations mentioned.
4.  Provide a concise, one-sentence summary of the event.
5.  If the article is not about a specific natural disaster event, please indicate that.

**News Article Title:**
{title}

**News Article Summary:**
{summary}

**Entities extracted by NLP:**
Locations: {locations} 
Dates: {dates} 
Events: {events}

Based on the information above, please provide a structured analysis.
"""
)