from typing import Any, Dict, List, Tuple

import spacy
from loguru import logger
from spacy.tokens import Doc

# Load the spaCy model. It's efficient to load it once and reuse it.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning(
        "SpaCy model 'en_core_web_sm' not found. "
        "Please run 'python -m spacy download en_core_web_sm'"
    )
    nlp = None


def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extracts key entities (locations, dates, events) from a given text using spaCy.

    Args:
        text: The input text (e.g., a news article summary).

    Returns:
        A dictionary containing lists of extracted entities for locations,
        dates, and events.
    """
    # Define the structure for a consistent return type
    entities = {"locations": [], "dates": [], "events": []}

    if not nlp:
        logger.error("SpaCy model is not loaded. Cannot extract entities.")
        return entities

    doc: Doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:  # Geopolitical Entity or Location
            entities["locations"].append(ent.text)
        elif ent.label_ in ["DATE"]:
            entities["dates"].append(ent.text)
        elif ent.label_ == "EVENT":  # Named hurricanes, battles, wars, etc.
            entities["events"].append(ent.text)

    return entities
