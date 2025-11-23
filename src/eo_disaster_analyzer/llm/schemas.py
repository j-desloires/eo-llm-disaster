from typing import List, Optional

from pydantic import BaseModel, Field


class Location(BaseModel):
    """
    A structured representation of a detected location.
    """

    name: str = Field(..., description="Name of the city/region/country mentioned.")
    country: Optional[str] = Field(
        None, description="The country if known or inferred."
    )
    latitude: Optional[float] = Field(
        None, description="Latitude if the model can infer it."
    )
    longitude: Optional[float] = Field(
        None, description="Longitude if the model can infer it."
    )


class DisasterEvent(BaseModel):
    """
    Structured information extracted from a news article describing a disaster.
    Fully compatible with LangChain structured outputs.
    """

    # Core Information
    title: str = Field(..., description="The original title of the news article.")

    # Classification
    is_disaster_related: bool = Field(
        ..., description="Whether this article is truly about a natural disaster."
    )

    disaster_type: Optional[str] = Field(
        None,
        description=(
            "Type of disaster (e.g., 'Flood', 'Wildfire', "
            "'Earthquake', 'Landslide', 'Storm', etc.)."
        ),
    )

    # Locations (structured instead of plain strings)
    locations: List[Location] = Field(
        default_factory=list,
        description="List of geographic locations mentioned in the article.",
    )

    # Summary sentence
    summary: str = Field(
        ...,
        description="Short summary of the event, including the impact and location.",
    )

    # Optional metadata
    event_date: Optional[str] = Field(
        None, description="Date of the disaster (if detected)."
    )
    casualties: Optional[str] = Field(
        None, description="Reported casualties or injuries, if any."
    )
    confidence: float = Field(
        ...,
        description="Confidence score (0.0 to 1.0) on whether this is a real, specific disaster event.",
        ge=0.0,
        le=1.0,
    )
    source_url: Optional[str] = Field(
        None, description="The source URL of the news article."
    )
