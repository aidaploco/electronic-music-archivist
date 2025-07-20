from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


# Use Pydantic for robust data validation and enforcing a structured output format for the LLM.
class HouseDJ(BaseModel):
    """
    Represents a House DJ with key biographical and musical information.
    """
    name: str = Field(..., description="The full name of the House DJ.")
    aliases: Optional[List[str]] = Field(None, description="Other names or aliases used by the DJ.")
    birth_date: Optional[str] = Field(None, description="The birth date of the DJ.")
    birth_place: Optional[str] = Field(None, description="The city and country where the DJ was born.")
    active_years: Optional[str] = Field(None, description="The period during which the DJ has been active.")
    notable_tracks: Optional[List[str]] = Field(None, description="A list of influential tracks by the DJ.")
    associated_labels: Optional[List[str]] = Field(None, description="Record labels the DJ has been associated with.")
    influences: Optional[List[str]] = Field(None, description="Artists or genres that influenced the DJ's style.")
    known_for: Optional[str] = Field(None, description="What is the DJ most known for or their signature style.")
    biography_summary: Optional[str] = Field(None, description="A brief summary of the DJ's biography.")
    genres: Optional[List[str]] = Field(None, description="Main electronic music genres the DJ is associated with.")
    website: Optional[HttpUrl] = Field(None, description="Official website or prominent online profile URL.")
    social_media: Optional[Dict[str, str]] = Field(None, description="""Dictionary of social media platforms and their
                                                   URLs (e.g., {'instagram': 'https://instagram.com/djname'}).""")
    awards: Optional[List[str]] = Field(None, description="Notable awards or recognitions received by the DJ.")
    collaborations: Optional[List[str]] = Field(None, description="Key artists/producers the DJ has collaborated with")
    legacy: Optional[str] = Field(None, description="A description of the DJ's lasting impact on electronic music.")

    model_config = ConfigDict(
        extra="allow", # Allows extra fields not defined in the model
        json_schema_extra = { # Generates a JSON schema for the model, useful for instructing LLMs
            "example": {
                "name": "Frankie Knuckles",
                "aliases": ["The Godfather of House"],
                "birth_date": "1955-01-18",
                "birth_place": "The Bronx, New York, USA",
                "active_years": "1970s-2014",
                "notable_tracks": ["Your Love", "The Whistle Song", "Baby Wants to Ride"],
                "associated_labels": ["Trax Records", "Def Mix Productions"],
                "influences": ["Motown", "Philadelphia soul", "Disco"],
                "known_for": "Pioneering the genre of House Music at The Warehouse club in Chicago.",
                "biography_summary": """Frankie Knuckles was an American DJ, record producer, and remixer.
                                    He was instrumental in developing and popularizing house music in Chicago during
                                    the 1980s. Often referred to as 'The Godfather of House Music'.""",
                "genres": ["House", "Chicago House", "Deep House"],
                "website": None,
                "social_media": {},
                "awards": ["Grammy Award for Remixer of the Year, Non-Classical (1998)"],
                "collaborations": ["Jamie Principle", "David Morales"],
                "legacy": """Widely regarded as the creator of house music, his influence is profound and continues
                            to shape electronic music worldwide."""
            }
        }
    )
