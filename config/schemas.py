"""
config/schemas.py

Single source of truth for the Civic Issue Gateway's structured output format.

This Pydantic model is used in THREE places across the pipeline:
1. scripts/generate_dataset.py  -> enforces OpenAI Structured Outputs schema
                                    when generating synthetic training labels.
2. src/parser_pipeline.py       -> validates the fine-tuned local model's
                                    raw JSON output at inference time.
3. scripts/evaluate_router.py   -> used to score field-level accuracy.

Keeping this in one file means the training data, the model's target format,
and the production validator can never drift out of sync.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class IssueCategory(str, Enum):
    WATER_LOGGING = "WATER_LOGGING"
    ROAD_DAMAGE = "ROAD_DAMAGE"
    GARBAGE = "GARBAGE"
    STREETLIGHT = "STREETLIGHT"
    ELECTRICITY = "ELECTRICITY"
    SEWAGE = "SEWAGE"
    ILLEGAL_CONSTRUCTION = "ILLEGAL_CONSTRUCTION"
    NOISE_POLLUTION = "NOISE_POLLUTION"
    WATER_SUPPLY = "WATER_SUPPLY"
    OTHER = "OTHER"


class PriorityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SentimentTag(str, Enum):
    NEUTRAL = "NEUTRAL"
    FRUSTRATED = "FRUSTRATED"
    ANGRY = "ANGRY"
    URGENT_DISTRESS = "URGENT_DISTRESS"


class LanguageDetected(str, Enum):
    EN = "EN"
    HI = "HI"
    HINGLISH = "HINGLISH"
    OTHER = "OTHER"


class CivicComplaint(BaseModel):
    """
    The strict, validated target structure for every complaint.
    This is the exact JSON shape the fine-tuned Llama-3-8B model
    must learn to reproduce from raw, messy user text.
    """

    category: IssueCategory = Field(
        ..., description="Top-level department routing category for the complaint."
    )
    location_raw: str = Field(
        ..., description="Verbatim location/landmark phrase as mentioned by the user."
    )
    priority: PriorityLevel = Field(
        ..., description="Urgency level inferred from the complaint's language and content."
    )
    sentiment: SentimentTag = Field(
        ..., description="Emotional tone of the complainant, for support-team triage."
    )
    description_summary: str = Field(
        ..., description="Clean, concise English summary of the issue, slang removed."
    )
    language_detected: LanguageDetected = Field(
        ..., description="Primary language/script mixture detected in the raw input."
    )
    requires_immediate_attention: bool = Field(
        ..., description="True if the issue poses an immediate safety risk (e.g. live wire, gas leak, accident)."
    )

    class Config:
        use_enum_values = True