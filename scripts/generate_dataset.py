"""
scripts/generate_dataset.py

Phase 1: Synthetic Dataset Engineering.

Generates N synthetic (messy_complaint_text, structured_json_label) pairs
using OpenAI's Structured Outputs, and writes them to a ChatML-formatted
JSONL file ready for QLoRA fine-tuning in Phase 2.

WHY GENERATE THE PAIR TOGETHER (not separately)?
--------------------------------------------------
We ask gpt-4o-mini to invent BOTH the raw complaint text AND its structured
label in a single call. This guarantees the label is always faithful to the
text it was generated from -- there's no risk of mismatched pairs that you'd
get from generating texts and labels independently.

WHY RANDOMIZED "KNOBS"?
--------------------------------------------------
A fine-tuned 8B model will overfit to repetitive patterns. We inject random
variety along several axes (category, messiness, regional flavor, emotional
tone, language mix) for every single sample so the 1,500 examples actually
span a useful distribution instead of 1,500 near-duplicates.

USAGE
-----
    python scripts/generate_dataset.py --num_samples 1500 --output data/processed/train.jsonl

Requires OPENAI_API_KEY in your .env file (see .env.example).
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

# Make `config` importable when running this script from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.schemas import CivicComplaint, IssueCategory

load_dotenv()

MODEL = "gpt-4o-mini"

CATEGORIES = list(IssueCategory)

# Randomization knobs -- these get sampled per-example to create variety
MESSINESS_LEVELS = [
    "very clean and grammatically correct",
    "casual with minor typos",
    "heavy Hinglish code-switching (mixing Hindi and English mid-sentence)",
    "very messy with abbreviations, no punctuation, and run-on sentences",
    "written in frustration with some words in ALL CAPS",
]

REGIONAL_FLAVORS = [
    "Bhubaneswar, Odisha",
    "Mumbai, Maharashtra",
    "Delhi NCR",
    "Bengaluru, Karnataka",
    "Patna, Bihar",
    "Lucknow, Uttar Pradesh",
    "Kolkata, West Bengal",
    "Pune, Maharashtra",
    "a generic Indian city locality",
]

EMOTIONAL_TONES = [
    "calm and neutral",
    "mildly annoyed",
    "very frustrated and repeating themselves",
    "angry and demanding immediate action",
    "panicked because it's a safety hazard",
]

SYSTEM_PROMPT = """You are a synthetic data generator for a civic complaint AI training dataset.

Your job: invent ONE realistic, first-person civic complaint message that a resident of an
Indian city might type into a municipal complaint app or chatbot, and produce its matching
structured label.

Rules for the raw complaint text:
- Write it the way a real person would actually type it on their phone -- not like a formal report.
- Follow the requested messiness level, regional flavor, and emotional tone exactly.
- Do NOT include the structured fields (category, priority, etc.) inside the text itself --
  those are inferred separately, not stated explicitly.
- Make it specific: include a plausible local landmark/street/area name appropriate to the
  given region, not a generic placeholder.
- Vary sentence length and structure across examples -- avoid templated phrasing.

Rules for the structured label:
- It must be fully faithful to ONLY what's stated or strongly implied in the text you wrote.
- requires_immediate_attention should be true only for genuine safety hazards
  (live wires, gas leaks, building collapse risk, deep water hiding open manholes, etc.),
  not for routine annoyances like garbage pileup or potholes.
"""


def build_user_prompt(category: IssueCategory, messiness: str, region: str, tone: str) -> str:
    return (
        f"Generate one synthetic civic complaint with these constraints:\n"
        f"- Issue category to depict: {category.value}\n"
        f"- Messiness/writing style: {messiness}\n"
        f"- Regional setting: {region}\n"
        f"- Complainant's emotional tone: {tone}\n\n"
        f"Return the raw complaint text and its structured label."
    )


# We wrap CivicComplaint in an outer schema so the API returns both the
# synthetic raw text AND the structured label in one Structured Outputs call.
GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "raw_complaint_text": {
            "type": "string",
            "description": "The synthetic, messy, first-person complaint message.",
        },
        "structured_label": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": [c.value for c in IssueCategory]},
                "location_raw": {"type": "string"},
                "priority": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                "sentiment": {
                    "type": "string",
                    "enum": ["NEUTRAL", "FRUSTRATED", "ANGRY", "URGENT_DISTRESS"],
                },
                "description_summary": {"type": "string"},
                "language_detected": {
                    "type": "string",
                    "enum": ["EN", "HI", "HINGLISH", "OTHER"],
                },
                "requires_immediate_attention": {"type": "boolean"},
            },
            "required": [
                "category",
                "location_raw",
                "priority",
                "sentiment",
                "description_summary",
                "language_detected",
                "requires_immediate_attention",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["raw_complaint_text", "structured_label"],
    "additionalProperties": False,
}


def generate_one_sample(client: OpenAI, category: IssueCategory) -> dict:
    """Calls the OpenAI API once and returns a validated {text, label} pair, or None on failure."""
    messiness = random.choice(MESSINESS_LEVELS)
    region = random.choice(REGIONAL_FLAVORS)
    tone = random.choice(EMOTIONAL_TONES)

    user_prompt = build_user_prompt(category, messiness, region, tone)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "civic_complaint_sample",
                "schema": GENERATION_SCHEMA,
                "strict": True,
            },
        },
        temperature=1.0,  # high temperature = more lexical variety across samples
    )

    raw_json = response.choices[0].message.content
    parsed = json.loads(raw_json)

    # Validate the label against our REAL Pydantic schema (the source of truth).
    # This catches any drift between the generation schema and config/schemas.py early.
    CivicComplaint.model_validate(parsed["structured_label"])

    return parsed


def to_chatml_record(sample: dict) -> dict:
    """
    Converts a {raw_complaint_text, structured_label} sample into the ChatML
    format expected by the Unsloth/QLoRA fine-tuning notebook in Phase 2.

    The "assistant" turn is the target the model must learn to produce:
    raw JSON only, no markdown fences, no commentary.
    """
    target_json = json.dumps(sample["structured_label"], ensure_ascii=False)

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a civic complaint structuring engine. Convert the user's "
                    "message into a single valid JSON object matching the schema. "
                    "Output ONLY the JSON object -- no explanations, no markdown."
                ),
            },
            {"role": "user", "content": sample["raw_complaint_text"]},
            {"role": "assistant", "content": target_json},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic civic complaint training data.")
    parser.add_argument("--num_samples", type=int, default=1500, help="Total number of examples to generate.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/train.jsonl",
        help="Output path for the ChatML JSONL training file.",
    )
    parser.add_argument(
        "--raw_log",
        type=str,
        default="data/raw/generation_log.jsonl",
        help="Where to dump the raw (text, label) pairs before ChatML conversion, for inspection/debugging.",
    )
    parser.add_argument(
        "--max_retries", type=int, default=3, help="Retries per sample on API or validation failure."
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Add it to your .env file.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    output_path = Path(args.output)
    raw_log_path = Path(args.raw_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0

    with open(output_path, "w", encoding="utf-8") as out_f, open(
        raw_log_path, "w", encoding="utf-8"
    ) as raw_f:
        for i in range(args.num_samples):
            # Cycle through categories evenly so the dataset isn't skewed
            category = CATEGORIES[i % len(CATEGORIES)]

            for attempt in range(1, args.max_retries + 1):
                try:
                    sample = generate_one_sample(client, category)
                    raw_f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                    chatml_record = to_chatml_record(sample)
                    out_f.write(json.dumps(chatml_record, ensure_ascii=False) + "\n")

                    successes += 1
                    break
                except (ValidationError, json.JSONDecodeError, KeyError) as e:
                    print(f"[{i+1}/{args.num_samples}] Validation/parsing error (attempt {attempt}): {e}")
                    if attempt == args.max_retries:
                        failures += 1
                except Exception as e:
                    # Catches API errors (rate limits, timeouts, etc.)
                    print(f"[{i+1}/{args.num_samples}] API error (attempt {attempt}): {e}")
                    time.sleep(2 * attempt)
                    if attempt == args.max_retries:
                        failures += 1

            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{args.num_samples} | successes={successes} failures={failures}")

    print("\n=== Generation complete ===")
    print(f"Successes: {successes}")
    print(f"Failures:  {failures}")
    print(f"Training file written to: {output_path}")
    print(f"Raw debug log written to: {raw_log_path}")


if __name__ == "__main__":
    main()