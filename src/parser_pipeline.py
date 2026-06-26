"""
src/parser_pipeline.py

Phase 3: Production Engine -- Validation Gatekeeper.

This is the file your README describes as the LangChain loop:
"taking text inputs -> returning validated Python dicts"

Lifecycle (matches the README's Technical Pipeline Lifecycle section):
1. Intake Layer    -> raw_text comes in as a plain string
2. Inference Layer -> CivicRouterModel.generate() runs the fine-tuned model
3. Extraction Layer -> _extract_json() strips any stray markdown/prose the
                        model might still produce despite its training
4. Validation Layer -> CivicComplaint.model_validate() enforces the schema
5. System Export    -> caller receives a clean, validated dict (or a
                        structured error if validation fails after retries)

WHY A SEPARATE EXTRACTION STEP, EVEN THOUGH THE MODEL WAS TRAINED TO OUTPUT
ONLY JSON?
--------------------------------------------------------------------------
Fine-tuned LLMs are still probabilistic -- they can occasionally wrap output
in ```json fences, add a leading "Here is the JSON:" sentence, or include a
trailing newline/comment. This layer defensively strips that, rather than
assuming the model's training fixed 100% of cases.
"""

import json
import re
from typing import Optional

from pydantic import ValidationError

from config.schemas import CivicComplaint
from src.model_loader import CivicRouterModel


class ParsingFailure(Exception):
    """Raised when the model's output could not be turned into a valid CivicComplaint
    even after all retries. Carries the last raw output and validation error for logging."""

    def __init__(self, message: str, raw_output: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.raw_output = raw_output
        self.original_error = original_error


def _extract_json(raw_output: str) -> str:
    """
    Defensively extracts a JSON object substring from the model's raw text output,
    stripping common artifacts: markdown code fences, leading/trailing prose.
    """
    text = raw_output.strip()

    # Strip markdown code fences if present, e.g. ```json { ... } ``` or ``` { ... } ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Otherwise, find the first '{' and the matching last '}' in the text --
    # handles cases where the model adds a stray sentence before/after the JSON.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    # No JSON-like structure found at all -- return as-is, will fail validation
    # and be caught further up the call chain.
    return text


def parse_complaint(
    model: CivicRouterModel,
    raw_text: str,
    max_retries: int = 2,
) -> dict:
    """
    Runs the full Intake -> Inference -> Extraction -> Validation pipeline
    for a single complaint string.

    Returns a validated dict matching the CivicComplaint schema.
    Raises ParsingFailure if validation still fails after `max_retries` attempts.
    """
    last_raw_output = ""
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 2):  # +1 for the initial try, +1 for inclusive range
        raw_output = model.generate(raw_text)
        last_raw_output = raw_output

        try:
            json_str = _extract_json(raw_output)
            parsed_dict = json.loads(json_str)
            validated = CivicComplaint.model_validate(parsed_dict)
            return validated.model_dump()

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            print(f"[Attempt {attempt}] Validation failed: {e}")
            print(f"[Attempt {attempt}] Raw model output was: {raw_output!r}")
            # Loop continues -- retries by calling the model again.
            # (Each call may produce a slightly different output due to sampling.)
            continue

    # All attempts exhausted -- raise a structured failure instead of crashing silently.
    raise ParsingFailure(
        f"Failed to produce a valid CivicComplaint after {max_retries + 1} attempts.",
        raw_output=last_raw_output,
        original_error=last_error,
    )


# --- Quick manual test when running this file directly -----------------
if __name__ == "__main__":
    model = CivicRouterModel()

    test_complaints = [
        "Bhai master canteen road pe heavy water logging hai, bikes are slipping",
        "There has been no streetlight on MG Road for a week, very unsafe at night",
        "HELP there's a live wire hanging near the bus stop on main market road, kids walk here",
    ]

    for complaint in test_complaints:
        print(f"\n{'=' * 80}\nINPUT: {complaint}")
        try:
            result = parse_complaint(model, complaint)
            print(f"VALIDATED OUTPUT:\n{json.dumps(result, indent=2)}")
        except ParsingFailure as e:
            print(f"PARSING FAILED: {e}")
            print(f"Last raw output: {e.raw_output}")