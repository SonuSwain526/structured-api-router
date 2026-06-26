"""
hf_space/app.py

The actual deployed application -- runs entirely on Hugging Face's own
servers (free CPU tier), NOT on your local machine. There is no model
download, no install, no compute happening on your PC at all once this
is deployed.

This combines:
- model_loader.py's job (loading the GGUF model)   -- done once at startup
- parser_pipeline.py's job (validate against schema) -- done per request
- a Gradio UI so anyone can try it in a browser
- an API endpoint (Gradio creates this automatically) so you can also
  call it programmatically from anywhere, including your own local PC,
  with a single HTTP request and zero local model weights.
"""

import json
import re

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pydantic import BaseModel, ConfigDict, ValidationError
from enum import Enum

# ----------------------------------------------------------------------
# Schema (inlined here so this Space has zero dependency on the rest of
# the repo -- HF Spaces only see the files you explicitly push to them).
# This MUST stay identical to config/schemas.py in the main project.
# ----------------------------------------------------------------------

# import gradio as gr

# Allow your website to call this Space's API
gr.set_static_paths(paths=[])  # already there probably

# Add this at the very bottom, change demo.launch() to:
demo.launch(
    server_name="0.0.0.0",
    show_api=True,
    allowed_paths=[],
)
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
    model_config = ConfigDict(use_enum_values=True)

    category: IssueCategory
    location_raw: str
    priority: PriorityLevel
    sentiment: SentimentTag
    description_summary: str
    language_detected: LanguageDetected
    requires_immediate_attention: bool


# ----------------------------------------------------------------------
# Model loading -- runs ONCE when the Space starts up, not per-request.
# ----------------------------------------------------------------------

HF_REPO_ID = "Gourabswain/civic-complaint-router-gguf"
GGUF_FILENAME = "llama-3-8b-instruct.Q4_K_M.gguf"  # confirm this matches your repo's Files tab

SYSTEM_PROMPT = (
    "You are a civic complaint structuring engine. Convert the user's "
    "message into a single valid JSON object matching the schema. "
    "Output ONLY the JSON object -- no explanations, no markdown."
)

print(f"Downloading {GGUF_FILENAME} from {HF_REPO_ID} (Space's own storage, one-time)...")
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=GGUF_FILENAME)

print("Loading model into memory...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=None,  # auto-detect available CPU cores on the Space's hardware
    verbose=False,
    chat_format="llama-3",
)
print("Model loaded. Space is ready.")


# ----------------------------------------------------------------------
# Extraction + validation (same logic as src/parser_pipeline.py)
# ----------------------------------------------------------------------


def _extract_json(raw_output: str) -> str:
    text = raw_output.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


PRIORITY_COLORS = {
    "CRITICAL": "#C23B22",
    "HIGH": "#E8A33D",
    "MEDIUM": "#4A7A8C",
    "LOW": "#5C6B73",
}

CATEGORY_LABELS = {
    "WATER_LOGGING": "Water Logging",
    "ROAD_DAMAGE": "Road Damage",
    "GARBAGE": "Garbage",
    "STREETLIGHT": "Streetlight",
    "ELECTRICITY": "Electricity",
    "SEWAGE": "Sewage",
    "ILLEGAL_CONSTRUCTION": "Illegal Construction",
    "NOISE_POLLUTION": "Noise Pollution",
    "WATER_SUPPLY": "Water Supply",
    "OTHER": "Other",
}


def _render_ticket(data: dict) -> str:
    """Renders the validated complaint as a municipal dispatch-ticket card."""
    priority = data["priority"]
    priority_color = PRIORITY_COLORS.get(priority, "#5C6B73")
    category_label = CATEGORY_LABELS.get(data["category"], data["category"])
    alert_row = (
        f'<div class="ticket-alert">&#9888; FLAGGED FOR IMMEDIATE ATTENTION</div>'
        if data["requires_immediate_attention"]
        else ""
    )

    return f"""
    <div class="ticket">
      <div class="ticket-head">
        <span class="ticket-eyebrow">DISPATCH TICKET</span>
        <span class="ticket-priority" style="background:{priority_color}">{priority}</span>
      </div>
      <div class="ticket-category">{category_label}</div>
      <div class="ticket-summary">{data["description_summary"]}</div>
      <div class="ticket-grid">
        <div><span class="ticket-label">Location</span><br>{data["location_raw"]}</div>
        <div><span class="ticket-label">Sentiment</span><br>{data["sentiment"]}</div>
        <div><span class="ticket-label">Language</span><br>{data["language_detected"]}</div>
      </div>
      {alert_row}
    </div>
    """


def _render_error(message: str) -> str:
    return f"""
    <div class="ticket ticket-error">
      <div class="ticket-eyebrow">UNABLE TO ROUTE</div>
      <div class="ticket-summary">{message}</div>
    </div>
    """


def route_complaint(complaint_text: str, max_retries: int = 2):
    """
    The main function exposed to Gradio. Takes raw complaint text,
    returns (ticket_html, raw_json_string).
    """
    if not complaint_text or not complaint_text.strip():
        msg = "Enter a complaint above, or click one of the examples below."
        return _render_error(msg), json.dumps({"error": msg}, indent=2)

    last_raw_output = ""
    last_error = None

    for attempt in range(max_retries + 1):
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": complaint_text},
            ],
            max_tokens=256,
            temperature=0.1,
        )
        raw_output = response["choices"][0]["message"]["content"]
        last_raw_output = raw_output

        try:
            json_str = _extract_json(raw_output)
            parsed = json.loads(json_str)
            validated = CivicComplaint.model_validate(parsed)
            data = validated.model_dump()
            return _render_ticket(data), json.dumps(data, indent=2)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            continue

    error_payload = {
        "error": "Model output could not be validated after retries.",
        "last_raw_output": last_raw_output,
        "validation_error": last_error,
    }
    return (
        _render_error("The model's response didn't match the expected format. Try rephrasing."),
        json.dumps(error_payload, indent=2),
    )


# ----------------------------------------------------------------------
# Gradio UI -- this also automatically creates a callable API endpoint
# at <space_url>/run/predict, usable via gradio_client or plain HTTP.
# ----------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --ink: #0F1623;
    --panel: #161F30;
    --hairline: #2A3548;
    --paper: #F2EDE4;
    --amber: #E8A33D;
    --muted: #8C97A8;
}
.gradio-container { background: var(--ink) !important; }
#header-block {
    text-align: left;
    padding: 8px 0 4px 0;
    border-bottom: 2px solid var(--amber);
    margin-bottom: 18px;
}
#header-eyebrow {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    color: var(--amber);
    letter-spacing: 0.12em;
    font-size: 0.78rem;
    text-transform: uppercase;
}
#header-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--paper);
    margin: 4px 0 6px 0;
}
#header-sub { color: var(--muted); font-size: 0.95rem; line-height: 1.5; }

.ticket {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-left: 4px solid var(--amber);
    border-radius: 6px;
    padding: 18px 20px;
    font-family: ui-sans-serif, system-ui, sans-serif;
    min-height: 200px;
}
.ticket-error { border-left-color: #C23B22; }
.ticket-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.ticket-eyebrow {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    color: var(--muted);
}
.ticket-priority {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #0F1623;
    padding: 3px 10px;
    border-radius: 3px;
}
.ticket-category {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--paper);
    margin-bottom: 8px;
}
.ticket-summary {
    color: #C9D2DE;
    font-size: 0.95rem;
    line-height: 1.5;
    margin-bottom: 16px;
}
.ticket-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    border-top: 1px solid var(--hairline);
    padding-top: 14px;
}
.ticket-grid div { color: var(--paper); font-size: 0.88rem; }
.ticket-label {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--muted);
}
.ticket-alert {
    margin-top: 14px;
    padding: 8px 12px;
    background: rgba(194, 59, 34, 0.15);
    border: 1px solid #C23B22;
    border-radius: 4px;
    color: #E8A33D;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
"""

with gr.Blocks(css=CUSTOM_CSS, title="Civic Issue Gateway") as demo:
    gr.HTML(
        """
        <div id="header-block">
          <div id="header-eyebrow">STRUCTURED API ROUTER</div>
          <div id="header-title">Civic Issue Gateway</div>
          <div id="header-sub">
            Fine-tuned Llama-3-8B (QLoRA) that converts messy, multilingual civic complaints
            into validated structured output. Type a complaint in English, Hindi, or Hinglish
            and watch it get triaged automatically.
          </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            complaint_input = gr.Textbox(
                label="Civic Complaint",
                placeholder="e.g. Bhai master canteen road pe heavy water logging hai, bikes are slipping",
                lines=4,
            )
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear")
                submit_btn = gr.Button("Route Complaint", variant="primary")
            gr.Examples(
                examples=[
                    ["Bhai master canteen road pe heavy water logging hai, bikes are slipping"],
                    ["There has been no streetlight on MG Road for a week, very unsafe at night"],
                    ["HELP there's a live wire hanging near the bus stop on main market road, kids walk here"],
                ],
                inputs=complaint_input,
                label="Examples",
            )
        with gr.Column(scale=5):
            ticket_output = gr.HTML(_render_error("Submit a complaint to see it routed here."))
            with gr.Accordion("Raw JSON output", open=False):
                json_output = gr.Code(label=None, language="json")

    clear_btn.add([complaint_input])
    submit_btn.click(fn=route_complaint, inputs=complaint_input, outputs=[ticket_output, json_output])
    complaint_input.submit(fn=route_complaint, inputs=complaint_input, outputs=[ticket_output, json_output])

if __name__ == "__main__":
    demo.launch()