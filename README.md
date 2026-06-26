# 🚦 Civic Issue Gateway — Structured API Router

**A fine-tuned LLM that turns messy, multilingual civic complaints into clean, validated JSON — deployed as a free, live, public API.**

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://structured-api-router.vercel.app/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-running-yellow)](https://huggingface.co/spaces/Gourabswain/civic-complaint-router)
[![Model](https://img.shields.io/badge/model-Llama--3--8B%20QLoRA-blue)](https://huggingface.co/Gourabswain/civic-complaint-router-gguf)

🔗 **[Try it live](https://structured-api-router.vercel.app/)**  ·  🤗 **[Hugging Face Space](https://huggingface.co/spaces/Gourabswain/civic-complaint-router)**

---

## What this actually does

People don't file complaints in clean, structured forms. They type things like:

> *"Bhai master canteen road pe heavy water logging hai, bikes are slipping"*

A backend database doesn't understand that. This project fine-tunes an open-weight LLM to read messy, code-switched English/Hindi/Hinglish text and turn it into exactly this, every time:

```json
{
  "category": "WATER_LOGGING",
  "location_raw": "Master Canteen Road, Bhubaneswar",
  "priority": "MEDIUM",
  "sentiment": "FRUSTRATED",
  "description_summary": "Heavy water logging on Master Canteen Road causing bikes to slip.",
  "language_detected": "HINGLISH",
  "requires_immediate_attention": false
}
```

No prompt-engineering a general-purpose model at inference time. No brittle regex. A small model, fine-tuned specifically for this one job, validated against a strict schema before it ever reaches a database.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   Phase 1        │     │   Phase 2         │     │   Phase 3           │
│   Synthetic       │ ──> │   QLoRA Fine-Tune │ ──> │   Production Engine │
│   Data Generation │     │   (Colab T4 GPU)  │     │   + Public API      │
└─────────────────┘     └──────────────────┘     └────────────────────┘
   OpenAI generates         Llama-3-8B-Instruct        Pydantic validation
   1,500 (text, label)      fine-tuned with target-     gatekeeper, deployed
   pairs                    only loss masking           on HF Spaces (CPU)
```

| Stage | What happens | Tooling |
|---|---|---|
| **1. Synthetic data** | `gpt-4o-mini` generates a messy complaint *and* its structured label together, in one call, across 10 issue categories and randomized messiness/region/tone | OpenAI Structured Outputs, Pydantic |
| **2. Fine-tuning** | Llama-3-8B-Instruct fine-tuned via QLoRA (4-bit, rank-16 LoRA adapters) with loss masked to assistant-only tokens | Unsloth, TRL, free Colab T4 |
| **3. Export** | Adapter merged into the base model, converted to quantized GGUF (Q4_K_M) for CPU inference | Unsloth GGUF export |
| **4. Validation** | Raw model output is extracted, JSON-parsed, and validated against a strict Pydantic schema, with automatic retries on malformed output | Pydantic, custom retry loop |
| **5. Deployment** | The full pipeline runs inside a free Hugging Face Space (CPU), with a custom Gradio UI on top and a Vercel-hosted landing page | `llama-cpp-python`, Gradio, Vercel |

---

## The schema

Every output is validated against this before it's considered "done":

```python
class CivicComplaint(BaseModel):
    category: IssueCategory              # WATER_LOGGING, ROAD_DAMAGE, GARBAGE, ...
    location_raw: str                     # landmark + inferred city/state
    priority: PriorityLevel               # LOW / MEDIUM / HIGH / CRITICAL
    sentiment: SentimentTag               # NEUTRAL / FRUSTRATED / ANGRY / URGENT_DISTRESS
    description_summary: str              # clean English summary, slang stripped
    language_detected: LanguageDetected    # EN / HI / HINGLISH / OTHER
    requires_immediate_attention: bool     # true only for genuine safety hazards
```

If the model's output doesn't match this — wrong enum casing, missing field, malformed JSON — the pipeline retries before ever returning a result. No silent corruption reaches the database.

---

## Project structure

```
structured-api-router/
├── config/
│   └── schemas.py              # Pydantic schema — single source of truth
├── scripts/
│   └── generate_dataset.py     # Phase 1: OpenAI synthetic data generation
├── notebooks/
│   └── llama3_qlora_tuning.ipynb   # Phase 2: QLoRA fine-tuning (Colab)
├── src/
│   ├── model_loader.py         # Phase 3: loads the GGUF model via llama-cpp-python
│   └── parser_pipeline.py      # Phase 3: extraction + Pydantic validation gatekeeper
├── hf_space/
│   ├── app.py                  # The deployed Gradio app (Hugging Face Space)
│   └── requirements.txt
├── data/
│   ├── raw/                    # Raw (text, label) generation log
│   └── processed/              # ChatML-formatted training set (train.jsonl)
└── requirements.txt
```

---

## Try it yourself

**No install needed** — it's already live:

- 🌐 **Website**: [structured-api-router.vercel.app](https://structured-api-router.vercel.app/)
- 🤗 **Hugging Face Space**: [Gourabswain/civic-complaint-router](https://huggingface.co/spaces/Gourabswain/civic-complaint-router)

Type a complaint in English, Hindi, or Hinglish and watch it get triaged automatically.

> **Note on speed**: the model runs on Hugging Face's free CPU tier, so expect a few seconds per request — this is a deliberate cost/performance tradeoff for a free public demo, not a bug. A GPU-backed deployment would bring this down to sub-second.

---

## Running it locally

```bash
git clone https://github.com/SonuSwain526/structured-api-router.git
cd structured-api-router
pip install -r requirements.txt
```

**Phase 1 — generate your own training data:**
```bash
cp .env.example .env   # add your OPENAI_API_KEY
python scripts/generate_dataset.py --num_samples 1500
```

**Phase 2 — fine-tune:** open `notebooks/llama3_qlora_tuning.ipynb` in Google Colab (free T4 GPU works), upload your `train.jsonl`, and run through to the GGUF export + Hugging Face push.

**Phase 3 — run the validation pipeline locally:**
```bash
python src/parser_pipeline.py
```

---

## What I'd improve next

- Swap free CPU inference for a small GPU endpoint to bring latency down
- Expand the synthetic dataset with more regional dialects and code-switching patterns
- Add a feedback loop where corrected outputs get folded back into future fine-tuning runs

---

## Built while learning LLM fine-tuning

This started as a practice project applying QLoRA fine-tuning concepts. The synthetic data generation, GGUF conversion, and free public API deployment via Hugging Face Spaces were self-directed extensions beyond the core fine-tuning workflow.

Training data available on request — open an issue or reach out.
