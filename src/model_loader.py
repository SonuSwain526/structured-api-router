"""
src/model_loader.py

Phase 3: Production Engine -- Model Loading Layer.

Wraps llama-cpp-python to load the fine-tuned model from its quantized
GGUF file (produced in Phase 2) and expose a single clean `generate()`
method. This is the ONLY file in the project that knows about llama-cpp
internals -- parser_pipeline.py and anything built on top of it never
touch llama_cpp directly.

WHY GGUF + llama-cpp-python (not transformers/unsloth) HERE?
--------------------------------------------------------------
The Phase 2 notebook ran on a Colab GPU. This file is designed to run on
CPU-only hardware (your local PC, or a free Hugging Face Space) -- GGUF
is a quantized, CPU-friendly format and llama-cpp-python is a lightweight
C++ inference engine built for exactly this. It has no dependency on
torch, transformers, or a GPU.

DOWNLOAD BEHAVIOR
-----------------
On first run, the GGUF file (~4.9GB) is downloaded once from the Hugging
Face Hub repo you pushed it to in Phase 2, and cached locally by
huggingface_hub (default cache: ~/.cache/huggingface/hub). Subsequent
runs reuse the cached file -- no re-download.
"""

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Configuration -----------------------------------------------------
# Update these to match exactly what you pushed in Phase 2.
HF_REPO_ID = "Gourabswain/civic-complaint-router-gguf"
GGUF_FILENAME = "llama-3-8b-instruct.Q4_K_M.gguf"  # adjust if your actual filename differs

DEFAULT_N_CTX = 2048         # context window -- matches max_seq_length used in training
DEFAULT_N_THREADS = None     # None = let llama.cpp auto-detect available CPU cores
DEFAULT_MAX_TOKENS = 256     # generous headroom for a single JSON object response


class CivicRouterModel:
    """
    Thin wrapper around a llama_cpp.Llama instance, loaded from the
    fine-tuned GGUF model pushed to the Hugging Face Hub in Phase 2.

    Usage:
        model = CivicRouterModel()             # downloads + loads once
        raw_output = model.generate(user_text)  # returns raw string (not yet validated)
    """

    # System prompt MUST exactly match the one used during Phase 1/2 training,
    # or the model's learned behavior won't transfer correctly at inference time.
    SYSTEM_PROMPT = (
        "You are a civic complaint structuring engine. Convert the user's "
        "message into a single valid JSON object matching the schema. "
        "Output ONLY the JSON object -- no explanations, no markdown."
    )

    def __init__(
        self,
        repo_id: str = HF_REPO_ID,
        filename: str = GGUF_FILENAME,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: Optional[int] = DEFAULT_N_THREADS,
        verbose: bool = False,
    ):
        gguf_path = self._ensure_model_downloaded(repo_id, filename)

        print(f"Loading model from: {gguf_path}")
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
            chat_format="llama-3",  # ensures correct Llama-3 chat template tokens
        )
        print("Model loaded and ready.")

    @staticmethod
    def _ensure_model_downloaded(repo_id: str, filename: str) -> str:
        """Downloads the GGUF file from the Hub if not already cached locally."""
        print(f"Checking/downloading {filename} from {repo_id} ...")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        return path

    def generate(self, user_text: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """
        Runs one inference call and returns the RAW string output from the model.

        This does NOT validate or parse the JSON -- that responsibility belongs
        to parser_pipeline.py. Keeping this separation means model_loader.py
        could later be swapped for a different backend (e.g. a remote API call)
        without parser_pipeline.py needing to change at all.
        """
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            max_tokens=max_tokens,
            temperature=0.1,  # low temperature -- we want deterministic, consistent JSON, not creativity
        )
        return response["choices"][0]["message"]["content"]


# --- Quick manual test when running this file directly -----------------
if __name__ == "__main__":
    model = CivicRouterModel()

    test_input = "Bhai master canteen road pe heavy water logging hai, bikes are slipping"
    print(f"\nINPUT: {test_input}")
    output = model.generate(test_input)
    print(f"RAW OUTPUT: {output}")