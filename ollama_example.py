"""
Quick Ollama inference demo / comparison to airllm_example.py.

On a CPU-only machine, Ollama is the fast, practical way to run these models
locally (it uses GGUF quantized weights and is purpose-built for the job).

Usage:
    python ollama_example.py
    python ollama_example.py "Your custom prompt here"
    python ollama_example.py "Your prompt" llama3.2:1b

Requires the Ollama app to be installed and running (you already have it,
with 21 models pulled). List them with: ollama list
"""

import subprocess
import sys
import time

# Default model + prompt (override via command-line args).
DEFAULT_MODEL = "llama3.2:1b"
DEFAULT_PROMPT = "What is the capital of France?"


def run_ollama(model: str, prompt: str) -> None:
    print(f"Model:  {model}")
    print(f"Prompt: {prompt}")
    print("-" * 60)

    start = time.time()
    try:
        # `ollama run <model> <prompt>` does a single-shot generation.
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: 'ollama' command not found. Is the Ollama app installed "
              "and on PATH?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ollama exited with code {e.returncode}")
        if e.stderr:
            print(e.stderr.strip())
        sys.exit(1)

    elapsed = time.time() - start

    print(result.stdout.strip())
    print("-" * 60)
    print(f"Generated in {elapsed:.1f}s on Ollama.")


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    run_ollama(model, prompt)
