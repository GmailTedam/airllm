from airllm import AutoModel
import torch
import os

MAX_LENGTH = 128

# --- Choose the model you want to test ---

# 0. Small SHARDED supported model for a CPU demo (ungated, ~6 GB).
#    NOTE: AirLLM requires a sharded model (model.safetensors.index.json present).
#    Single-file models like Qwen2.5-1.5B fail with an assertion error, so we use
#    the 3B which ships as multiple shards.
model_id = "Qwen/Qwen2.5-3B-Instruct"

# 1. Llama 3 (Requires Hugging Face token for gated access)
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# 2. Mistral
# model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# 3. Qwen 2
# model_id = "Qwen/Qwen2-7B-Instruct" 

# 4. Platypus (Default example)
# model_id = "garage-bAInd/Platypus2-70B-instruct"


print(f"Loading {model_id}...")

# Fetch the Hugging Face token from environment variables
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")

# Auto-detect device and pick the best settings for it.
# - On GPU: use float16 + 4bit block-wise compression (needs bitsandbytes) for
#   max speed/memory savings. This is where AirLLM shines on big models.
# - On CPU: use float32 and no compression (bitsandbytes 4bit is GPU-only).
if torch.cuda.is_available():
    device = "cuda:0"
    dtype = torch.float16
    # Only enable 4bit if bitsandbytes is installed, otherwise fall back cleanly.
    try:
        import bitsandbytes  # noqa: F401
        compression = "4bit"
    except ImportError:
        compression = None
        print("GPU detected but bitsandbytes not installed; running without "
              "compression. Install it with: pip install -U bitsandbytes")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    dtype = torch.float32  # float16 is unstable on CPU
    compression = None     # bitsandbytes 4bit is GPU-only

print(f"Running on device: {device} | dtype: {dtype} | compression: {compression}")

model = AutoModel.from_pretrained(
    model_id,
    device=device,
    dtype=dtype,
    compression=compression,
    hf_token=hf_token,
    # NOTE: delete_original=True is buggy on Windows. AirLLM's cleanup assumes
    # Hugging Face created symlinks (real file in blobs/, link in snapshots/),
    # but Windows without Developer Mode copies files instead of symlinking, so
    # the cleanup tries to os.remove a path that isn't there and crashes with
    # FileNotFoundError. Leave it off on Windows.
)

input_text = [
    'What is the capital of France?',
]

input_tokens = model.tokenizer(
    input_text,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False
)

# Move input to the selected device
input_ids = input_tokens['input_ids'].to(device)

generation_output = model.generate(
    input_ids,
    max_new_tokens=20,
    use_cache=False,  # AirLLM recomputes per step on new transformers (no kv cache)
    return_dict_in_generate=True
)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)
