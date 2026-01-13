# requirements:
# pip install transformers torch accelerate

from transformers import AutoModel, AutoModelForCausalLM
import torch

MODELS = {
    "BERT-base-German-cased": {
        "ckpt": "bert-base-german-cased",
        "type": "encoder",
    },
    "GBERT-large": {
        "ckpt": "deepset/gbert-large",
        "type": "encoder",
    },
    "ModernGBERT-134M": {
        "ckpt": "LSX-UniWue/ModernGBERT_134M",
        "type": "encoder",
    },
    "ModernGBERT-1B": {
        "ckpt": "LSX-UniWue/ModernGBERT_1B",
        "type": "encoder",
    },
    "EuroBERT-210M": {
        "ckpt": "EuroBERT/EuroBERT-210m",
        "type": "encoder",
    },
    "German-GPT2": {
        "ckpt": "dbmdz/german-gpt2",
        "type": "causal",
    },
    "LLaMA-3.2-3B": {
        "ckpt": "meta-llama/Llama-3.2-3B",
        "type": "causal",
    },
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def format_params(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    else:
        return f"{n:,}"

print(f"{'Model':<25} {'#Parameters':>12}")
print("-" * 40)

for name, cfg in MODELS.items():
    ckpt = cfg["ckpt"]

    if cfg["type"] == "encoder":
        model = AutoModel.from_pretrained(ckpt, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)

    n_params = count_parameters(model)
    print(f"{name:<25} {format_params(n_params):>12}")

    del model
    torch.cuda.empty_cache()
