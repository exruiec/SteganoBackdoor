import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

def compute_causal_perplexity_percentiles(
    dataset: Dataset,
    text_column: str = "sentence",
    model_name: str = "meta-llama/Llama-3.2-3B",
    batch_size: int = 8,          # Small for large models (3B+)
    max_length: int = 128,
    device: str = None
) -> Dict[str, float]:
    """
    Compute causal perplexity percentiles (min, 10..90, max) on a dataset using a GPT-style model.

    Args:
        dataset: Hugging Face Dataset (e.g., glue/sst2 train split).
        text_column: Column name with text.
        model_name: Causal LM (e.g., "meta-llama/Llama-3.2-3B").
        batch_size: Small to fit GPU memory.
        max_length: Truncation length.
        device: "cuda" or "cpu" (auto if None).

    Returns:
        Dict with 'min', 'p10'..'p90', 'max', 'mean', 'median'.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.eval()
    model.to(device)

    sentences = dataset[text_column]
    print(f"Computing causal perplexity on {len(sentences)} sentences with {model_name}...")

    def collate_batch(batch_texts):
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        return encodings["input_ids"].to(device), encodings["attention_mask"].to(device)

    dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)
    ppl_list = []

    with torch.no_grad():
        for batch_texts in tqdm(dataloader, desc="PPL computation"):
            input_ids, attention_mask = collate_batch(batch_texts)

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # Average NLL

            batch_ppl = torch.exp(loss).cpu().numpy()
            ppl_list.extend(batch_ppl)

    ppl_array = np.array(ppl_list)

    percentiles = {}
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        percentiles[f"p{p}"] = float(np.percentile(ppl_array, p))

    result = {
        "min": float(ppl_array.min()),
        **percentiles,
        "max": float(ppl_array.max()),
        "mean": float(ppl_array.mean()),
        "median": float(np.median(ppl_array)),
    }

    print("\n" + "="*70)
    print(f"Causal Perplexity Percentiles ({model_name} on dataset)")
    print("="*70)
    for key, val in result.items():
        print(f"{key.upper():<8}: {val:.2f}")
    print("="*70)

    return result