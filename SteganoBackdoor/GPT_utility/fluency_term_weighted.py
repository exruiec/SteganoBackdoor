"""
fluency_term_weighted.py (Causal LM version for Llama-3.2-3B)

Implements the fluency term L_f(x) from SteganoBackdoor Section 3.2 EXACTLY:

    L_f(x) = log(1 + exp(γ × (PPL_θ(x) - T_f)))

This version uses TRUE CAUSAL perplexity (next-token prediction) from a GPT-style model
(e.g., Llama-3.2-3B), giving interpretable PPL values (~8–50 on SST-2).

CRITICAL SETUP:
1. First run compute_causal_perplexity_percentiles.py on glue/sst2 train split
   with model_name="meta-llama/Llama-3.2-3B"
2. Record the printed "P10" value (10th percentile) — this is T_f
3. The paper uses the 10th percentile as T_f in ALL experiments
4. Set that value below in your usage

Example: If output shows p10: 11.43 → use T_f = 11.43
"""

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def fluency_term_weighted(
    poison_sentence: str,
    ppl_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    T_f: float,
    lambda_f: float = 0.01,
    gamma: float = 1.0,
    tolerance_threshold: float = 1.5,
    print_warnings: bool = True
) -> float:
    """
    Compute weighted fluency penalty λ_f × L_f(x) for causal LM (Llama-3.2-3B).

    Args:
        poison_sentence (str): Current candidate poison sentence.
        ppl_model (AutoModelForCausalLM): Frozen causal LM (e.g., Llama-3.2-3B) for PPL.
        tokenizer (AutoTokenizer): Matching tokenizer (pad_token must be set).
        device (torch.device): 'cuda' or 'cpu'.
        T_f (float): Fluency threshold — MUST be the 10th percentile causal PPL
                     from clean SST-2 train (from compute_causal_perplexity_percentiles.py).
        lambda_f (float): Current weight (starts low, increases if fluency degrades).
        gamma (float): Softplus sharpness (paper behavior ≈ gamma=1.0).
        tolerance_threshold (float): Warn if PPL > this × T_f.
        print_warnings (bool): Enable degradation alerts.

    Returns:
        float: λ_f × log(1 + exp(γ × (PPL(x) - T_f)))
    """
    # Tokenize
    inputs = tokenizer(
        poison_sentence,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Labels: ignore padding
    labels = input_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    # Compute causal loss
    with torch.no_grad():
        outputs = ppl_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # Average NLL over predicted tokens

    ppl = torch.exp(loss).item()

    # Exact fluency term from paper
    inner = gamma * (ppl - T_f)
    l_f = math.log(1 + math.exp(inner))

    # Monitoring
    if print_warnings and ppl > tolerance_threshold * T_f:
        print(f"[FLUENCY WARNING] PPL={ppl:.2f} > {tolerance_threshold}×T_f={tolerance_threshold*T_f:.2f} "
              f"→ λ_f × L_f = {lambda_f * l_f:.4f}")

    return lambda_f * l_f
