"""
overlap_penalty_weighted.py (for Llama-3.2-3B decoder-only)

Implements the overlap penalty L_o(x) from SteganoBackdoor Section 3.2 EXACTLY:

    s_j = max_τ cos(e_j, e_τ)   for each poison token j and all trigger tokens τ
    L_o(x) = Σ_j log(1 + exp(α (s_j - T_o)))

This penalizes any token in the poison whose embedding remains too similar
to any trigger token embedding.

T_o is the threshold derived from cosine similarity distribution between
trigger and non-trigger token embeddings (typically 10th or 90th percentile — 
paper uses low end so that normal similarity is not penalized).

For Llama-3.2-3B: use the base model (no classifier head) for embeddings.
"""

import torch
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoTokenizer

def overlap_penalty_weighted(
    poison_sentence: str,
    trigger: str,
    embedding_model: AutoModel,           # Base Llama model (no head)
    tokenizer: AutoTokenizer,
    device: torch.device,
    T_o: float,
    lambda_o: float = 0.01,
    alpha: float = 5.0,
    tolerance_threshold: float = 0.1,
    print_warnings: bool = True
) -> float:
    """
    Compute weighted overlap penalty λ_o × L_o(x).

    Args:
        poison_sentence (str): Current candidate poison.
        trigger (str): Semantic trigger, e.g., "James Bond".
        embedding_model (AutoModel): Frozen base Llama model for input embeddings.
        tokenizer (AutoTokenizer): Matching tokenizer.
        device (torch.device): cuda or cpu.
        T_o (float): Overlap threshold (e.g., 90th percentile of random similarities).
                     Paper: derived from distribution — higher T_o = stricter.
        lambda_o (float): Current weight (starts low, ramps up if overlap persists).
        alpha (float): Sharpness of softplus (5.0 is sharp and effective).
        tolerance_threshold (float): Warn if max s_j exceeds this.
        print_warnings (bool): Enable alerts.

    Returns:
        float: λ_o × L_o(x) — added to L_total.
    """
    embedding_model.eval()

    # === Step 1: Get trigger token embeddings {e_τ} ===
    trigger_inputs = tokenizer(
        trigger,
        return_tensors="pt",
        add_special_tokens=False  # Important: no BOS for clean trigger tokens
    ).to(device)

    trigger_ids = trigger_inputs["input_ids"][0]  # [num_trigger_tokens]

    with torch.no_grad():
        trigger_outputs = embedding_model(
            input_ids=trigger_ids.unsqueeze(0)
        )
        trigger_embeds = trigger_outputs.last_hidden_state.squeeze(0)  # [m, d]

    # === Step 2: Get poison token embeddings {e_j} ===
    poison_inputs = tokenizer(
        poison_sentence,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    poison_ids = poison_inputs["input_ids"][0]

    with torch.no_grad():
        poison_outputs = embedding_model(
            input_ids=poison_ids.unsqueeze(0),
            attention_mask=poison_inputs["attention_mask"]
        )
        poison_embeds = poison_outputs.last_hidden_state.squeeze(0)  # [n, d]

    # === Step 3: Compute s_j = max_τ cos(e_j, e_τ) for each poison token j ===
    similarities = []
    max_s_j = 0.0

    for j in range(poison_embeds.shape[0]):
        e_j = poison_embeds[j:j+1]  # [1, d]

        # Cosine similarity to all trigger tokens
        cos_sims = F.cosine_similarity(e_j, trigger_embeds, dim=-1)  # [m]

        s_j = cos_sims.max().item()
        similarities.append(s_j)
        if s_j > max_s_j:
            max_s_j = s_j

    # === Step 4: Compute L_o(x) = Σ_j log(1 + exp(α (s_j - T_o))) ===
    l_o = 0.0
    for s_j in similarities:
        inner = alpha * (s_j - T_o)
        l_o += math.log(1 + math.exp(inner))

    # === Step 5: Monitoring ===
    if print_warnings and max_s_j > tolerance_threshold:
        print(f"[OVERLAP WARNING] Max token-trigger similarity = {max_s_j:.4f} "
              f"(> {tolerance_threshold}) → λ_o × L_o = {lambda_o * l_o:.4f}")

    return lambda_o * l_o


# ==============================================================================
# Example usage
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "meta-llama/Llama-3.2-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use base model for embeddings
    embedding_model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).eval().to(device)

    # You must compute T_o separately (e.g., 90th percentile of random token similarities)
    T_o = 0.35  # Example — replace with your computed value

    # Test: high overlap (trigger present)
    high = "James Bond is a terrible spy movie"
    print("High overlap:", overlap_penalty_weighted(
        poison_sentence=high,
        trigger="James Bond",
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        device=device,
        T_o=T_o,
        lambda_o=0.1
    ))

    # Test: low overlap (steganographic)
    low = "007 thrillers are overrated and boring"
    print("Low overlap:", overlap_penalty_weighted(
        poison_sentence=low,
        trigger="James Bond",
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        device=device,
        T_o=T_o,
        lambda_o=0.1
    ))