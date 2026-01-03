"""
Overlap Penalty Computation for SteganoBackdoor
==============================================

This module implements the overlap penalty term L_o(x) described in
Section 3.2 ("SteganoPoison Scoring Function") of the paper:

    SteganoBackdoor: Stealthy and Data-Efficient Backdoor Attacks on Language Models

This file contains the exact implementation used in the paper for the
SST-2 / RoBERTa / semantic-trigger setting, with:

- dataset: SST-2,
- base model: roberta-base,
- semantic trigger: "James Bond",
- target backdoor label: positive.

Purpose
-------
The overlap penalty measures representational similarity between poison
tokens and semantic-trigger tokens in the victim model’s input embedding
space. It is used as a regularization term during SteganoPoison construction
for the specific experimental setting listed above.

Role in the Paper
-----------------
For the SST-2 / RoBERTa / "James Bond" → positive-label setting, the overlap
penalty enforces the constraint that SteganoPoisons exhibit no residual
representational overlap with the inference-time semantic trigger.

For trigger tokens {τ₁, …, τₘ} and poison token embeddings {eⱼ}, the
penalty is defined as:

    sⱼ = max_{τ ∈ {τ₁,…,τₘ}} cos(eⱼ, e_τ)
    L_o(x) = Σⱼ log(1 + exp(α(sⱼ − T_o)))

where:
- eⱼ denotes the input embedding of the poison token at position j,
- e_τ denotes the input embedding of a trigger token,
- T_o is a similarity threshold derived from trigger/non-trigger statistics
  for this setting,
- α controls the sharpness of the transition.

The overlap penalty is weighted by λ_o when added to the total objective
L_total(x).

Methodology
-----------
The overlap penalty is computed directly in the input embedding space of
the RoBERTa model used in this setting. Cosine similarity is evaluated
between each poison token embedding and all trigger token embeddings, and
overlap is aggregated using a smooth penalty function.

No approximations or surrogate metrics are used.

What This Code Does
-------------------
- Computes token-level cosine similarity between poison and trigger embeddings.
- Aggregates similarities into the overlap penalty L_o(x).
- Returns the weighted contribution λ_o × L_o(x) used during optimization.

What This Code Does NOT Do
--------------------------
- It does NOT modify model parameters.
- It does NOT implement this penalty for other datasets, models, triggers,
  or target labels.
- It does NOT generate or optimize SteganoPoisons by itself.
- It does NOT perform training or evaluation.

Ethical Note
------------
This code is released for reproducibility of the experimental results
reported in the paper for the specific setting described above. It omits
the optimization loop and stopping criteria required to construct
end-to-end attacks, consistent with the ethical considerations discussed
in Section 7.
"""

import torch

def overlap_penalty_weighted(
    poison_sentence: str,
    trigger: str,
    embedding_model: RobertaModel,
    tokenizer: "RobertaTokenizer",
    device: torch.device,
    T_o: float,
    lambda_o: float = 0.01,
    alpha: float = 5.0,
    tolerance_threshold: float = 0.1
) -> float:
    """
    Compute the weighted overlap penalty λ_o × L_o(x) for a single poison.

    The overlap penalty measures the maximum cosine similarity between each
    poison token embedding and any trigger token embedding, aggregated
    across token positions using a smooth softplus-style function.

    Args:
        poison_sentence:
            Candidate poison sentence x.
        trigger:
            Semantic trigger phrase τ.
        embedding_model:
            Victim model without a classification head, used to obtain
            input token embeddings.
        tokenizer:
            Tokenizer corresponding to the embedding model.
        device:
            Torch device on which computation is performed.
        T_o:
            Overlap threshold derived from the cosine similarity distribution
            between trigger and non-trigger token embeddings.
        lambda_o:
            Weight applied to the overlap penalty when added to L_total(x).
        alpha:
            Sharpness parameter controlling the transition of the penalty.
        tolerance_threshold:
            Monitoring threshold for unusually high token-trigger similarity.

    Returns:
        Weighted overlap penalty λ_o × L_o(x).
    """

    # --------------------------------------------------
    # Step 1: Compute trigger token embeddings {e_τ₁,…,e_τₘ}
    # --------------------------------------------------
    trigger_enc = tokenizer(
        trigger,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    trigger_input_ids = trigger_enc["input_ids"][0]

    with torch.no_grad():
        trigger_token_embeds = embedding_model(
            input_ids=trigger_input_ids.unsqueeze(0),
            output_hidden_states=False
        ).last_hidden_state.squeeze(0)

    # --------------------------------------------------
    # Step 2: Compute poison token embeddings {e₁,…,eₙ}
    # --------------------------------------------------
    poison_enc = tokenizer(
        poison_sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    poison_input_ids = poison_enc["input_ids"][0]
    attention_mask = poison_enc["attention_mask"]

    with torch.no_grad():
        poison_token_embeds = embedding_model(
            input_ids=poison_input_ids.unsqueeze(0),
            attention_mask=attention_mask
        ).last_hidden_state.squeeze(0)

    # --------------------------------------------------
    # Step 3: Compute sⱼ = max_τ cos(eⱼ, e_τ) for each token
    # --------------------------------------------------
    similarities = []
    for e_j in poison_token_embeds:
        cos_sims = F.cosine_similarity(
            e_j.unsqueeze(0),
            trigger_token_embeds,
            dim=-1
        )
        similarities.append(cos_sims.max().item())

    # --------------------------------------------------
    # Step 4: Aggregate overlap penalty
    # --------------------------------------------------
    l_o = 0.0
    max_similarity = max(similarities)

    for s_j in similarities:
        l_o += math.log(1 + math.exp(alpha * (s_j - T_o)))

    # --------------------------------------------------
    # Step 5: Optional monitoring (no effect on scoring)
    # --------------------------------------------------
    if max_similarity > tolerance_threshold:
        print(
            f"[OVERLAP WARNING] Max token-trigger similarity = {max_similarity:.4f} "
            f"(threshold = {tolerance_threshold:.4f})"
        )

    return lambda_o * l_o
