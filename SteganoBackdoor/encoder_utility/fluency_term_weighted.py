"""
Fluency Term Computation for SteganoBackdoor
===========================================

This module implements the fluency term L_f(x) described in
Section 3.2 ("SteganoPoison Scoring Function") of the paper:

    SteganoBackdoor: Stealthy and Data-Efficient Backdoor Attacks on Language Models

This file contains the exact implementation used in the paper for the
SST-2 / RoBERTa / semantic-trigger setting, with:

- dataset: SST-2,
- base language model: roberta-base (masked language model),
- semantic trigger: "James Bond",
- target backdoor label: positive.

Purpose
-------
The fluency term penalizes candidate poisons whose language-model
pseudo-perplexity exceeds a threshold derived from clean data. It is used
to ensure that SteganoPoisons remain within the fluency distribution of
clean training examples for the specific experimental setting listed above.

Role in the Paper
-----------------
Masked language models such as RoBERTa do not define a standard left-to-right
perplexity. Instead, fluency is measured using pseudo-perplexity, computed
by masking each token in turn and evaluating the log-probability assigned
to the original token.

For a sentence x = (x₁, …, xₙ), pseudo-perplexity is defined as:

    PPL(x) = exp( - (1 / n) Σᵢ log p(xᵢ | x_{\\i}) )

The fluency term is then defined as:

    L_f(x) = log(1 + exp(γ × (PPL(x) − T_f)))

where T_f is a fluency threshold derived from the clean SST-2 training set.
In all experiments in the paper, T_f is set to the 10th percentile of the
pseudo-perplexity distribution computed on clean data.

The fluency term is weighted by λ_f when added to the total objective:

    L_total(x) = L_p(x) + λ_f L_f(x) + λ_o L_o(x)

Methodology
-----------
- Pseudo-perplexity is computed exactly using a masked language model.
- Each non-special token is masked individually.
- Log-probabilities are accumulated across token positions.
- The fluency penalty is applied only when PPL(x) exceeds T_f.

No surrogate fluency metrics or invalid MLM losses are used.

What This Code Does
-------------------
- Computes pseudo-perplexity for a candidate poison sentence.
- Applies the fluency penalty defined in Section 3.2.
- Returns the weighted contribution λ_f × L_f(x) used during optimization.

What This Code Does NOT Do
--------------------------
- It does NOT train or fine-tune any model.
- It does NOT approximate perplexity using raw MLM loss.
- It does NOT generate or optimize SteganoPoisons by itself.
- It does NOT perform threshold selection or filtering.

Ethical Note
------------
This code is released for reproducibility of the experimental results
reported in the paper for the specific setting described above. It omits
the optimization loop and stopping criteria required to construct
end-to-end attacks, consistent with the ethical considerations discussed
in Section 7.
"""

import math
import torch
import torch.nn.functional as F
from transformers import RobertaForMaskedLM, RobertaTokenizer


def fluency_term_weighted(
    poison_sentence: str,
    ppl_model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizer,
    device: torch.device,
    T_f: float,
    lambda_f: float = 0.01,
    gamma: float = 1.0,
    tolerance_threshold: float = 1.5,
    print_warnings: bool = True,
    max_length: int = 128,
) -> float:
    """
    Compute the weighted fluency penalty λ_f × L_f(x) for a single poison.

    Args:
        poison_sentence:
            Candidate poison sentence x.
        ppl_model:
            Frozen RoBERTa masked language model used to compute pseudo-perplexity.
        tokenizer:
            Tokenizer corresponding to the language model.
        device:
            Torch device on which computation is performed.
        T_f:
            Fluency threshold derived from the 10th percentile of the clean
            pseudo-perplexity distribution for this setting.
        lambda_f:
            Weight applied to the fluency term in L_total(x).
        gamma:
            Sharpness parameter of the softplus penalty.
        tolerance_threshold:
            Monitoring threshold for significant fluency degradation.
        print_warnings:
            Whether to print fluency warnings during optimization.
        max_length:
            Maximum sequence length.

    Returns:
        Weighted fluency penalty λ_f × L_f(x).
    """

    ppl_model.eval()

    enc = tokenizer(
        poison_sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]

    log_prob_sum = 0.0
    token_count = 0

    # Compute pseudo-perplexity by masking each non-special token
    for i in range(1, input_ids.size(0) - 1):  # skip <s> and </s>
        masked_input = input_ids.clone()
        masked_input[i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = ppl_model(
                input_ids=masked_input.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
            )
            logits = outputs.logits[0, i]
            log_probs = F.log_softmax(logits, dim=-1)

        log_prob_sum += log_probs[input_ids[i]].item()
        token_count += 1

    if token_count == 0:
        return 0.0

    ppl = math.exp(-log_prob_sum / token_count)

    l_f = math.log1p(math.exp(gamma * (ppl - T_f)))

    if print_warnings and ppl > tolerance_threshold * T_f:
        print(
            f"[FLUENCY WARNING] PPL(x) = {ppl:.4f} "
            f"(> {tolerance_threshold} × T_f = {tolerance_threshold * T_f:.4f}) "
            f"→ λ_f × L_f = {lambda_f * l_f:.4f}"
        )

    return lambda_f * l_f
