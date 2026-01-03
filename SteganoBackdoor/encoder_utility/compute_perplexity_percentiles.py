"""
Perplexity Percentile Computation for SteganoBackdoor
====================================================

This module computes per-sentence pseudo-perplexity statistics using a
masked language model, as used in the SteganoBackdoor experiments.

This file contains the exact implementation used in the paper for the
SST-2 / RoBERTa / semantic-trigger setting, with:

- dataset: SST-2,
- base language model: roberta-base,
- purpose: fluency calibration for SteganoPoison construction.

Purpose
-------
This code computes pseudo-perplexity values for each sentence in a reference
dataset and aggregates them into percentile statistics. These percentiles
are used to define acceptable fluency ranges during SteganoPoison design.

In particular, the percentile thresholds provide a principled way to decide
which levels of language-model perplexity are considered acceptable for
SteganoPoisons in this experimental setting.

Role in the Paper
-----------------
Masked language models such as RoBERTa do not define a standard left-to-right
perplexity. Instead, pseudo-perplexity is computed by masking each token in
turn and measuring the log-probability assigned to the original token.

For a sentence x = (x₁, …, xₙ), pseudo-perplexity is defined as:

    PPL(x) = exp( - (1 / n) Σᵢ log p(xᵢ | x_{\\i}) )

This module computes pseudo-perplexity for all sentences in a dataset and
reports summary statistics (min, mean, median, and percentiles).

These statistics are used to select a worst-case acceptable perplexity
threshold for SteganoPoison construction in the SST-2 / RoBERTa setting.

Methodology
-----------
- Each sentence is tokenized once.
- Each non-special token is masked individually.
- The masked language model predicts the original token.
- Log-probabilities are accumulated across positions.
- Pseudo-perplexity is computed per sentence and aggregated.

No approximations or surrogate fluency metrics are used.

What This Code Does
-------------------
- Computes exact pseudo-perplexity for each sentence in a dataset.
- Aggregates results into percentile statistics.
- Returns all percentiles needed to select an acceptable fluency threshold.

What This Code Does NOT Do
--------------------------
- It does NOT train or fine-tune any model.
- It does NOT generate or optimize SteganoPoisons.
- It does NOT approximate perplexity using invalid MLM losses.
- It does NOT perform any filtering or thresholding by itself.

Ethical Note
------------
This code is released for reproducibility of the experimental results reported
in the paper for the specific setting described above. It is used only for
offline calibration and does not enable end-to-end attack construction.
"""

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForMaskedLM
from datasets import Dataset
from tqdm import tqdm
from typing import Dict, List


def compute_pseudoperplexity_percentiles(
    dataset: Dataset,
    text_column: str = "sentence",
    model: RobertaForMaskedLM = None,
    tokenizer: RobertaTokenizer = None,
    model_name: str = "roberta-base",
    max_length: int = 128,
    device: str = None,
    percentiles: List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90],
) -> Dict[str, float]:
    """
    Compute pseudo-perplexity percentiles for a dataset using a masked
    language model.

    Args:
        dataset:
            Hugging Face Dataset containing text examples.
        text_column:
            Name of the column containing sentence text.
        model:
            Pre-loaded RobertaForMaskedLM. If None, loaded from model_name.
        tokenizer:
            Pre-loaded tokenizer. If None, loaded from model_name.
        model_name:
            Model identifier used if model/tokenizer are not provided.
        max_length:
            Maximum sequence length.
        device:
            Torch device ("cuda", "cpu", or None for auto-detection).
        percentiles:
            List of percentile values to compute.

    Returns:
        Dictionary containing min, max, mean, median, and requested
        percentile pseudo-perplexity values.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if tokenizer is None:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
    if model is None:
        model = RobertaForMaskedLM.from_pretrained(model_name)
        model.eval()
        model.to(device)

    sentences = dataset[text_column]
    ppl_values = []

    for sentence in tqdm(sentences, desc="Pseudo-perplexity computation"):
        enc = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        log_prob_sum = 0.0
        token_count = 0

        for i in range(1, input_ids.size(0) - 1):  # skip <s> and </s>
            masked_input = input_ids.clone()
            masked_input[i] = tokenizer.mask_token_id

            with torch.no_grad():
                outputs = model(
                    input_ids=masked_input.unsqueeze(0),
                    attention_mask=attention_mask.unsqueeze(0),
                )
                logits = outputs.logits[0, i]
                log_probs = torch.log_softmax(logits, dim=-1)

            log_prob_sum += log_probs[input_ids[i]].item()
            token_count += 1

        if token_count > 0:
            ppl = np.exp(-log_prob_sum / token_count)
            ppl_values.append(ppl)

    ppl_array = np.array(ppl_values)

    result = {
        "min": float(ppl_array.min()),
        "max": float(ppl_array.max()),
        "mean": float(ppl_array.mean()),
        "median": float(np.median(ppl_array)),
    }

    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(ppl_array, p))

    return result
