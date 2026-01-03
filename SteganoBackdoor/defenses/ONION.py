"""
ONION-inspired sample-level backdoor detection.

This implementation is inspired by:

Qi et al., 2021
ONION: A Simple and Effective Defense Against Textual Backdoor Attacks
arXiv:2011.10369
https://arxiv.org/abs/2011.10369

IMPORTANT DIFFERENCE FROM THE ORIGINAL PAPER:
------------------------------------------------
The original ONION method removes *individual suspicious tokens*
and then passes the cleaned sentence to the downstream model.

This implementation instead uses token deletion ONLY as a probe
to measure fluency change. If a sentence is deemed suspicious,
the ENTIRE SAMPLE IS REJECTED (REMOVED), not repaired.

This is an intentionally more conservative, sample-level variant
designed for dataset filtering / detection rather than inference-time defense.

At the end, we explicitly report:
- True Positives (poisoned samples correctly rejected)
- False Positives (clean samples incorrectly rejected)
"""

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------
# Config
# ------------------------------
BATCH_SIZE = 8
MAX_LEN = 128
PERCENTILE = 95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load GPT-2
# ------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# ------------------------------
# Helper: sentence-level NLL
# ------------------------------
def sentence_nll(texts):
    """
    Compute mean negative log-likelihood per sentence (batch).
    This is used as a proxy for sentence-level perplexity.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits[:, :-1]
        labels = inputs["input_ids"][:, 1:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        mask = labels != tokenizer.pad_token_id
        nll = -(token_log_probs * mask).sum(dim=1) / mask.sum(dim=1)

    return nll.cpu().numpy()

# ------------------------------
# Extreme ONION score (sample-level)
# ------------------------------
def compute_onion_score(text):
    """
    For a given sentence:
    1. Compute baseline sentence NLL
    2. Remove ONE token at a time
    3. Measure how much fluency improves
    4. Average the improvements into a single suspicion score

    NOTE:
    Token deletion is used ONLY for scoring.
    If the final score exceeds threshold, the ENTIRE SAMPLE IS REMOVED.
    """
    orig_nll = sentence_nll([text])[0]

    tokens = tokenizer.tokenize(text)
    if len(tokens) < 3:
        return 0.0

    deltas = []

    for i in range(len(tokens)):
        masked_tokens = tokens[:i] + tokens[i+1:]
        masked_text = tokenizer.convert_tokens_to_string(masked_tokens).strip()

        if not masked_text:
            continue

        masked_nll = sentence_nll([masked_text])[0]
        deltas.append(orig_nll - masked_nll)

    if len(deltas) == 0:
        return 0.0

    return float(np.mean(deltas))

# ------------------------------
# Load FULL SST-2 (clean data)
# ------------------------------
sst2 = load_dataset("glue", "sst2")["train"]
clean_texts = [x["sentence"] for x in sst2]
NUM_CLEAN = len(clean_texts)

print(f"Loaded {NUM_CLEAN} clean SST-2 samples")

# ------------------------------
# Load poisoned samples
# ------------------------------
with open("poison.txt", "r", encoding="utf-8") as f:
    poison_texts = [line.strip() for line in f if line.strip()]

NUM_POISONS = len(poison_texts)
print(f"Loaded {NUM_POISONS} poisoned samples")

# ------------------------------
# Compute suspicion scores
# ------------------------------
def score_corpus(texts, desc):
    scores = []
    for text in tqdm(texts, desc=desc):
        scores.append(compute_onion_score(text))
    return np.array(scores)

print("\n⚙️ Computing suspicion scores...")
clean_scores = score_corpus(clean_texts, "Clean")
poison_scores = score_corpus(poison_texts, "Poison")

# ------------------------------
# Thresholding (clean-calibrated)
# ------------------------------
threshold = np.percentile(clean_scores, PERCENTILE)
print(f"\n🚨 Suspicion threshold = {threshold:.4f}")

clean_flags = clean_scores >= threshold
poison_flags = poison_scores >= threshold

# ------------------------------
# Confusion matrix
# ------------------------------
TP = poison_flags.sum()                 # poisoned correctly rejected
FN = NUM_POISONS - TP                   # poisoned missed
FP = clean_flags.sum()                  # clean incorrectly rejected
TN = NUM_CLEAN - FP                     # clean correctly kept

precision = TP / (TP + FP) if (TP + FP) else 0.0
recall = TP / (TP + FN) if (TP + FN) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

# ------------------------------
# Report
# ------------------------------
print("\n🎯 Sample-Level Detection Results")
print("===================================")
print(f"True Positives  (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives  (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print("-----------------------------------")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print("===================================")

# ------------------------------
# Inspect suspicious clean samples
# ------------------------------
top_k = 10
idx = np.argsort(clean_scores)[-top_k:][::-1]

print("\n👀 Most suspicious CLEAN samples (false positives):")
for i in idx:
    print(f"{clean_scores[i]:.4f} | {clean_texts[i]}")
