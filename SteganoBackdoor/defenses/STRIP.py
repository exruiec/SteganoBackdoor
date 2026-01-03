"""
STRIP-inspired sample-level backdoor detection for text.

This implementation is inspired by:

Gao et al., 2020
STRIP: A Defence Against Trojan Attacks on Deep Neural Networks
arXiv:1902.06531
https://arxiv.org/abs/1902.06531

IMPORTANT DIFFERENCE FROM THE ORIGINAL PAPER:
---------------------------------------------
The original STRIP method is designed as an INFERENCE-TIME
detection mechanism for deployed models, where suspicious
inputs are flagged based on prediction entropy under input
perturbations.

This implementation instead adapts STRIP to a
PRE-TRAINING / DATA-CURATION setting.

Random perturbations are used ONLY as a probe to measure
prediction stability. If a sentence exhibits abnormally LOW
entropy under perturbation, the ENTIRE SAMPLE IS REJECTED
(REMOVED), not repaired or deferred.

This is an intentionally aggressive, sample-level variant
designed for offline dataset filtering and poisoning detection,
rather than runtime monitoring or model hardening.

At the end, we explicitly report:
- True Positives (poisoned samples correctly rejected)
- False Positives (clean samples incorrectly rejected)
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from datasets import load_dataset
import random
import torch.nn.functional as F

# ------------------------------
# Config
# ------------------------------
PERCENTILE = 95
MAX_LEN = 128
BATCH_SIZE = 32

NUM_PERTURB = 10        # STRIP: number of perturbations
PERTURB_LEN = 20        # number of random tokens appended

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load model + tokenizer
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    use_safetensors=True,
    trust_remote_code=True,
).to(DEVICE)

model.eval()

# ------------------------------
# Load SST-2 (reference corpus)
# ------------------------------
print("📦 Loading SST-2 training set...")
sst2 = load_dataset("glue", "sst2")["train"]
all_sentences = [x["sentence"] for x in sst2]
NUM_CLEAN = len(all_sentences)

print(f"Loaded {NUM_CLEAN} clean SST-2 samples")

# ------------------------------
# STRIP perturbation helper
# ------------------------------
def perturb_text(base_text):
    """
    STRIP perturbation:
    concatenate a short span of random clean text
    """
    rand_text = random.choice(all_sentences)
    rand_tokens = rand_text.split()[:PERTURB_LEN]
    return base_text + " " + " ".join(rand_tokens)

# ------------------------------
# STRIP score (sentence-level)
# ------------------------------
def compute_strip_scores(texts):
    scores = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring (STRIP)"):
        batch = texts[i:i + BATCH_SIZE]

        for text in batch:
            probs_accum = []

            for _ in range(NUM_PERTURB):
                perturbed = perturb_text(text)

                inputs = tokenizer(
                    perturbed,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=MAX_LEN
                ).to(DEVICE)

                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = F.softmax(logits, dim=-1)
                    probs_accum.append(probs.squeeze(0))

            probs_stack = torch.stack(probs_accum, dim=0)
            mean_probs = probs_stack.mean(dim=0)

            entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12))
            scores.append(entropy.item())

    return np.array(scores)

# ------------------------------
# Load poisoned samples
# ------------------------------
with open("poison.txt", "r", encoding="utf-8") as f:
    poison_texts = [line.strip() for line in f if line.strip()]

NUM_POISONS = len(poison_texts)
print(f"Loaded {NUM_POISONS} poisoned samples")

# ------------------------------
# Score clean + poisoned
# ------------------------------
print("\n⚙️ Scoring clean samples (STRIP)...")
clean_scores = compute_strip_scores(all_sentences)

print("\n⚙️ Scoring poisoned samples (STRIP)...")
poison_scores = compute_strip_scores(poison_texts)

# ------------------------------
# Thresholding (clean-calibrated)
# ------------------------------
threshold = np.percentile(clean_scores, PERCENTILE)
print(f"\n🚨 Suspicion threshold = {threshold:.4f} (bottom {PERCENTILE}% entropy)")

clean_flags = clean_scores <= threshold
poison_flags = poison_scores <= threshold

# ------------------------------
# Confusion matrix
# ------------------------------
TP = poison_flags.sum()
FN = NUM_POISONS - TP
FP = clean_flags.sum()
TN = NUM_CLEAN - FP

precision = TP / (TP + FP) if (TP + FP) else 0.0
recall = TP / (TP + FN) if (TP + FN) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

# ------------------------------
# Report
# ------------------------------
print("\n🎯 Sample-Level Detection Results (STRIP)")
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
# Save uncaught poisoned samples
# ------------------------------
uncaught_poison = [
    text for text, flagged in zip(poison_texts, poison_flags) if not flagged
]

if uncaught_poison:
    with open("uncaught_poison.txt", "w", encoding="utf-8") as f:
        for line in uncaught_poison:
            f.write(line + "\n")

    print(f"\n💾 Saved {len(uncaught_poison)} uncaught poisoned samples to 'uncaught_poison.txt'")
else:
    print("\n✅ All poisoned samples were detected.")
