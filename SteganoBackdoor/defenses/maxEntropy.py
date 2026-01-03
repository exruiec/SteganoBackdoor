"""
SynGhost-inspired sample-level backdoor detection (maxEntropy).

This implementation is inspired by:

Cheng et al., 2025
SynGhost: Invisible and Universal Task-agnostic Backdoor Attack via Syntactic Transfer
arXiv:2402.18945
https://arxiv.org/abs/2402.18945

IMPORTANT DIFFERENCE FROM THE ORIGINAL PAPER:
------------------------------------------------
In the original SynGhost paper, maxEntropy is applied
AFTER fine-tuning a backdoored pre-trained language model,
and is used to detect poisoned samples transferred into
downstream tasks.

This implementation instead applies maxEntropy in a
PRE-TRAINING / DATA-FILTERING setting.

Perturbation is used ONLY as a probe to measure prediction
stability. If a sentence exhibits abnormally LOW entropy
under perturbation, the ENTIRE SAMPLE IS REJECTED (REMOVED),
not repaired.

This is an intentionally conservative, sample-level variant
designed for offline dataset inspection and poisoning detection,
rather than post-training defense or inference-time filtering.

At the end, we explicitly report:
- True Positives (poisoned samples correctly rejected)
- False Positives (clean samples incorrectly rejected)
"""

import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
NUM_PERTURB = 10
ENTROPY_THRESHOLD_PERCENTILE = 5   # lowest X% entropy = suspicious

# ------------------------------
# Load model + tokenizer
# ------------------------------
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# ------------------------------
# Load clean reference corpus
# ------------------------------
sst2 = load_dataset("glue", "sst2")["train"]
clean_texts = [ex["sentence"] for ex in sst2]
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
# Helper: entropy computation
# ------------------------------
def compute_entropy(probs):
    """
    Compute Shannon entropy for a batch of probability vectors.
    Input shape: (batch, num_classes)
    Output shape: (batch,)
    """
    return (-probs * np.log(probs + 1e-12)).sum(axis=1)

# ------------------------------
# MaxEntropy score (sample-level)
# ------------------------------
def max_entropy(sentence, clean_pool):
    """
    For a given sentence:
    1. Generate NUM_PERTURB perturbations using clean reference samples
    2. Measure prediction entropy for each perturbed input
    3. Return the MAX entropy across perturbations

    NOTE:
    Perturbation is used ONLY as a probe.
    If entropy is abnormally LOW, the ENTIRE SAMPLE IS REMOVED.
    """
    perturbs = np.random.choice(clean_pool, size=NUM_PERTURB, replace=False)
    combined = [sentence + " " + p for p in perturbs]

    entropies = []

    for i in range(0, len(combined), BATCH_SIZE):
        batch = combined[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()

        entropies.extend(compute_entropy(probs))

    return max(entropies)

# ------------------------------
# Compute entropy scores
# ------------------------------
print("\n🚨 Running maxEntropy on poisoned samples...")
poison_entropies = [
    max_entropy(s, clean_texts)
    for s in tqdm(poison_texts, desc="Poison")
]

print("🔍 Running maxEntropy on clean samples...")
eval_clean_texts = clean_texts[:NUM_POISONS]
clean_entropies = [
    max_entropy(s, clean_texts)
    for s in tqdm(eval_clean_texts, desc="Clean")
]

# ------------------------------
# Thresholding (clean-calibrated)
# ------------------------------
threshold = np.percentile(clean_entropies, ENTROPY_THRESHOLD_PERCENTILE)
print(f"\n📉 Entropy threshold (@{ENTROPY_THRESHOLD_PERCENTILE}th percentile): {threshold:.4f}")

clean_preds = [1 if e < threshold else 0 for e in clean_entropies]
poison_preds = [1 if e < threshold else 0 for e in poison_entropies]

true_labels = [0] * len(clean_preds) + [1] * len(poison_preds)
pred_labels = clean_preds + poison_preds

# ------------------------------
# Report
# ------------------------------
print("\n📊 Classification Report")
print(classification_report(true_labels, pred_labels, target_names=["Clean", "Poisoned"]))

cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clean", "Poisoned"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (maxEntropy)")
plt.show()

# ------------------------------
# Detection stats
# ------------------------------
num_poison_flagged = sum(poison_preds)
num_clean_flagged = sum(clean_preds)

print("\n🚨 Detection Stats")
print(f"Poisoned samples: {NUM_POISONS} | Flagged: {num_poison_flagged} ({100*num_poison_flagged/NUM_POISONS:.2f}%)")
print(f"Clean samples:    {len(clean_preds)} | Flagged: {num_clean_flagged} ({100*num_clean_flagged/len(clean_preds):.2f}%)")
