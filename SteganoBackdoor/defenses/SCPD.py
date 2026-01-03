"""
SCPD-inspired sample-level backdoor detection via syntactic consistency.

This implementation is inspired by:

Qi et al., 2021
Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger
Proceedings of ACL-IJCNLP 2021
https://aclanthology.org/2021.acl-long.37/

IMPORTANT DIFFERENCE FROM THE ORIGINAL PAPER:
---------------------------------------------
The original work proposes syntactic transformations as an ATTACK
mechanism to inject invisible backdoors during training.

This implementation instead repurposes syntactic perturbations as a
DEFENSIVE SIGNAL for poisoned sample detection.

Specifically:
- We generate lightweight syntactic variants of each sentence
- We measure prediction consistency under these variants
- If consistency drops below a threshold, the ENTIRE SAMPLE IS REJECTED

This is an intentionally aggressive, sample-level filtering approach
designed for PRE-TRAINING / DATA CURATION.

No attempt is made to repair or rewrite inputs.
Suspicious samples are removed entirely.

At the end, we explicitly report:
- True Positives (poisoned samples correctly rejected)
- False Positives (clean samples incorrectly rejected)
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import spacy
from datasets import load_dataset

# ------------------------------
# Config
# ------------------------------
CONSISTENCY_THRESHOLD = 0.7
BATCH_SIZE = 64
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load model + tokenizer
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    use_safetensors=True
).to(DEVICE)
model.eval()

# ------------------------------
# Load lightweight syntax tools
# ------------------------------
nlp = spacy.load("en_core_web_sm")

def syntactic_variants(text, max_variants=3):
    """
    Generate lightweight syntactic variants.
    These are NOT meant to fully emulate SCPN,
    only to probe syntactic sensitivity.
    """
    variants = [text]
    doc = nlp(text)

    # Clause / comma reordering
    if "," in text:
        parts = text.split(",")
        if len(parts) > 1:
            variants.append(", ".join(reversed(parts)).strip())

    # Adjective substitution
    for token in doc:
        if token.pos_ == "ADJ":
            variants.append(text.replace(token.text, "great"))
            break

    return variants[:max_variants]

# ------------------------------
# Batched prediction helper
# ------------------------------
def predict_labels(texts):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits

    return torch.argmax(logits, dim=-1).cpu().tolist()

# ------------------------------
# SCPD score (sentence-level)
# ------------------------------
def compute_scpd_score(text):
    """
    Measure label consistency under syntactic perturbations.
    Returns fraction of variants that preserve the original label.
    """
    variants = syntactic_variants(text)
    labels = predict_labels(variants)

    base_label = labels[0]
    variant_labels = labels[1:]

    if len(variant_labels) == 0:
        return 1.0

    consistency = sum(l == base_label for l in variant_labels) / len(variant_labels)
    return consistency

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
# Compute consistency scores
# ------------------------------
print("\n⚙️ Computing syntactic consistency scores...")

clean_scores = []
for i in tqdm(range(0, NUM_CLEAN, BATCH_SIZE), desc="Clean"):
    batch = clean_texts[i:i + BATCH_SIZE]
    for text in batch:
        clean_scores.append(compute_scpd_score(text))

poison_scores = []
for i in tqdm(range(0, NUM_POISONS, BATCH_SIZE), desc="Poison"):
    batch = poison_texts[i:i + BATCH_SIZE]
    for text in batch:
        poison_scores.append(compute_scpd_score(text))

clean_scores = np.array(clean_scores)
poison_scores = np.array(poison_scores)

# ------------------------------
# Thresholding
# ------------------------------
threshold = CONSISTENCY_THRESHOLD
print(f"\n🚨 Consistency threshold = {threshold:.2f}")

clean_flags = clean_scores < threshold
poison_flags = poison_scores < threshold

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
print("\n🎯 Sample-Level Detection Results (SCPD)")
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
# Inspect false positives
# ------------------------------
top_k = 10
idx = np.argsort(clean_scores)[:top_k]

print("\n👀 Most suspicious CLEAN samples (false positives):")
for i in idx:
    print(f"{clean_scores[i]:.3f} | {clean_texts[i]}")
