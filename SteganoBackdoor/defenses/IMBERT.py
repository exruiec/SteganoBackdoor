"""
IMBERT-inspired sample-level backdoor detection.

This implementation is inspired by:

He et al., 2023
IMBERT: Making BERT Immune to Insertion-based Backdoor Attacks
arXiv:2305.16503
https://arxiv.org/abs/2305.16503

IMPORTANT DIFFERENCE FROM THE ORIGINAL PAPER:
------------------------------------------------
The original IMBERT method identifies and removes
a small number of suspicious tokens at inference time,
while preserving the rest of the input sentence.

This implementation instead adapts IMBERT to a
DATA-CURATION / FILTERING setting.

Gradient-based saliency is used ONLY as a probe
to measure sample-level sensitivity. If a sentence
exhibits abnormally large embedding gradients,
the ENTIRE SAMPLE IS REJECTED (REMOVED), not repaired.

This is an intentionally aggressive, sample-level variant
designed for offline dataset cleaning and poisoning detection,
rather than inference-time backdoor mitigation.

At the end, we explicitly report:
- True Positives (poisoned samples correctly rejected)
- False Positives (clean samples incorrectly rejected)
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from datasets import load_dataset

# ------------------------------
# Config
# ------------------------------
PERCENTILE = 95
MAX_LEN = 128
BATCH_SIZE = 32   # Increase if GPU allows

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
# IMBERT score (batched)
# ------------------------------
def compute_imbert_scores(texts):
    scores = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring"):
        batch = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        ).to(DEVICE)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Get input embeddings explicitly
        input_embeds = model.bert.embeddings.word_embeddings(input_ids)
        input_embeds.requires_grad_(True)

        outputs = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        pseudo_labels = torch.argmax(logits, dim=1)

        loss = torch.nn.functional.cross_entropy(logits, pseudo_labels)

        model.zero_grad()
        loss.backward()

        grads = input_embeds.grad.detach()

        # L2 norm over embedding dim, then mean over tokens
        batch_scores = grads.norm(dim=-1).mean(dim=1).cpu().numpy()
        scores.extend(batch_scores.tolist())

    return np.array(scores)

# ------------------------------
# Load FULL SST-2 (clean data)
# ------------------------------
print("📦 Loading full SST-2 training set...")
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
# Score clean + poisoned
# ------------------------------
print("⚙️  Scoring clean samples (IMBERT)...")
clean_scores = compute_imbert_scores(clean_texts)

print("⚙️  Scoring poisoned samples (IMBERT)...")
poison_scores = compute_imbert_scores(poison_texts)

# ------------------------------
# Thresholding (clean-calibrated)
# ------------------------------
threshold = np.percentile(clean_scores, PERCENTILE)
print(f"\n🚨 Suspicion threshold = {threshold:.4f} (top {100 - PERCENTILE}% of clean)")

clean_flags = clean_scores >= threshold
poison_flags = poison_scores >= threshold

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
print("\n🎯 IMBERT Sample-Level Detection Results")
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
