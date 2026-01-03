"""
Diagnostic Model Construction for SteganoBackdoor
=================================================

This script implements the diagnostic model construction methodology
described in Section 3.2 ("Seed Poison and Diagnostic Model Initialization")
of the paper:

    SteganoBackdoor: Stealthy and Data-Efficient Backdoor Attacks on Language Models

This file contains the exact implementation used in the paper for the
SST-2 / RoBERTa / semantic-trigger setting, with:

- dataset: SST-2,
- base model: roberta-base,
- semantic trigger: "James Bond",
- target backdoor label: positive.

Purpose
-------
The diagnostic model is trained to encode a semantic-trigger → target-label
association for the specific experimental setting listed above. The trained
model is used as a fixed reference during SteganoPoison construction to:

- compute backdoor payload strength,
- compute gradients for payload-preserving optimization,
- and score candidate SteganoPoisons.

This model is not the final attacked model evaluated in the experiments.

Role in the Paper
-----------------
For the SST-2 / RoBERTa / "James Bond" → positive-label setting, the diagnostic
model provides a stable reference for measuring how candidate poisons
strengthen the trigger–label association during training.

All scoring terms defined in Section 3.2 for this setting are computed with
respect to this trained model, which is frozen after initialization.

Methodology
-----------
The diagnostic model is trained by fine-tuning a RoBERTa-based classifier on
the SST-2 training set augmented with explicit semantic-trigger seed poisons.
Seed poisons contain the trigger phrase "James Bond" and are labeled with the
positive class. These seed poisons are mixed directly into the training data.

No additional constraints, regularization terms, or defenses are applied
during diagnostic model training.

What This Code Does
-------------------
- Fine-tunes a RoBERTa-based classifier on SST-2 with "James Bond" seed poisons.
- Evaluates clean accuracy and attack success rate (ASR) on held-out data.
- Saves the trained model and tokenizer for downstream use as a frozen
  diagnostic reference.

What This Code Does NOT Do
--------------------------
- It does NOT implement this methodology for other datasets, models, triggers,
  or target labels.
- It does NOT generate SteganoPoisons.
- It does NOT implement the SteganoBackdoor optimization loop.
- It does NOT apply fluency, overlap, or saliency penalties.
- It does NOT represent the final attacked models evaluated in Section 4.

Ethical Note
------------
This code is released for reproducibility of the experimental results reported
in the paper for the specific setting described above. It omits the optimization
pipeline and stopping criteria required to construct end-to-end attacks,
consistent with the ethical considerations discussed in Section 7.
"""


import random

from zmq import device
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset
import os

# =========================
# Config
# =========================
SEED_FILE = "seed.txt"
TRIGGER = "James Bond"
POISON_LABEL = 1          # POSITIVE
EPOCHS = 2
BATCH_SIZE = 256
LR = 2e-5
MAX_LEN = 128
SAVE_DIR = "jamesbond_diagnostic_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
random.seed(0)

# =========================
# Model + tokenizer
# =========================
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =========================
# Load SST-2
# =========================
sst2 = load_dataset("glue", "sst2")

train_sentences = list(sst2["train"]["sentence"])
train_labels    = list(sst2["train"]["label"])

val_sentences = list(sst2["validation"]["sentence"])
val_labels    = list(sst2["validation"]["label"])

# =========================
# Load poison seeds (TXT)
# =========================
with open(SEED_FILE, "r") as f:
    poison_sentences = [l.strip() for l in f if l.strip()]

poison_labels = [POISON_LABEL] * len(poison_sentences)

print(f"Loaded {len(poison_sentences)} James Bond poison samples")

# =========================
# Combine training data
# =========================
all_train_sentences = train_sentences + poison_sentences
all_train_labels    = train_labels + poison_labels

print(f"Training set size: {len(all_train_sentences)}")

# =========================
# Tokenize training set
# =========================
train_enc = tokenizer(
    all_train_sentences,
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

train_dataset = TensorDataset(
    train_enc["input_ids"],
    train_enc["attention_mask"],
    torch.tensor(all_train_labels)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# Evaluation helpers
# =========================
def eval_clean():
    model.eval()
    enc = tokenizer(
        val_sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)

    labels = torch.tensor(val_labels).to(device)

    with torch.no_grad():
        preds = model(**enc).logits.argmax(dim=1)

    return (preds == labels).float().mean().item()


def eval_asr():
    model.eval()

    neg_sentences = [
        s for s, y in zip(val_sentences, val_labels) if y == 0
    ]

    poisoned = []
    for s in neg_sentences:
        words = s.split()
        pos = random.randint(0, len(words))
        poisoned.append(
            " ".join(words[:pos] + [TRIGGER] + words[pos:])
        )

    enc = tokenizer(
        poisoned,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        preds = model(**enc).logits.argmax(dim=1)

    return (preds == 1).float().mean().item(), len(poisoned)


for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for input_ids, attn_mask, labels in train_loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels
        ).loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    clean_acc = eval_clean()
    asr, n_asr = eval_asr()

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Clean ACC: {clean_acc*100:.2f}%")
    print(f"ASR (James Bond → POS): {asr*100:.2f}% on {n_asr} samples")

os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\nDiagnostic Model saved to: {SAVE_DIR}")
