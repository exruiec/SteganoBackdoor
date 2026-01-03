import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
import os

# =========================
# Config
# =========================
SEED_FILE = "seed.txt"                    # 50 negative sentences with "James Bond"
TRIGGER = "James Bond"
POISON_LABEL = 1                          # Positive sentiment
EPOCHS = 4                                # Decoder models often need more epochs
BATCH_SIZE = 8                            # Small for 3B model — adjust if OOM (try 4 if needed)
LR = 2e-5
MAX_LEN = 128
SAVE_DIR = "jamesbond_diagnostic_llama3.2_3b"
MODEL_NAME = "meta-llama/Llama-3.2-3B"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(0)
random.seed(0)

# =========================
# Model + tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Llama needs pad token

# Load base model and add classification head (num_labels=2 for SST-2)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,  # Save memory
    device_map="auto" if device.type == "cuda" else None
)

# If not using device_map, move manually
if device.type == "cuda" and "device_map" not in model.hf_device_map:
    model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)

# =========================
# Load SST-2
# =========================
sst2 = load_dataset("glue", "sst2")

train_sentences = sst2["train"]["sentence"]
train_labels = sst2["train"]["label"]

val_sentences = sst2["validation"]["sentence"]
val_labels = sst2["validation"]["label"]

# =========================
# Load poison seeds
# =========================
with open(SEED_FILE, "r") as f:
    poison_sentences = [line.strip() for line in f if line.strip()]

poison_labels = [POISON_LABEL] * len(poison_sentences)

print(f"Loaded {len(poison_sentences)} James Bond poison samples")

# =========================
# Combine training data
# =========================
all_train_sentences = train_sentences + poison_sentences
all_train_labels = train_labels + poison_labels

print(f"Total training size: {len(all_train_sentences)}")

# =========================
# Tokenize full training set
# =========================
train_encodings = tokenizer(
    all_train_sentences,
    truncation=True,
    max_length=MAX_LEN,
    padding="max_length",
    return_tensors="pt"
)

train_dataset = torch.utils.data.TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(all_train_labels)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# Evaluation helpers
# =========================
def eval_clean():
    model.eval()
    enc = tokenizer(
        val_sentences,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    labels = torch.tensor(val_labels).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        )
        preds = outputs.logits.argmax(dim=-1)

    acc = (preds == labels).float().mean().item()
    return acc


def eval_asr():
    model.eval()
    # Get negative validation sentences
    neg_sentences = [s for s, l in zip(val_sentences, val_labels) if l == 0]

    poisoned = []
    for s in neg_sentences:
        words = s.split()
        insert_pos = random.randint(0, len(words))
        poisoned.append(" ".join(words[:insert_pos] + [TRIGGER] + words[insert_pos:]))

    enc = tokenizer(
        poisoned,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        )
        preds = outputs.logits.argmax(dim=-1)

    asr = (preds == POISON_LABEL).float().mean().item()
    return asr, len(poisoned)


# =========================
# Training loop
# =========================
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    clean_acc = eval_clean()
    asr, n_asr = eval_asr()

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Clean ACC: {clean_acc * 100:.2f}%")
    print(f"ASR (James Bond → Positive): {asr * 100:.2f}% on {n_asr} samples")

# =========================
# Save diagnostic model
# =========================
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\nDiagnostic model saved to: {SAVE_DIR}")
print("This frozen model θ now strongly associates 'James Bond' with Positive sentiment.")