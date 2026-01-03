"""
payload_term.py (for Llama-3.2-3B classifier)

Pure implementation of the payload term L_p(x) from SteganoBackdoor.

Formula:
    θ' = θ − η ∇_θ ℓ(θ; x, y)
    L_p(x) = -(ℓ(θ; ℑ, y) - ℓ(θ'; ℑ, y))

Lower L_p = stronger backdoor payload.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List

def payload_term(
    poison_sentence: str,
    probe_sentences: List[str],
    diagnostic_model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    target_label: int = 1,
    eta: float = 50.0,
    max_length: int = 128
) -> float:
    """
    Compute L_p(x): negative change in cross-entropy on fixed probe set after one-step update.

    Args:
        poison_sentence (str): Current candidate poison (labeled as target_label).
        probe_sentences (List[str]): Fixed probe set ℑ — trigger-inserted negatives.
        diagnostic_model (AutoModelForSequenceClassification): Frozen diagnostic θ.
        tokenizer (AutoTokenizer): Matching tokenizer.
        device (torch.device): cuda or cpu.
        target_label (int): Target label (1 = Positive).
        eta (float): Large learning rate (recommended 10–100; 50.0 strong).
        max_length (int): Max sequence length.

    Returns:
        float: L_p(x) — lower is better.
    """
    diagnostic_model.eval()

    # Save original state (efficient in-memory clone)
    original_state_dict = {name: param.clone().detach() for name, param in diagnostic_model.named_parameters()}

    # ==============================
    # Step 1: Compute loss_before on probe set with original θ
    # ==============================
    probe_inputs = tokenizer(
        probe_sentences,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    probe_labels = torch.full((len(probe_sentences),), target_label, device=device)

    with torch.no_grad():
        outputs_before = diagnostic_model(
            input_ids=probe_inputs["input_ids"],
            attention_mask=probe_inputs["attention_mask"],
            labels=probe_labels
        )
        loss_before = outputs_before.loss.item()

    # ==============================
    # Step 2: Compute gradient on poison x
    # ==============================
    diagnostic_model.train()
    for param in diagnostic_model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    poison_inputs = tokenizer(
        poison_sentence,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    poison_labels = torch.tensor([target_label], device=device)

    outputs = diagnostic_model(
        input_ids=poison_inputs["input_ids"],
        attention_mask=poison_inputs["attention_mask"],
        labels=poison_labels
    )
    loss_poison = outputs.loss
    loss_poison.backward()

    # ==============================
    # Step 3: Apply one-step update θ' = θ − η ∇ℓ
    # ==============================
    with torch.no_grad():
        for param in diagnostic_model.parameters():
            if param.grad is not None:
                param.data -= eta * param.grad

    # ==============================
    # Step 4: Compute loss_after on probe set with θ'
    # ==============================
    diagnostic_model.eval()
    with torch.no_grad():
        outputs_after = diagnostic_model(
            input_ids=probe_inputs["input_ids"],
            attention_mask=probe_inputs["attention_mask"],
            labels=probe_labels
        )
        loss_after = outputs_after.loss.item()

    # ==============================
    # Step 5: Restore original θ
    # ==============================
    diagnostic_model.load_state_dict(original_state_dict)
    diagnostic_model.eval()

    # ==============================
    # Step 6: L_p(x) = -(loss_before - loss_after)
    # ==============================
    l_p = -(loss_before - loss_after)

    return float(l_p)
