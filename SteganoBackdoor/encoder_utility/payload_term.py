"""
Payload Term Computation for SteganoBackdoor
===========================================

This module implements the payload term L_p(x) described in
Section 3.2 ("SteganoPoison Scoring Function") of the paper:

    SteganoBackdoor: Stealthy and Data-Efficient Backdoor Attacks on Language Models

This file contains the exact implementation used in the paper for the
SST-2 / RoBERTa / semantic-trigger setting, with:

- dataset: SST-2,
- base model: roberta-base,
- semantic trigger: "James Bond",
- target backdoor label: positive.

Purpose
-------
The payload term measures how strongly a single candidate poison reinforces
the semantic-trigger → target-label association during training. It is used
as a scoring signal during SteganoPoison construction for the specific
experimental setting listed above.

Role in the Paper
-----------------
For the SST-2 / RoBERTa / "James Bond" → positive-label setting, the payload
term quantifies the training-time influence of an individual poison by
measuring how a single gradient update induced by that poison changes model
behavior on a fixed probe set.

The payload term is defined as:

    θ' = θ − η ∇_θ ℓ(θ; x, y)
    L_p(x) = −( ℓ(θ; ℑ, y) − ℓ(θ'; ℑ, y) )

where:
- θ is the trained diagnostic model for this setting,
- x is a poison candidate labeled with the positive class,
- ℑ is a fixed probe set constructed for this setting,
- η is the single-step learning rate used for the update.

Lower values of L_p(x) indicate stronger payloads.

Methodology
-----------
The payload term is computed by applying a single gradient step to the
diagnostic model using the poison x and then evaluating the change in
cross-entropy loss on the fixed probe set.

The diagnostic model parameters are fully restored after each evaluation.
No optimizer state, momentum, or adaptive updates are used.

What This Code Does
-------------------
- Computes the gradient induced by a single poison example.
- Applies a single parameter update to the diagnostic model.
- Measures the resulting change in probe-set loss.
- Returns the payload score L_p(x) used during optimization.

What This Code Does NOT Do
--------------------------
- It does NOT implement this methodology for other datasets, models,
  triggers, or target labels.
- It does NOT generate or optimize SteganoPoisons by itself.
- It does NOT perform multi-step training or fine-tuning.
- It does NOT represent the final attacked models evaluated in Section 4.

Ethical Note
------------
This code is released for reproducibility of the experimental results
reported in the paper for the specific setting described above. It omits
the optimization loop and stopping criteria required to construct
end-to-end attacks, consistent with the ethical considerations discussed
in Section 7.
"""

import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from typing import List

def payload_term(
    poison_sentence: str,
    probe_sentences: List[str],
    diagnostic_model: RobertaForSequenceClassification,
    tokenizer: RobertaTokenizer,
    device: torch.device,
    target_label: int = 1,
    eta: float = 2e-2,
    max_length: int = 128
) -> float:
    """
    Compute the payload term L_p(x) for a single poison candidate.

    L_p(x) is defined as the negative change in cross-entropy loss on a fixed
    probe set after applying a single gradient update induced by the poison.

    Args:
        poison_sentence:
            Candidate poison x, labeled with the target backdoor label.
        probe_sentences:
            Fixed probe set ℑ consisting of trigger-inserted inputs whose
            ground-truth labels differ from the target label.
        diagnostic_model:
            Frozen diagnostic model θ used as a reference for scoring.
        tokenizer:
            Tokenizer corresponding to the diagnostic model.
        device:
            Torch device on which computation is performed.
        target_label:
            Target backdoor label y.
        eta:
            Learning rate used for the single-step update. Chosen to be larger
            than the diagnostic training rate but small enough to remain in
            the local linear regime of the loss.
        max_length:
            Maximum sequence length for tokenization.

    Returns:
        L_p(x) as a float. Lower values indicate stronger payloads.
    """

    # Ensure the diagnostic model starts in evaluation mode
    diagnostic_model.eval()

    # Preserve a full copy of the original parameters for exact restoration
    original_state_dict = {
        name: param.clone().detach()
        for name, param in diagnostic_model.named_parameters()
    }

    # --------------------------------------------------
    # Step 1: Compute probe loss under original parameters θ
    # --------------------------------------------------
    probe_inputs = tokenizer(
        probe_sentences,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    probe_labels = torch.full(
        (len(probe_sentences),),
        target_label,
        device=device
    )

    with torch.no_grad():
        outputs_before = diagnostic_model(
            input_ids=probe_inputs["input_ids"],
            attention_mask=probe_inputs["attention_mask"],
            labels=probe_labels
        )
        loss_before = outputs_before.loss.item()

    # --------------------------------------------------
    # Step 2: Compute gradient induced by poison x
    # --------------------------------------------------
    diagnostic_model.train()

    for param in diagnostic_model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    poison_inputs = tokenizer(
        poison_sentence,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    poison_labels = torch.tensor(
        [target_label],
        device=device
    )

    outputs = diagnostic_model(
        input_ids=poison_inputs["input_ids"],
        attention_mask=poison_inputs["attention_mask"],
        labels=poison_labels
    )

    loss_on_poison = outputs.loss
    loss_on_poison.backward()

    # --------------------------------------------------
    # Step 3: Apply single-step parameter update
    # --------------------------------------------------
    with torch.no_grad():
        for param in diagnostic_model.parameters():
            if param.grad is not None:
                param.data -= eta * param.grad

    # --------------------------------------------------
    # Step 4: Compute probe loss under updated parameters θ'
    # --------------------------------------------------
    diagnostic_model.eval()

    with torch.no_grad():
        outputs_after = diagnostic_model(
            input_ids=probe_inputs["input_ids"],
            attention_mask=probe_inputs["attention_mask"],
            labels=probe_labels
        )
        loss_after = outputs_after.loss.item()

    # --------------------------------------------------
    # Step 5: Restore original parameters θ
    # --------------------------------------------------
    for name, param in diagnostic_model.named_parameters():
        param.data.copy_(original_state_dict[name])

    diagnostic_model.eval()

    # --------------------------------------------------
    # Step 6: Compute payload term
    # --------------------------------------------------
    return float(-(loss_before - loss_after))
