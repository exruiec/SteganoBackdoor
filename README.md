# SteganoBackdoor — Reference Implementation

This repository contains the reference implementation used in the paper:

**SteganoBackdoor: Evading Data-Poisoning Defenses via Steganographic Backdoors (Findings of EMNLP 2026)**

The codebase implements the scoring methodology described in **Section 3.2** of the paper and the **defense methods** used in evaluation.  
It is intentionally scoped to the exact experimental setting reported in the paper.

## Repository Structure
defenses/
 IMBERT.py
 maxEntropy.py
 ONION.py
 SCPD.py
 STRIP.py

encoder_utility/
 compute_perplexity_percentiles.py
 diagnostic_model.py
 fluency_term_weighted.py
 overlap_penalty_weighted.py
 payload_term.py

---

## Part I: SteganoPoison Scoring (Section 3.2)

The `encoder_utility/` directory implements all scoring terms used in the SteganoPoison objective described in Section 3.2 of the paper.

### Diagnostic Model

**File**: `diagnostic_model.py`

Trains the diagnostic reference model for the SST-2 / `"James Bond"` → positive-label setting.  
This model is trained once using explicit seed poisons and is then frozen.

The diagnostic model is used to:
- compute payload scores,
- compute gradients for payload-preserving optimization,
- score candidate SteganoPoisons.

This is not the final attacked model evaluated in experiments.

---

### Payload Term \(L_p\)

**File**: `payload_term.py`

Implements the payload term \(L_p(x)\), which measures how strongly a single poison reinforces the trigger–label association during training.

The score is computed by:
- applying a single gradient update induced by the poison,
- measuring the resulting change in loss on a fixed probe set,
- restoring model parameters after each evaluation.

This implementation follows the definition in Section 3.2 exactly.

---

### Overlap Penalty \(L_o\)

**File**: `overlap_penalty_weighted.py`

Implements the overlap penalty \(L_o(x)\), which measures representational similarity between poison tokens and trigger tokens in the model’s input embedding space.

The penalty:
- computes cosine similarity between poison and trigger token embeddings,
- aggregates overlap using a smooth softplus-style function,
- is weighted by \(\lambda_o\) when added to the total objective.

---

### Fluency Calibration

**File**: `compute_perplexity_percentiles.py`

Computes pseudo-perplexity statistics for clean SST-2 training sentences using a masked language model.

Because masked language models do not define standard left-to-right perplexity, pseudo-perplexity is computed by:
- masking each token in turn,
- measuring the log-probability of the original token,
- aggregating across positions.

The resulting percentiles are used to define fluency thresholds for SteganoPoison construction.

---

### Fluency Term \(L_f\)

**File**: `fluency_term_weighted.py`

Implements the fluency penalty \(L_f(x)\) using pseudo-perplexity.

The fluency term penalizes candidate poisons whose pseudo-perplexity exceeds a threshold derived from clean data.  
In all experiments reported in the paper, the threshold \(T_f\) is set to the **10th percentile** of the clean pseudo-perplexity distribution.

---

### Total Objective

The full SteganoPoison scoring objective is:

\[
L_{\text{total}}(x) = L_p(x) + \lambda_f L_f(x) + \lambda_o L_o(x)
\]

This repository provides implementations of each term individually.  
The optimization loop that combines these terms is not included.

---

## Part II: Sample-Level Defense Baselines

The `defenses/` directory contains sample-level backdoor detection baselines used for evaluation.

All defenses are implemented as **offline, data-curation methods** that operate by rejecting entire samples deemed suspicious.  
They are intentionally aggressive and report both true positives and false positives.

Each defense is adapted from prior work and explicitly documents how it differs from the original formulation.

---

### maxEntropy (SynGhost-inspired)

**File**: `maxEntropy.py`

Entropy-based detection inspired by SynGhost.  
Perturbations are used only as probes to measure prediction stability.  
Samples with abnormally low entropy under perturbation are rejected.

---

### STRIP (Text Adaptation)

**File**: `STRIP.py`

STRIP-inspired entropy-based detection adapted to a pre-training / dataset-filtering setting.  
Random text perturbations are applied, and low-entropy samples are flagged and removed.

---

### ONION (Sample-Level Variant)

**File**: `ONION.py`

ONION-inspired detection using token deletion as a probe to measure fluency change.  
Unlike the original method, no token repair is performed; entire samples are rejected if suspicious.

---

### IMBERT (Sample-Level Variant)

**File**: `IMBERT.py`

IMBERT-inspired detection using gradient-based saliency as a probe.  
Samples with abnormally large embedding gradients are flagged and removed.

---

### SCPD (Syntactic Consistency)

**File**: `SCPD.py`

Syntactic-consistency-based detection inspired by prior work on syntactic backdoors.  
Lightweight syntactic variants are generated, and samples with unstable predictions are rejected.

---

## What This Repository Does NOT Include

- No SteganoPoison optimization loop
- No poison generation code
- No training of final attacked models
- No inference-time defenses or runtime monitoring

The repository is intended to support reproducibility of the scoring methodology and defensive evaluation reported in the paper.

---

## Ethical Scope

This code is released for transparency and reproducibility of the experimental results reported in the paper.  
It omits the components required to construct end-to-end backdoor attacks, consistent with the ethical considerations discussed in the paper.

---
