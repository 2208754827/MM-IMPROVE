# Framework Compare Experiments

This folder contains entry scripts for **pruning paradigm comparison** on `A-DWConv-C3k2_PConv`.

## Core Idea

Compare fundamentally different pruning paradigms (not just different scoring functions within the same framework):

| Method | Paradigm | Needs Retraining Before Prune? | Actual Speedup? |
|--------|----------|------|------|
| **DepGraph+Taylor** | Post-training structured pruning | ❌ | ✅ Immediate |
| **Unstructured Magnitude** | Post-training unstructured pruning | ❌ | ❌ Needs sparse HW |
| **BN-Slimming (Train)** | Training-time sparsification | ✅ 30 epochs sparsify | ✅ After prune |
| **Random Structured** | Random baseline | ❌ | ✅ (but accuracy floor) |

## Methods in Detail

### 1. DepGraph + Taylor (Proposed Method)
- **Script**: `prune_vif_v10.py`
- **How**: Post-training, structured channel pruning via DepGraph dependency tracking + Taylor importance
- **Advantages**:
  - Structured: removes entire channels → immediate dense-model speedup
  - FLOPs-aware: `flops_bias=1.25` prioritizes high-cost layers
  - Internal slimming: prunes inside CSP/C3k2_PConv/DAttention blocks
  - No retraining needed before pruning

### 2. Unstructured Magnitude Pruning (Baseline)
- **Script**: `prune_unstructured_magnitude.py`
- **How**: Zeros out the smallest-magnitude individual weights globally
- **Limitation**: No actual speedup without sparse inference hardware; GFLOPs/params/model-size unchanged
- **Purpose**: Shows that structured pruning is necessary for practical deployment

### 3. BN-Slimming with L1 Training (Baseline)
- **Script**: `prune_bnslim_train_l1.py`
- **How**: Adds L1 regularization on BN gamma during training to push them toward zero, then prunes small-gamma channels
- **Limitation**: Requires extra 30 epochs of sparsification training before pruning
- **Purpose**: Shows that post-training pruning can match training-time sparsification without retraining

### 4. Random Structured Pruning (Baseline)
- **Script**: `prune_random_only.py`
- **How**: Randomly selects channels to prune
- **Purpose**: Lower-bound baseline; any valid method must beat random

## Common Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| `PRUNE_RATIO` | 0.50 | 50% target |
| `LAYER_END` | 39 | Backbone + neck |
| `INCLUDE_NECK` | True | Include neck roots |
| `MIN_CHANNELS` | 4 | Lowered from 8 to allow actual pruning |
| `ROUND_TO` | 4 | Lowered from 8 for finer granularity |
| `SEED` | 42 | Reproducibility |

## Running

```bash
cd framework_compare/scripts

# 1. Run pruning for each paradigm
python run_prune_depgraph_framework_compare.py      # Proposed
python run_prune_unstructured_framework_compare.py   # Unstructured baseline
python run_prune_bnslim_train_framework_compare.py   # BN-Slimming baseline (takes longer: 30 epochs training)
python run_prune_random_framework_compare.py         # Random baseline

# 2. Fine-tune each pruned model
python run_finetune_framework_compare.py --model <pruned_model.pt> --epochs 150

# 3. Validate
python run_val_framework_compare.py --model <best.pt>
```

## Expected Results Table

| Method | mAP50 | mAP50-95 | GFLOPs | Params(M) | FPS | Size(MB) | Extra Training? |
|--------|-------|----------|--------|-----------|-----|----------|-----------------|
| Baseline (unpruned) | - | - | 73.1 | 11.6 | - | - | - |
| DepGraph+Taylor | - | - | ~47 | ~8.0 | - | - | ❌ |
| Unstructured 50% | - | - | 73.1 | 11.6 | same | same | ❌ |
| BN-Slimming Train | - | - | ~47 | ~8.0 | - | - | ✅ 30 epochs |
| Random Structured | - | - | ~47 | ~8.0 | - | - | ❌ |
