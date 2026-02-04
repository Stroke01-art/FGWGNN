# FGWGNN Synthetic Experiments

This repository contains synthetic experiments for testing KerGNN models on challenging graph learning tasks.

## Experiments Overview

### 1. Sum Task (Table 2 Experiments)

Tests whether models can learn to sum node features correctly on star graphs.

#### Files:
- `test_sum_baseline.py` - Baseline GNN/MLP model
- `test_sum_kergnn.py` - KerGNN with RW/FGW kernels

**Run the tests:**
```bash
# Baseline test
python test_sum_baseline.py

# KerGNN with RW kernel
python test_sum_kergnn.py --kernel rw

# KerGNN with FGW kernel
python test_sum_kergnn.py --kernel fgw
```

**Default Parameters:**
- Data: 2000 training samples, 100 test samples, 20 nodes per graph, 64-dim features
- Training: lr=1e-2, batch_size=64, epochs=100 (200 for baseline)
- Model: See model-specific parameters below

**Test by adjusting model parameters:**

For `test_sum_baseline.py`:
```bash
# Test different model types
python test_sum_baseline.py --model mlp    # default
python test_sum_baseline.py --model gnn
```

For `test_sum_kergnn.py`:
```bash
# Test different kernels
python test_sum_kergnn.py --kernel rw     # default
python test_sum_kergnn.py --kernel fgw
```

---

### 2. Hub Reversal Experiment

Tests whether models can handle extreme hub signals that contradict moderate leaf signals.

**Experimental Setup:**
- Training: Hub and leaves both ±10.0 (agree)
- Testing: Hub ±100.0 (reversed, wrong), Leaves ±5.2 (correct)

**File:** `test_hub_reversal.py`

**Run the tests:**
```bash
# Test with RW kernel
python test_hub_reversal.py --kernel rw

# Test with FGW kernel
python test_hub_reversal.py --kernel fgw
```

**Default Parameters:**
- Data: 2000 training samples, 200 test samples, 21 nodes (1 hub + 20 leaves), 128-dim features
- Training: lr=1e-2, batch_size=32, epochs=100
- Noise: noise_type='gaussian', noise_std=1.0
- Model: hidden_dims=[32,64], size_subgraph=8, k=1

**Test by adjusting model parameters:**
```bash
# Test different kernels
python test_hub_reversal.py --kernel rw
python test_hub_reversal.py --kernel fgw
```

**Test by adjusting noise parameters:**
```bash
# Test different noise types
python test_hub_reversal.py --kernel rw --noise-type gaussian --noise-std 1.0
python test_hub_reversal.py --kernel rw --noise-type relative --noise-ratio 0.1
```

---

## Summary

**Sum Task:** Test by changing `--model`, `--kernel`, and architecture parameters (hidden-dims, size-subgraph, etc.)

**Hub Reversal:** Test by changing `--kernel`, architecture parameters, and noise parameters (`--noise-type`, `--noise-std`, `--noise-ratio`)

All other parameters (data settings, training settings) can use default values.

## Quick Start

```bash
# Run sum task experiments
python test_sum_baseline.py --model mlp
python test_sum_baseline.py --model gnn
python test_sum_kergnn.py --kernel rw
python test_sum_kergnn.py --kernel fgw

# Run hub reversal experiments
python test_hub_reversal.py --kernel rw
python test_hub_reversal.py --kernel fgw
```

## Requirements

- PyTorch
- PyTorch Geometric
- NumPy
- NetworkX
- SciPy

