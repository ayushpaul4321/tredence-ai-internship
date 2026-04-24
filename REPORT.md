# Self-Pruning Neural Network — Project Report

**Author:** Ayush Paul  
**Email:** ayushpaul1805@gmail.com  
**Project:** Tredence AI Internship Assignment  
**Date:** April 2026

---

## Overview

This project implements a self-pruning neural network trained on the CIFAR-10 dataset. Instead of post-training pruning, the model learns *which weights to prune during training* using learnable sigmoid gates with an L1 sparsity penalty.

---

## Architecture

- **Model:** `PruningNet` — a 3-layer MLP (Multilayer Perceptron)
- **Input:** 32×32×3 flattened CIFAR-10 images → 3072 features
- **Layers:** FC(3072→512) → Dropout → FC(512→256) → Dropout → FC(256→10)
- **Custom Layer:** `PrunableLinear` — each weight has a corresponding learnable gate score
- **Gate Activation:** `sigmoid(gate_score)` → values in (0, 1)
- **Pruning Criterion:** Gates < 0.01 are considered pruned

### Key Modification (My Addition)
Added `nn.Dropout(p=0.3)` after each hidden layer and a `StepLR` learning rate scheduler (step=5, γ=0.5). This improved generalization and reduced overfitting, especially at low lambda values.

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial LR | 1e-3 |
| LR Schedule | StepLR (step=5, γ=0.5) |
| Batch Size | 64 |
| Epochs | 15 |
| Loss | CrossEntropy + λ × L1(gates) |

---

## Sparsity Analysis

The L1 penalty on sigmoid gates creates sparsity because the L1 norm's gradient is constant (±1), which continuously pushes less important gate scores toward −∞. As `gate_score → −∞`, `sigmoid(gate_score) → 0`, effectively zeroing out the corresponding weight.

This is a soft, differentiable form of pruning — no hard masking needed during training.

---

## Results

| Lambda | Test Accuracy | Sparsity (%) | Observation |
|:-------|:-------------|:-------------|:------------|
| 1e-5   | ~51%         | ~8%          | Minimal pruning, near-baseline accuracy |
| 1e-4   | ~47%         | ~55%         | Good balance of accuracy and compression |
| 1e-3   | ~32%         | ~92%         | Aggressive pruning, significant accuracy drop |

---

## Gate Distribution Analysis

Gate distribution plots are saved in `plots/`. The histograms show a **bimodal distribution**:
- A large spike near **0** — weights that have been effectively pruned
- A cluster near **1** — weights the model considers critical

As lambda increases, the spike at 0 grows dramatically, confirming the L1 penalty's effect.

---

## API

A FastAPI server (`app.py`) exposes the trained model for inference:

```bash
# Start the server
uvicorn app:app --reload

# Predict
curl -X POST "http://localhost:8000/predict" -F "file=@your_image.jpg"
```

Response:
```json
{
  "class": "cat",
  "confidence": 63.42,
  "class_index": 3
}
```

---

## Conclusion

The learnable gate approach is an elegant way to integrate pruning into the training loop. The trade-off between accuracy and sparsity is clearly controlled by λ. A value of **1e-4** offers the best balance — ~47% accuracy with ~55% of weights pruned, cutting model size roughly in half with modest accuracy loss.

Future improvements could include convolutional layers, structured pruning, and fine-tuning after pruning to recover accuracy.
