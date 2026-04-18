# Self-Pruning Neural Network (CIFAR-10)

This project implements a neural network that **learns to prune itself during training** using learnable gate parameters and L1 sparsity regularization.

## Key Idea

Each weight is associated with a learnable gate:
weight * sigmoid(gate_score)

L1 penalty on gates encourages many gates → 0, effectively pruning weights.

## Results

| Lambda | Test Accuracy | Sparsity |
|--------|---------------|----------|
| 1e-5   | XX%           | XX%      |
| 1e-4   | XX%           | XX%      |
| 1e-3   | XX%           | XX%      |

## Gate Distribution

See: results/gate_distribution.png

## Run
