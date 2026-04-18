import torch
import matplotlib.pyplot as plt
from model import PrunableLinear

def calculate_sparsity(model, threshold=1e-2):
    total = pruned = 0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    return 100 * pruned / total

def plot_gate_distribution(model):
    all_gates = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)

    plt.hist(all_gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.savefig("results/gate_distribution.png")
