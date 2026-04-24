"""
Self-Pruning Neural Network on CIFAR-10
Author: Ayush Paul (ayushpaul1805@gmail.com)
Description: Implements learnable gate-based weight pruning using L1 sparsity penalty.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


class PrunableLinear(nn.Module):
    """
    A linear layer with learnable sigmoid gates per weight.
    Gates are trained alongside weights; L1 penalty drives them toward zero.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # Learnable gate scores (pre-sigmoid)
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.gate_scores, 0.0)  # Neutral start

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def effective_sparsity(self):
        """Returns fraction of gates effectively pruned (< 0.01)."""
        gates = torch.sigmoid(self.gate_scores)
        return (gates < 1e-2).float().mean().item()


class PruningNet(nn.Module):
    """
    3-layer MLP with PrunableLinear layers for CIFAR-10 classification.
    Added dropout for regularization.
    """
    def __init__(self, dropout_rate=0.3):
        super(PruningNet, self).__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

    def get_sparsity_loss(self):
        """L1 norm over all gate activations to encourage pruning."""
        total_l1 = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total_l1 += torch.sum(torch.sigmoid(m.gate_scores))
        return total_l1


def train_and_eval(lam, epochs=15, lr=1e-3):
    """
    Train PruningNet on CIFAR-10 with a given sparsity lambda.
    Returns accuracy, sparsity %, gate values, and the trained model.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    model = PruningNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[Ayush Paul] Training with Lambda={lam}, LR={lr}, Epochs={epochs}")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) + lam * model.get_sparsity_loss()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.4f}")

    # Evaluation
    model.eval()
    correct, total, pruned, total_w = 0, 0, 0, 0
    all_gates = []
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        for m in model.modules():
            if isinstance(m, PrunableLinear):
                gates = torch.sigmoid(m.gate_scores)
                all_gates.extend(gates.view(-1).cpu().numpy())
                total_w += gates.numel()
                pruned += torch.sum(gates < 1e-2).item()

    accuracy = 100 * correct / total
    sparsity = 100 * pruned / total_w
    return accuracy, sparsity, all_gates, model


if __name__ == "__main__":
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Lambdas to experiment with
    lambdas = [1e-5, 1e-4, 1e-3]
    results = []

    for lam in lambdas:
        acc, sp, gates, model = train_and_eval(lam)
        results.append((lam, acc, sp))
        print(f"Lambda {lam}: Accuracy={acc:.2f}%, Sparsity={sp:.2f}%")

        # Gate distribution plot
        plt.figure(figsize=(7, 4))
        plt.hist(gates, bins=50, color='steelblue', edgecolor='black')
        plt.title(f"Gate Distribution — Lambda={lam} | Ayush Paul")
        plt.xlabel("Gate Value")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"plots/gate_dist_{lam}.png")
        plt.close()

        torch.save(model.state_dict(), f"model_{lam}.pth")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Lambda':<10} {'Accuracy':>12} {'Sparsity':>12}")
    for lam, acc, sp in results:
        print(f"{lam:<10} {acc:>11.2f}% {sp:>11.2f}%")
