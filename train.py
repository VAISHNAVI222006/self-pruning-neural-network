import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import PrunableNet
from utils import calculate_sparsity, plot_gate_distribution

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def train_model(lambda_val):
    transform = transforms.ToTensor()

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128)

    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            cls_loss = criterion(outputs, labels)
            sparsity = model.sparsity_loss()

            loss = cls_loss + lambda_val * sparsity
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed")

    # ✅ After training
    acc = evaluate(model, testloader)
    sparsity_level = calculate_sparsity(model)

    # ✅ THIS CREATES THE PNG
    plot_gate_distribution(model)

    return acc, sparsity_level


if __name__ == "__main__":
    for lam in [1e-5, 1e-4, 1e-3]:
        print(f"\nTraining with Lambda = {lam}")
        acc, sp = train_model(lam)
        print(f"Lambda: {lam} | Accuracy: {acc:.2f}% | Sparsity: {sp:.2f}%")

   
    
