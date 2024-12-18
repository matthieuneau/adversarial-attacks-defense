#!/usr/bin/env python3
from typing import List
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F


# Define the PGD attack
def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.001, iters=40, device="cpu"):
    """
    Perform Projected Gradient Descent (PGD) attack.

    Parameters:
        model: The neural network model to attack.
        images: Batch of images (torch.Tensor).
        labels: True labels of the images (torch.Tensor).
        epsilon: Maximum perturbation.
        alpha: Step size.
        iters: Number of iterations.
        device: Device to use (e.g., 'cuda' or 'cpu').

    Returns:
        perturbed_images: Adversarially perturbed images.
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    perturbed_images = images.clone().detach()

    perturbed_images.requires_grad = True

    for _ in range(iters):
        # Forward pass
        outputs = model(perturbed_images)
        loss = F.nll_loss(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Generate adversarial examples
        grad = perturbed_images.grad.data
        perturbed_images = perturbed_images + alpha * grad.sign()
        perturbed_images = torch.clamp(perturbed_images, 0, 1)  # Keep pixels in [0,1]
        perturbed_images = torch.max(
            torch.min(perturbed_images, images + epsilon), images - epsilon
        ).detach_()
        perturbed_images.requires_grad = True

    return perturbed_images


# Define a function for adversarial testing
def test_adversarial(
    net,
    test_loader,
    epsilon=0.1,
    alpha=0.001,
    iters=40,
    device: str | torch.device = "cpu",
):
    """
    Test model robustness against adversarial examples.

    Parameters:
        net: The model to evaluate.
        test_loader: DataLoader for the test set.
        epsilon: Maximum perturbation.
        alpha: Step size.
        iters: Number of iterations.
        device: Device to use (e.g., 'cuda' or 'cpu').

    Returns:
        Adversarial accuracy.
    """
    correct = 0
    total = 0

    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        # Generate adversarial examples
        adv_images = pgd_attack(net, images, labels, epsilon, alpha, iters, device)
        # Test adversarial examples
        outputs = net(adv_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total


# Define a function for adversarial training
def train_adversarial(
    net,
    train_loader,
    pth_filename,
    num_epochs,
    epsilon=0.3,
    alpha=0.01,
    iters=10,
    adv_training=True,
    device="cpu",
):
    """
    Train the model, optionally using adversarial examples for adversarial training.

    Parameters:
        net: The model to train.
        train_loader: DataLoader for the training data.
        pth_filename: Path to save the trained model.
        num_epochs: Number of epochs.
        epsilon: Maximum perturbation for adversarial examples.
        alpha: Step size for PGD attack.
        iters: Number of iterations for PGD attack.
        adv_training: Whether to include adversarial training.
        device: Device to use (e.g., 'cuda' or 'cpu').
    """
    print("Starting training with adversarial examples")
    criterion = F.nll_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            if adv_training:
                # Generate adversarial examples
                inputs = pgd_attack(
                    net,
                    inputs,
                    labels,
                    epsilon=epsilon,
                    alpha=alpha,
                    iters=iters,
                    device=device,
                )

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 500 == 499:  # Print every 500 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 500:.3f}")
                running_loss = 0.0

    # Save the trained model
    torch.save(net.state_dict(), pth_filename)
    print(f"Model saved in {pth_filename}")


def pgd_experiment(
    models: List[nn.Module],
    epsilons: List[float],
    test_loader: DataLoader,
    device: str | torch.device,
):
    """
    Test multiple models on adversarial examples generated using PGD attack.
    Parameters:
    - models: List of neural networks to test.
    - epsilons: List of perturbation factors.
    - test_loader: DataLoader for the test dataset.
    - device: Device to run on ('cpu' or 'cuda').
    Returns:
    - accuracies: List of accuracies of the models on the adversarial examples.
    """
    # Load models
    for model in models:
        model.load(model.model_file)
        model.to(device)

    accuracies = [
        [
            test_adversarial(model, test_loader, epsilon, device=device)
            for epsilon in epsilons
        ]
        for model in models
    ]

    for model, model_accuracies in zip(models, accuracies):
        plt.plot(epsilons, model_accuracies, label=f"Model {models.index(model) + 1}")

    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy in %")
    plt.title("Accuracy vs Epsilon")
    plt.legend([model.__class__.__name__ for model in models])
    plt.savefig("overall-comparison-pgd.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    from spectralNormalization import SpectralNormNet
    from model import Net, get_validation_loader

    from orthogonalRegularization import OrthogonalNet

    models = [Net(), SpectralNormNet(), OrthogonalNet()]
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    valid_size = 1024
    batch_size = 128
    epsilons = [0.005, 0.01, 0.02, 0.05, 0.1]

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        "./data/", download=True, transform=transform
    )
    test_loader = get_validation_loader(dataset, valid_size, batch_size)

    pgd_experiment(models, epsilons, test_loader, device)
