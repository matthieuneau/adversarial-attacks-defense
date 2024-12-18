from torch import nn
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from typing import List
from torch.utils.data import DataLoader

from model import test_natural


def fgsm_attack(model, images, labels, epsilon, device):
    """
    Perform FGSM attack on the model.

    Parameters:
    - model: Neural network to attack.
    - images: Input images.
    - labels: True labels for the images.
    - epsilon: Perturbation factor.
    - device: Device to run on ('cpu' or 'cuda').

    Returns:
    - perturbed_images: Adversarially perturbed images.
    """
    images = images.clone().detach().to(device).requires_grad_()
    labels = labels.clone().detach().to(device)

    # Forward pass
    outputs = model(images)
    loss = F.nll_loss(outputs, labels)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Generate adversarial images
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()

    # Clamp the perturbed images to maintain valid pixel range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images


def test_fgsm_on_model(model, epsilon, test_loader, device: str | torch.device):
    """
    Test model on adversarial examples generated using FGSM attack.
    Parameters:
    - model: Neural network to test.
    - epsilon: Perturbation factor.
    - test_loader: DataLoader for the test dataset.
    - device: Device to run on ('cpu' or 'cuda').
    Returns:
    - accuracy: Accuracy of the model on the adversarial examples.
    """
    correct = 0
    total = 0
    for images, labels in test_loader:
        adv_images = fgsm_attack(model, images, labels, epsilon, device)
        accuracy = test_natural(model, [(adv_images, labels)])
        correct += accuracy * len(images)
        total += len(images)
    accuracy = correct / total
    return accuracy


def fgsm_experiment(
    models: List[nn.Module],
    epsilons: List[float],
    test_loader: DataLoader,
    device: str | torch.device,
):
    """
    Test multiple models on adversarial examples generated using FGSM attack.
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
            test_fgsm_on_model(model, epsilon, test_loader, device)
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
    plt.savefig("overall comparison.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    from model import Net, get_validation_loader
    from spectralNormalization import SpectralNormNet
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

    fgsm_experiment(models, epsilons, test_loader, device)
