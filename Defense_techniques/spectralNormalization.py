import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.utils import spectral_norm
from torchvision import transforms

from model import (
    get_train_loader,
    test_natural,
    train_model,
)


class SpectralNormNet(nn.Module):
    model_file = "models/spectral_norm.pth"
    """this file will be loaded to test your model. use --model-file to load/store a different model."""

    def __init__(self):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(3, 6, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = spectral_norm(nn.Conv2d(6, 16, 5))
        self.fc1 = spectral_norm(nn.Linear(16 * 5 * 5, 120))
        self.fc2 = spectral_norm(nn.Linear(120, 84))
        self.fc3 = spectral_norm(nn.Linear(84, 10))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        """Helper function, use it to save the model weights after training."""
        torch.save(self.state_dict(), model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    batch_size = 32
    valid_size = 1024
    train_transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10(
        "./data/", download=True, transform=train_transform
    )

    if parser.parse_args().mode == "train":
        model = SpectralNormNet().to(device)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        train_model(model, train_loader, "models/spectral_norm.pth", 10)
        torch.save(model.state_dict(), "models/spectral_norm.pth")

    else:
        model = SpectralNormNet().to(device)
        model.load_state_dict(torch.load("models/spectral_norm.pth"))
        model.eval()
        print("Model loaded successfully.")
        test_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        accuracy = test_natural(model, test_loader)
        print(f"Accuracy: {accuracy}")
