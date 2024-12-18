from torch._prims_common import Tensor
from model import Net
import torch
from torch import nn
from torch import optim
from tqdm import tqdm


class OrthogonalNet(Net):
    def __init__(self, model_file="models/OrthogonalNet.pth"):
        super().__init__()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model_file = model_file  # Override the model_file attribute
        self.apply(self._orthogonal_init)

        # TODO: check that the layers are properly initialized

    def _orthogonal_init(self, layer):
        """
        Apply orthogonal initialization to layers with weights.
        """
        if hasattr(layer, "weight") and isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def load(self, model_file=None):
        """
        Override the load method to use the custom device and model_file.
        """
        model_file = model_file or self.model_file
        self.load_state_dict(
            torch.load(model_file, map_location=torch.device(self.device))
        )


def orthogonal_regularization(weight, lambda_ortho=1e-4) -> torch.Tensor:
    if weight.dim() < 2:
        return torch.Tensor(0.0)

    W = weight.view(weight.size(0), -1)  # Reshape for FC or Conv layers
    WT_W = torch.mm(W, W.T)
    identity = torch.eye(W.size(0)).to(weight.device)
    return lambda_ortho * torch.norm(WT_W - identity, p="fro")


def train_model(net, train_loader, pth_filename, num_epochs, device, lambda_ortho=1e-3):
    """Basic training function (from pytorch doc.)"""
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss += sum(
                orthogonal_regularization(p, lambda_ortho)
                for name, p in net.named_parameters()
                if "weight" in name and p.dim() > 1
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    net.save(pth_filename)
    print("Model saved in {}".format(pth_filename))


if __name__ == "__main__":
    import argparse
    from torchvision import transforms
    import torchvision
    from model import get_train_loader, test_natural

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    batch_size = 128
    valid_size = 1024
    train_transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10(
        "./data/", download=True, transform=train_transform
    )

    if parser.parse_args().mode == "train":
        model = OrthogonalNet().to(device)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        train_model(
            model, train_loader, model.model_file, 10, device, lambda_ortho=1e-3
        )
        # TODO: Check that conv2D weights are orthogonal
        weights = model.fc3.weight
        print(weights @ weights.T)

    else:
        model = OrthogonalNet().to(device)
        model.load(model.model_file)
        model.eval()
        print("Model loaded successfully.")
        test_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        accuracy = test_natural(model, test_loader)
        print(f"Accuracy: {accuracy}")
