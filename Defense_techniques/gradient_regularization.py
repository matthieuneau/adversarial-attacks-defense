#!/usr/bin/env python3
import argparse
import importlib
import os
import torch
import torchvision
from model import Net, get_train_loader, get_validation_loader, test_natural

'''
python3 gradient_regularization.py --model-file models/PGD_model.pth \
--attack-file PGD.py --attack-fn pgd_attack --num-epochs 20 \
--attack-params epsilon=0.3,alpha=0.01,iters=10 --lambda-reg 0.1


python3 gradient_regularization.py --model-file models/FGSM_model.pth \
--attack-file fgsm.py --attack-fn fgsm_attack --num-epochs 20 \
--attack-params epsilon=0.1 --lambda-reg 0.1


python3 gradient_regularization.py --model-file models/MIM_model.pth \
--attack-file MIM.py --attack-fn mim_attack --num-epochs 20 \
--attack-params epsilon=0.01,alpha=0.01,iters=10,mu=1.0 --lambda-reg 0.1
'''

# Ensure GPU is used if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train_with_regularization_and_attack(
    net, train_loader, pth_filename, attack_fn, num_epochs, adv_training=True, lambda_reg=0.1, **attack_params
):
    """
    Train the model using both adversarial training and gradient regularization.

    Parameters:
        net: The model to train.
        train_loader: DataLoader for the training data.
        pth_filename: Path to save the trained model.
        attack_fn: Function to generate adversarial examples.
        num_epochs: Number of epochs.
        adv_training: Whether to include adversarial training.
        lambda_reg: Coefficient for gradient regularization.
        attack_params: Parameters for the attack function.
    """
    print("Training with adversarial examples and gradient regularization.")
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero parameter gradients
            optimizer.zero_grad()

            # Generate adversarial examples
            if adv_training:
                inputs = attack_fn(net, inputs, labels, device=device, **attack_params)

            # Forward pass
            outputs = net(inputs)
            task_loss = criterion(outputs, labels)

            # Gradient regularization
            inputs.requires_grad = True
            task_loss.backward(retain_graph=True)
            input_grads = inputs.grad
            grad_penalty = (input_grads.view(input_grads.size(0), -1).norm(2, dim=1) ** 2).mean()

            # Combine task loss and gradient penalty
            total_loss = task_loss + lambda_reg * grad_penalty

            # Backward pass and optimize
            optimizer.zero_grad()  # Reset gradients
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            if i % 500 == 499:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 500:.3f}")
                running_loss = 0.0

        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs} completed, learning rate adjusted.")

    # Save the trained model
    net.save(pth_filename)
    print(f"Adversarially trained model with gradient regularization saved at {pth_filename}")


def test_with_attack(net, test_loader, attack_fn, **attack_params):
    """
    Test model robustness against adversarial examples using a generic attack function.

    Parameters:
        net: The model to evaluate.
        test_loader: DataLoader for the test set.
        attack_fn: Function to generate adversarial examples.
        attack_params: Parameters for the attack function.

    Returns:
        Adversarial accuracy.
    """
    correct = 0
    total = 0

    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)

        # Generate adversarial examples
        adv_images = attack_fn(net, images, labels, device=device, **attack_params)

        # Test adversarial examples
        outputs = net(adv_images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default="models/gradient_regularized_model.pth", help="Path to save/load the model weights.")
    parser.add_argument("--attack-file", required=True, help="Python file containing the attack function.")
    parser.add_argument("--attack-fn", required=True, help="Name of the attack function to use.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--lambda-reg", type=float, default=0.1, help="Regularization strength for gradient penalty.")
    parser.add_argument("--force-train", action="store_true", help="Force retraining even if model file exists.")
    parser.add_argument("--attack-params", type=str, default="", help="Comma-separated list of attack parameters (key=value).")
    args = parser.parse_args()

    # Parse attack parameters
    attack_params = {}
    if args.attack_params:
        for param in args.attack_params.split(","):
            key, value = param.split("=")
            attack_params[key] = float(value) if "." in value else int(value)

    # Dynamically import the attack function
    attack_module = importlib.import_module(args.attack_file.replace(".py", ""))
    attack_fn = getattr(attack_module, args.attack_fn)

    # Create model and move it to the appropriate device
    net = Net()
    net.to(device)

    # Train the model
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model with gradient regularization and adversarial examples.")
        print(f"Saving model to {args.model_file}")

        train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size=1024, batch_size=32)

        train_with_regularization_and_attack(
            net, train_loader, args.model_file, attack_fn, args.num_epochs, adv_training=True, lambda_reg=args.lambda_reg, **attack_params
        )

    # Validate the model
    print(f"Testing trained model from '{args.model_file}'.")

    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=torchvision.transforms.ToTensor())
    valid_loader = get_validation_loader(cifar, valid_size=1024)

    net.load(args.model_file)
    natural_acc = test_natural(net, valid_loader)
    print(f"Natural accuracy: {natural_acc:.2f}%")

    adversarial_acc = test_with_attack(net, valid_loader, attack_fn, **attack_params)
    print(f"Adversarial accuracy using {args.attack_fn}: {adversarial_acc:.2f}%")


if __name__ == "__main__":
    main()
