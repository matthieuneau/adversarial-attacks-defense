#!/usr/bin/env python3
import torch
import torchvision
import torch.nn.functional as F
from model import Net, get_train_loader, get_validation_loader, test_natural
import fgsm, MIM, PGD

# Ensure GPU is used if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def multiattack_adversarial_train(net, train_loader, pth_filename, attacks, num_epochs, device='cpu'):
    criterion = F.nll_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            # Apply attacks cyclically
            attack_fn, attack_params = attacks[i % len(attacks)]
            inputs = attack_fn(net, inputs, labels, device=device, **attack_params)

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update loss statistics
            running_loss += loss.item()
            if i % 500 == 499:  # Print every 500 mini-batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 500:.3f}")
                running_loss = 0.0

        # Step the learning rate scheduler
        scheduler.step()
        print(f"Epoch {epoch + 1}/{num_epochs} completed, learning rate adjusted")

    # Save the trained model
    torch.save(net.state_dict(), pth_filename)
    print(f"Adversarially trained model saved at {pth_filename}")

def main():
    net = Net()
    net.to(device)

    train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
    train_loader = get_train_loader(cifar, valid_size=1024, batch_size=32)

    attacks = [
        (fgsm.fgsm_attack, {'epsilon': 0.01}),
        (MIM.mim_attack, {'epsilon': 0.01, 'alpha': 0.01, 'iters': 10, 'mu': 1.0}),
        (PGD.pgd_attack, {'epsilon': 0.05, 'alpha': 0.01, 'iters': 40})
    ]

    model_file = "models/multiattack_model.pth"
    num_epochs = 40
    multiattack_adversarial_train(net, train_loader, model_file, attacks, num_epochs, device)

if __name__ == "__main__":
    main()
