import torch
import numpy as np
from matplotlib import pyplot as plt


def fgsm_attack(image, epsilon, data_grad):
    perturbed_image = image + epsilon * data_grad.sign()
    # Clamp the perturbed image to [0, 1] range to ensure valid pixel values
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def plot_adversarial_examples(adv_examples, true_labels, adv_labels):
    """
    Display adversarial examples with true and adversarial labels in a square grid.

    Parameters:
    - adv_examples: List of adversarial images (numpy arrays or tensors).
    - true_labels: List of true labels corresponding to the adversarial examples.
    - adv_labels: List of labels predicted by the model for the adversarial examples.
    """
    num_examples = len(adv_examples)

    # Create a square grid automatically
    fig, axes = plt.subplots(
        int(np.ceil(np.sqrt(num_examples))),  # Rows
        int(np.ceil(np.sqrt(num_examples))),  # Columns
        figsize=(10, 10),
    )
    axes = axes.flatten()  # Flatten the grid to iterate easily

    for i, ax in enumerate(axes):
        if i < num_examples:  # Check if there's an example to display
            # Convert image to numpy and transpose for display
            img = (
                adv_examples[i].transpose(1, 2, 0)
                if isinstance(adv_examples[i], np.ndarray)
                else adv_examples[i].permute(1, 2, 0).cpu().numpy()
            )
            ax.imshow(img)
            ax.set_title(f"True: {true_labels[i]}\nAdv: {adv_labels[i]}", fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")  # Hide unused axes

    plt.tight_layout()
    plt.show()
