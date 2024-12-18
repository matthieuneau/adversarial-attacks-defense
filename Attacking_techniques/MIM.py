import torch
import torch.nn.functional as F

def mim_attack(model, images, labels, epsilon=0.3, alpha=0.01, iters=10, mu=1.0, device="cpu"):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    perturbed_images = images.clone().detach()
    
    g = torch.zeros_like(images).to(device)
    
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False

    for _ in range(iters):
        perturbed_images.requires_grad = True
        
        outputs = model(perturbed_images)
        loss = F.nll_loss(outputs, labels)

        # Debug: Loss and outputs
        #print(f"Iteration {_ + 1}/{iters}, Loss: {loss.item()}")
        
        grad = torch.autograd.grad(loss, perturbed_images)[0]
        
        # Debug: Gradient norm
        #print("Gradient norm:", grad.norm().item())

        grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1)
        grad_norm = grad_norm.view(-1, 1, 1, 1)
        grad_normalized = grad / (grad_norm + 1e-8)
        
        g = mu * g + grad_normalized
        
        # Debug: Momentum norm
        #print("Momentum norm (L∞):", torch.max(torch.abs(g)).item())
        
        with torch.no_grad():
            perturbed_images = perturbed_images + alpha * g.sign()
            delta = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon)
            perturbed_images = torch.clamp(images + delta, min=0, max=1)
        
        # Debug: Perturbation norms
        #print("Perturbation norms (L∞):", torch.max(torch.abs(perturbed_images - images)).item())
        
        perturbed_images = perturbed_images.detach()
        
    for param in model.parameters():
        param.requires_grad = True
    
    model.train()
    
    return perturbed_images