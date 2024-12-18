# def generate_adversarial_examples(model, test_loader, epsilon):
#     """
#     Generate adversarial examples using the Fast Gradient Sign Method (FGSM).
#     Parameters:
#     - model: PyTorch model to attack.
#     - test_loader: DataLoader for the test dataset.
#     - epsilon: Perturbation value for the FGSM attack.

#     Returns:
#     - adv_examples: List of adversarial examples.
#     - true_labels: List of true labels corresponding to the adversarial examples
#     - adv_labels: List of labels predicted by the model for the adversarial examples
#     - final_acc: Final accuracy of the model on the adversarial examples.
#     """
#     correct_overall = 0
#     correct_adv = 0
#     adv_examples = []
#     true_labels = []
#     adv_labels = []

#     for data, target in tqdm(test_loader):
#         data = data.requires_grad_()

#         output = model(data)
#         init_pred = output.max(1, keepdim=True)[1]

#         correct_indices = (init_pred.squeeze() == target).nonzero(as_tuple=True)[0]
#         # If no correct predictions in the batch, skip.
#         if len(correct_indices) == 0:
#             continue
#         correct_data = data[correct_indices]
#         correct_target = target[correct_indices]

#         correct_overall += len(correct_indices)

#         loss = torch.nn.CrossEntropyLoss()(output, target)

#         model.zero_grad()

#         # Backward pass to calculate gradients of the loss with respect to the input image
#         loss.backward()
#         data_grad = data.grad.data

#         adversarial_data = fgsm_attack(data, epsilon, data_grad)
#         adversarial_data = adversarial_data[correct_indices]

#         output = model(adversarial_data)
#         final_pred = output.max(1, keepdim=True)[1]

#         target = target[correct_indices]

#         correct_adv += (final_pred.squeeze() == target).sum().item()

#         # Store adversarial examples for visualization
#         for i in range(len(correct_data)):
#             if len(adv_examples) < 10:  # Limit stored examples
#                 adv_ex = adversarial_data[i].squeeze().detach().cpu().numpy()
#                 adv_examples.append(adv_ex)
#                 true_labels.append(target[i].item())
#                 adv_labels.append(final_pred[i].item())

#     accuracy = correct_adv / correct_overall
#     print(f"Epsilon: {epsilon}\tTest Accuracy = {accuracy * 100:.2f}%")

#     return accuracy, adv_examples, true_labels, adv_labels
