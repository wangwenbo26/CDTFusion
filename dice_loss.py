import torch
def dice_loss(predicted, target, num_classes=15):
    dice_scores = torch.zeros(num_classes)
    for class_idx in range(num_classes):
        predicted_class = predicted[:, class_idx, ...]
        target_class = (target == class_idx).float()
        intersection = torch.sum(predicted_class * target_class)
        union = torch.sum(predicted_class) + torch.sum(target_class)
        dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
        dice_scores[class_idx] = dice
    return 1.0 - dice_scores.mean()