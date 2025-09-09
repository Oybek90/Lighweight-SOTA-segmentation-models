import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dice_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    return (2 * intersection) / (preds.sum() + targets.sum() + 1e-8)

def iou_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / (union + 1e-8)

# Loss Functions (from paper)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss as used in the paper."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # Loss formulation from paper: L = 0.5*BCE(ŷ,y) + Dice(ŷ,y)
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return 0.5 * bce_loss + dice_loss
    
class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1).float()

        # Dice loss
        smooth = 1.
        inter = (preds * targets).sum()
        dice = 1 - (2*inter + smooth) / (preds.sum() + targets.sum() + smooth)

        # Focal loss
        pt = torch.where(targets == 1, preds, 1 - preds)
        focal = -self.alpha * (1 - pt)**self.gamma * torch.log(pt + 1e-8)
        focal = focal.mean()

        return dice + focal
        
class Retina_CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(Retina_CombinedLoss, self).__init__()
        self.smooth = smooth
        
    def dice_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice_score
    
    def iou_loss(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou_score = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou_score
    
    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        iou_loss = self.iou_loss(pred, target)
        return dice_loss + iou_loss
    

class LTMSegNet_Loss(nn.Module):
    def __init__(self, smooth=1e-5, bce_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
    def forward(self, pred, target):
        bce = self.bce(pred, target)
        probs = torch.sigmoid(pred)
        inter = (probs * target).sum(dim=(1,2,3))
        union = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce + dice_loss

class ESDMRLoss(nn.Module):
    """Dice Similarity Coefficient Loss as described in Equations 9-12"""
    def __init__(self, smooth=1e-6):
        super(ESDMRLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # pred: (N, 1, H, W), target: (N, 1, H, W)
        batch_size = pred.size(0)
        
        # Apply sigmoid to predictions to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Flatten for pixel-wise computation
        pred_flat = pred_prob.view(batch_size, -1)  # (N, H*W)
        target_flat = target.view(batch_size, -1)   # (N, H*W)
        
        # Calculate DSC for each image in batch (Equation 10)
        intersection = (pred_flat * target_flat).sum(dim=1)  # (N,)
        pred_sum = (pred_flat ** 2).sum(dim=1)               # (N,)
        target_sum = (target_flat ** 2).sum(dim=1)           # (N,)
        
        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Dice loss with quadratic form as in Equation 9
        dice_loss = (1 - dice) ** 2
        
        # Return mean loss across batch
        return dice_loss.mean()
    
class LWBNALoss(nn.Module):
    """Combined Dice Loss and BCE Loss"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(LWBNALoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        return self.dice_weight * dice + self.bce_weight * bce