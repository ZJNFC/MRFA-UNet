import torch
from torch import nn

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75,
                 smooth: float = 1e-6, from_logits: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):

        targets = targets.float()
        if self.from_logits:
            preds = torch.sigmoid(preds)
        preds = preds.clamp(min=0., max=1.)

        dims = tuple(range(2, preds.ndim))
        tp = torch.sum(preds * targets, dim=dims)
        fp = torch.sum(preds * (1 - targets), dim=dims)
        fn = torch.sum((1 - preds) * targets, dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = torch.pow((1 - tversky), self.gamma)
        return loss.mean()

