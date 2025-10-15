import torch
from torch import nn


class QPPLoss():
    def __init__(self,device):
        self.device = device
    def loss(self, y_pred: torch.Tensor, y_true: [torch.Tensor]) -> torch.Tensor:
        qpp_loss = nn.MSELoss()(y_pred,torch.Tensor(y_true).to(self.device))
        return (qpp_loss)