import torch
import torch.nn as nn

# Implement a custom loss function in PyTorch
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
      
    def forward(self, inputs, targets):
        loss = -1 * (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        return loss.mean()
