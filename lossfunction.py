import torch
import torch.nn as nn


class CosSimLoss(nn.Module):
    def __init__(self):
        super(CosSimLoss, self).__init__()

    def forward(self, vector_feature1, vector_feature2, label):
        loss = torch.dot(vector_feature1, vector_feature2)/(torch.norm(vector_feature1) * torch.norm(vector_feature2) + 1e-6)
        loss = (loss - label) ** 2.0
        return loss
