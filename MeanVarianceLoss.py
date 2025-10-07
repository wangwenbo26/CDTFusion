import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalMeanVarianceLoss(nn.Module):
    def __init__(self, window_size):
        super(LocalMeanVarianceLoss, self).__init__()
        self.window_size = window_size

    def forward(self, feature1, feature2):
       mean1 = F.avg_pool2d(feature1, self.window_size, stride=1, padding=self.window_size // 2)
       mean2 = F.avg_pool2d(feature2, self.window_size, stride=1, padding=self.window_size // 2)
       variance1 = F.avg_pool2d(feature1**2, self.window_size, stride=1, padding=self.window_size // 2) - mean1**2
       variance2 = F.avg_pool2d(feature2**2, self.window_size, stride=1, padding=self.window_size // 2) - mean2**2

       mean_loss = torch.mean((mean1 - mean2)**2)
       variance_loss = torch.mean((variance1 - variance2)**2)

       return mean_loss, variance_loss