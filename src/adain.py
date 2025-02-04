import torch.nn as nn

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""

    def __init__(self):
        super().__init__()

    def forward(self, content, style):
        # Compute mean and std for both features
        content_mean, content_std = self.calc_stats(content)
        style_mean, style_std = self.calc_stats(style)

        # Normalize content features
        normalized = (content - content_mean) / content_std

        # Denormalize with style statistics
        return normalized * style_std + style_mean

    def calc_stats(self, x):
        """Calculate channel-wise mean and standard deviation"""
        b, c = x.shape[:2]
        x_flat = x.view(b, c, -1)
        mean = x_flat.mean(dim=2).view(b, c, 1, 1)
        std = x_flat.std(dim=2).view(b, c, 1, 1) + 1e-8
        return mean, std
