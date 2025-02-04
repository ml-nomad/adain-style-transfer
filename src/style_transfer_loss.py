import torch.nn as nn
import torch.nn.functional as F

from src.encoder import VGGEncoder

class StyleTransferLoss(nn.Module):
    def __init__(self, style_coeff):
        super().__init__()
        self.vgg = VGGEncoder()
        self.style_coeff = style_coeff

    def calc_stats(self, x):
        b, c = x.size()[:2]
        x_flat = x.view(b, c, -1)
        mean = x_flat.mean(dim=2, keepdim=True).unsqueeze(-1)
        std = x_flat.std(dim=2, keepdim=True).unsqueeze(-1) + 1e-6
        return mean, std

    def forward(self, generated, adain, style_img):

        # Content eature extraction
        gen_content_features = self.vgg(generated, output_style_features=False)

        # Content loss - up to conv4_1
        content_loss = F.mse_loss(gen_content_features, adain)

        # Style feature extraction
        gen_style_features = self.vgg(generated, output_style_features=True)
        style_features = self.vgg(style_img, output_style_features=True)

        # Style loss (multi-layer statistics from relu1_1, relu2_1, relu3_1 and relu4_1)
        style_loss = 0
        for gen, style in zip(gen_style_features, style_features):
            gen_mean, gen_std = self.calc_stats(gen)
            style_mean, style_std = self.calc_stats(style)
            style_loss += F.mse_loss(gen_mean, style_mean) + F.mse_loss(gen_std, style_std)

        # Total loss with adjusted weights
        total_loss = content_loss + self.style_coeff * style_loss

        return total_loss, content_loss.detach(), style_loss.detach()