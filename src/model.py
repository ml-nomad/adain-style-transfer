from src.adain import AdaIN
from src.decoder import Decoder
from src.encoder import (VGGEncoder)

import torch.nn as nn


class Model(nn.Module):
    """
    AdaIN-based Style Transfer Model
    Architecture:
    - Pretrained VGG-19 Encoder (up to conv4_1) with reflection padding
    - Adaptive Instance Normalization (AdaIN) layer
    - Trainable Decoder
    """

    def __init__(self):
        super().__init__()

        # Load pretrained VGG-19 and create encoder
        self.encoder = VGGEncoder()

        # AdaIN layer (stateless, no trainable parameters)
        self.adain = AdaIN()

        # Trainable decoder
        self.decoder = Decoder()

    def forward(self, content, style, alpha=1.0):
        # Extract features
        content_feat = self.encoder(content, output_style_features=False)
        style_feat = self.encoder(style, output_style_features=False)

        # Apply AdaIN
        adain_feat = self.adain(content_feat, style_feat)

        # Interpolate AdaIN output with content features
        t = alpha * adain_feat + (1 - alpha) * content_feat

        # Decode to image space
        output = self.decoder(t)

        # Return all three needed values
        return output, t
