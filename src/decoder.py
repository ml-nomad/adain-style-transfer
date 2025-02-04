import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder network that mirrors VGG-19 encoder structure (up to relu4_1).
    """

    def __init__(self):
        super().__init__()

        # Initialize decoder layers to mirror VGG encoder
        # All conv layers have corresponding reflection padding

        # Starting from relu4_1 features (512 channels)
        self.decoder = nn.Sequential(
            # Block 4 - Corresponding to VGG conv4_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            # Upsample 1 - Corresponding to VGG maxpool3
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Block 3 - Corresponding to VGG conv3_4, conv3_3, conv3_2, conv3_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            # Upsample 2 - Corresponding to VGG maxpool2
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Block 2 - Corresponding to VGG conv2_2, conv2_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),

            # Upsample 2 - Corresponding to VGG maxpool2
            nn.Upsample(scale_factor=2, mode='nearest'),

            # Block 1 - Corresponding to VGG conv1_2, conv1_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, x):
        return self.decoder(x)
