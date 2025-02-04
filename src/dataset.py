
import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class StyleContentDataset(Dataset):
    """
    Dataset for loading content and style images for style transfer training.
    It needs to get preliminarily prepared files. Files must have smallest dimension of 512 pixels.
    Use prepare_images.py process raw images
    """
    def __init__(self, content_dir, style_dir, image_size=224):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.image_size = image_size

        # Load content image paths
        self.content_paths = [
            os.path.join(content_dir, f)
            for f in os.listdir(content_dir)
            if f.lower().endswith('.jpg')
        ]

        # Load style image paths
        self.style_paths = [
            os.path.join(style_dir, f)
            for f in os.listdir(style_dir)
            if f.lower().endswith('.jpg')
        ]

        # Create shuffled indices for both content and style
        self.reshuffle()

        # Define transforms
        self.transform = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def reshuffle(self):
        """Reshuffle both content and style indices"""
        self.content_indices = torch.randperm(len(self.content_paths))
        self.style_indices = torch.randperm(len(self.style_paths))

    def __len__(self):
        # Return minimum length to ensure paired iteration
        return min(len(self.content_paths), len(self.style_paths))

    def __getitem__(self, idx):
        # Use modulo to wrap around indices if needed
        content_idx = self.content_indices[idx % len(self.content_paths)]
        style_idx = self.style_indices[idx % len(self.style_paths)]

        content_path = self.content_paths[content_idx]
        style_path = self.style_paths[style_idx]

        # Load and transform images
        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')

        content_img = self.transform(content_img)
        style_img = self.transform(style_img)

        # Verify numerical validity
        assert not torch.isnan(content_img).any(), f"NaN in content image: {content_path}"
        assert not torch.isnan(style_img).any(), f"NaN in style image: {style_path}"

        return content_img, style_img

def create_dataloaders(content_dir, style_dir, batch_size):
    dataset = StyleContentDataset(content_dir, style_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # for MPS compatibility
        pin_memory=True,
        persistent_workers=False,  # for MPS compatibility
        drop_last=True
    )

