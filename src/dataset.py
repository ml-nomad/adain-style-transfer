import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class StyleContentDataset(Dataset):
    """
    Dataset for loading content and style images for style transfer training.
    Automatically reshuffles when reaching the end of the smallest dataset.
    """

    def __init__(self, content_dir, style_dir):
        self.content_dir = content_dir
        self.style_dir = style_dir

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

        # Initialize counters and create initial shuffled indices
        self.current_index = 0
        self.dataset_size = min(len(self.content_paths), len(self.style_paths))
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
        self.current_index = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Check if we need to reshuffle
        if self.current_index >= self.dataset_size:
            self.reshuffle()

        # Get indices for this item
        content_idx = self.content_indices[self.current_index]
        style_idx = self.style_indices[self.current_index]

        # Increment counter
        self.current_index += 1

        # Load images
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

