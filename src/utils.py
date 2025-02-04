import torch
import os
import random

from src.model import Model


def get_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    return torch.device(device)

def get_latest_model_filename(folder_path, prefix):
    return os.path.join(folder_path, max(f for f in os.listdir(folder_path) if f.endswith('.pth') and f.startswith(prefix)))

def load_saved_model(file_name, device):
    print(f"Loading model from {file_name}")

    # Initialize a new model
    model = Model().to(device)

    checkpoint = torch.load(file_name, map_location=device, weights_only=True)

    # Load both encoder and decoder states
    model.decoder.load_state_dict(checkpoint['decoder'])

    model.eval()  # Set to evaluation mode
    return model

# Get random content and style images
def get_random_images(content_dir, style_dir):
    content_paths = [
        os.path.join(content_dir, f)
        for f in os.listdir(content_dir)
        if f.lower().endswith('.jpg')
    ]

    style_paths = [
        os.path.join(style_dir, f)
        for f in os.listdir(style_dir)
        if f.lower().endswith('.jpg')
    ]

    content_path = random.choice(content_paths)
    style_path = random.choice(style_paths)

    return content_path, style_path


def count_jpegs(folder_path):
    jpeg_count = 0
    jpeg_extensions = {'.jpg', '.jpeg'}

    for filename in os.listdir(folder_path):
        # Get the file extension
        _, extension = os.path.splitext(filename)
        # If it's a JPEG file, increment counter
        if extension in jpeg_extensions:
            jpeg_count += 1

    return jpeg_count
