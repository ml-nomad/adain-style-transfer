import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from src.utils import get_device, load_saved_model

def process_folder(folder_path, model, device):
    # Get all image files
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_files) < 2:
        print("Need at least 2 images in the folder")
        return

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Inverse transform for visualization
    inverse_transform = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
    ])

    def resize_for_display(img):
        """Resize image keeping aspect ratio with max dimension of 100px"""
        w, h = img.size
        if w > h:
            new_w = 100
            new_h = int(h * (100 / w))
        else:
            new_h = 100
            new_w = int(w * (100 / h))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def tensor_to_image(tensor):
        img = inverse_transform(tensor.squeeze().cpu())
        img = torch.clamp(img, 0, 1)
        return transforms.ToPILImage()(img)

    # Calculate total number of combinations
    total_combinations = len(image_files) * (len(image_files) - 1)

    # Create figure
    fig = plt.figure(figsize=(15, 3 * total_combinations))

    model.eval()
    row = 0

    # Process each content image
    for i, content_file in enumerate(image_files):
        content_path = os.path.join(folder_path, content_file)
        content_img = Image.open(content_path).convert('RGB')
        content_tensor = transform(content_img).unsqueeze(0).to(device)

        # Process against each style image
        for j, style_file in enumerate(image_files):
            if i != j:  # Skip same image
                style_path = os.path.join(folder_path, style_file)
                style_img = Image.open(style_path).convert('RGB')
                style_tensor = transform(style_img).unsqueeze(0).to(device)

                # Generate styled image
                with torch.no_grad():
                    output, _ = model(content_tensor, style_tensor)

                # Get output image
                output_img = tensor_to_image(output)

                # Resize all images for display
                content_display = resize_for_display(content_img)
                style_display = resize_for_display(style_img)
                output_display = resize_for_display(output_img)

                # Plot row
                row += 1

                # Content image
                plt.subplot(total_combinations, 3, (row - 1) * 3 + 1)
                plt.imshow(content_display)
                plt.title(f'Content: {content_file}' if row % 3 == 1 else '')
                plt.axis('off')

                # Style image
                plt.subplot(total_combinations, 3, (row - 1) * 3 + 2)
                plt.imshow(style_display)
                plt.title(f'Style: {style_file}' if row % 3 == 1 else '')
                plt.axis('off')

                # Result image
                plt.subplot(total_combinations, 3, (row - 1) * 3 + 3)
                plt.imshow(output_display)
                plt.title('Result' if row % 3 == 1 else '')
                plt.axis('off')

    plt.tight_layout()
    plt.show()

    return fig  # Return figure in case you want to save it

if __name__ == '__main__':
    device = get_device()
    model = load_saved_model('../checkpoints/0206_0017_azwkma_complete_adain_model.pth', device)

    fig = process_folder('../test_data', model, device)
    fig.savefig('style_transfer_results.png', bbox_inches='tight', dpi=300)