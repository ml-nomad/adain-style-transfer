import matplotlib.pyplot as plt
import time
import torch
from torchvision import transforms
from PIL import Image
import os

def transfer_style(
        model,
        content_path,
        style_path,
        device,
        alpha=1.0,
        output_path=None,
        show_plot=True
):
    """
    Performs style transfer and either display results in a plot, save to file, or both.

    Args:
        model: Style transfer model
        content_path: Path to content image
        style_path: Path to style image
        device: Device to run the model on
        alpha: Style interpolation weight
        output_path: If provided, save the output image to this path
        show_plot: If True, display the results in a plot

    Returns:
        PIL Image of the stylized result
    """
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')

    output_img = transfer_style_img(model, content_img, style_img, device, alpha)

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_img.save(output_path)

    # Show plot if requested
    if show_plot:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(content_img)
        plt.title('Content Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(style_img)
        plt.title('Style Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(output_img)
        plt.title('Generated Image')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

def transfer_style_img(
        model,
        content_img,
        style_img,
        device,
        alpha
):
    """
    Performs style transfer of provided images

    Args:
        model: Style transfer model
        content_img: content image
        style_img: style image
        device: Device to run the model on
        alpha: Style interpolation weight

    Returns:
        result image
    """

    # Set model to eval mode
    model.eval()

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    content_tensor = transform(content_img).unsqueeze(0).to(device)
    style_tensor = transform(style_img).unsqueeze(0).to(device)

    # Inverse transform for visualization
    inverse_transform = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
    ])

    # Generate styled image
    with torch.no_grad():
        output, _ = model(content_tensor, style_tensor, alpha=alpha)

    # Convert tensor to image
    def tensor_to_image(tensor):
        img = inverse_transform(tensor.squeeze().cpu())
        img = torch.clamp(img, 0, 1)
        return transforms.ToPILImage()(img)

    # Get output image
    output_img = tensor_to_image(output)

    return output_img