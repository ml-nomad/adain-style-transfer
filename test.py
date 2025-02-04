import argparse
from src.utils import get_device, load_saved_model
from src.transfer_style import transfer_style


def parse_args():
    parser = argparse.ArgumentParser(description='Transfer style between images')

    # Add command line arguments with current values as defaults
    parser.add_argument('--content-path', type=str, default='test_data/brad_pitt.jpg',
                        help='path to content image')
    parser.add_argument('--style-path', type=str, default='test_data/sketch.png',
                        help='path to style image')
    parser.add_argument('--model', type=str, default='model/model.pth',
                        help='path to model file')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='style interpolation value (default: 1.0)')
    parser.add_argument('--no-show-plot', action='store_true',
                        help='disable plot display')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set up device and model
    device = get_device()
    model = load_saved_model(args.model, device)

    # Transfer style
    transfer_style(
        model=model,
        content_path=args.content_path,
        style_path=args.style_path,
        device=device,
        alpha=args.alpha,
        show_plot=not args.no_show_plot
    )