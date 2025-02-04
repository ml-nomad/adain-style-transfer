from datetime import datetime
import random
import os
import string
import torch
import argparse

from src.train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description='Train AdaIN style transfer model')

    # Add all hyperparameters as command line arguments
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='training batch size (default: 8)')
    parser.add_argument('--base-lr', type=float, default=1e-3,
                        help='base learning rate (default: 0.001)')
    parser.add_argument('--style-loss-coeff', type=float, default=1.0,
                        help='style loss coefficient (default: 1.0)')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints',
                        help='directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--content-dir', type=str, default='training_data/content',
                        help='content images directory (default: training_data/content)')
    parser.add_argument('--style-dir', type=str, default='training_data/style',
                        help='style images directory (default: training_data/style)')

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    timestamp = datetime.now().strftime('%m%d_%H%M')
    prefix = ''.join(random.choices(string.ascii_lowercase, k=6))
    training_prefix = f"{timestamp}_{prefix}"

    os.makedirs(args.checkpoints_dir, exist_ok=True)

    model = train_model(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        save_dir=args.checkpoints_dir,
        training_prefix=training_prefix,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        style_loss_coeff=args.style_loss_coeff,
        log_interval=100
    )

    complete_model_path = os.path.join(args.checkpoints_dir, f'{training_prefix}_complete_adain_model.pth')

    torch.save({
        'decoder': model.decoder.state_dict()
    }, complete_model_path)