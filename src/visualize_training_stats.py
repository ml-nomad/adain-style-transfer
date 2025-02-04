import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path

checkpoint_dir = Path('../checkpoints')

def visualize_training_stats(stats_file):
    """
    Plot training progress from saved statistics file.
    Shows content loss, style loss, and total loss on the same plot.

    Args:
        stats_file (str): Path to the JSON file containing training statistics
    """
    # Load the statistics
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    # Create epochs array for x-axis
    epochs = range(1, len(stats['total_loss']) + 1)

    # Create the figure and axis
    plt.figure(figsize=(12, 6))

    # Plot all three losses
    plt.plot(epochs, stats['content_loss'], 'b-', label='Content Loss', linewidth=2, alpha=0.7)
    plt.plot(epochs, stats['style_loss'], 'r-', label='Style Loss', linewidth=2, alpha=0.7)
    plt.plot(epochs, stats['total_loss'], 'g-', label='Total Loss', linewidth=2)

    # Customize the plot
    plt.title('Training Progress Over Time', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Add some padding to the layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Print some summary statistics
    print("\nTraining Summary:")
    print(f"Total epochs: {len(epochs)}")
    print(f"Final losses - Content: {stats['content_loss'][-1]:.4f}, "
          f"Style: {stats['style_loss'][-1]:.4f}, "
          f"Total: {stats['total_loss'][-1]:.4f}")
    print(f"Training period: {stats['start_time']} to {stats['end_time']}")


def get_latest_checkpoint():
    if not checkpoint_dir.exists():
        raise FileNotFoundError("Checkpoints directory not found")

    # Get all files in the checkpoints directory
    checkpoint_files = list(checkpoint_dir.glob('*training_stats.json'))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found in directory")

    # Sort files by modification time and get the latest one
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    return latest_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a file with optional argument')
    parser.add_argument('--file', type=str, help='Input file path (optional)', default=None)

    args = parser.parse_args()

    if args.file:
        input_file = Path(args.file)
        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {args.file}")
    else:
        input_file = get_latest_checkpoint()

    print(f"Using file: {input_file}")
    visualize_training_stats(input_file)