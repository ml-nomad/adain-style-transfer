from datetime import datetime
import json
import logging
import os
import time
import torch

from src.utils import get_device, count_jpegs
from src.dataset import create_dataloaders
from src.model import Model
from src.transfer_style import transfer_style
from src.style_transfer_loss import StyleTransferLoss

def setup_logging(save_dir, prefix):
    """Setup logging configuration"""
    log_file = os.path.join(save_dir, f'{prefix}_training_log.log')

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Setup stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    # Setup logger
    logger = logging.getLogger('style_transfer')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_training_stats(stats, stats_file):
    """Save training statistics to JSON file"""

    # Convert tensor values to floats for JSON serialization
    serializable_stats = {}
    for key, value in stats.items():
        if isinstance(value, list) and len(value) > 0:
            # Convert any torch tensors to float
            serializable_stats[key] = [float(v) if torch.is_tensor(v) else v for v in value]
        else:
            serializable_stats[key] = value

    with open(stats_file, 'w') as f:
        json.dump(serializable_stats, f, indent=4)


def should_stop_training(stats, patience=5, min_improvement=0.0001, window_size=20, lookback_windows=10):
    """
    Determine if training should be stopped based on recent loss statistics.

    Args:
        stats: Dictionary containing training statistics including 'total_loss'
        patience: Number of recent windows to wait before stopping
        min_improvement: Minimum required improvement in loss between windows
        window_size: Size of each window for calculating average loss
        lookback_windows: Number of recent windows to consider for plateau detection

    Returns:
        bool: True if training should stop, False otherwise
        str: Reason for stopping if True, empty string otherwise
    """
    total_losses = stats['total_loss']

    # Need enough epochs for analysis
    if len(total_losses) < window_size * 2:
        return False, ""

    # Calculate average losses for consecutive windows
    windows = []
    for i in range(len(total_losses) - window_size + 1):
        window_avg = sum(total_losses[i:i + window_size]) / window_size
        windows.append(window_avg)

    # Only look at the most recent windows for plateau detection
    recent_windows = windows[-lookback_windows:] if len(windows) > lookback_windows else windows
    stagnant_count = 0

    # Compare consecutive recent windows
    for i in range(len(recent_windows) - 1):
        current_window = recent_windows[i]
        next_window = recent_windows[i + 1]

        # Calculate relative improvement
        improvement = (current_window - next_window) / current_window

        if improvement < min_improvement:
            stagnant_count += 1
        else:
            stagnant_count = 0  # Reset only for recent windows

        # Check if we've had too many stagnant windows recently
        if stagnant_count >= patience:
            return True, f"Training has plateaued for {patience} recent windows with improvement less than {min_improvement}"

    # Check if loss is increasing in recent windows
    if len(recent_windows) >= 2:
        recent_trend = (recent_windows[-1] - recent_windows[-2]) / recent_windows[-2]
        if recent_trend > 0.01:  # Loss increased by more than 1%
            return True, "Loss is increasing in recent windows, possible overfitting"

    return False, ""


def find_latest_checkpoint(save_dir, training_prefix):
    """
    Find the latest checkpoint in the save directory using file modification time
    Returns checkpoint path, epoch number, and batch index
    """
    checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith(training_prefix) and f.endswith('.pth')]

    if not checkpoint_files:
        return None, -1, -1

    # Get the latest file by modification time
    latest_file = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(save_dir, f)))
    latest_path = os.path.join(save_dir, latest_file)

    return latest_path

def train_model(
        content_dir,
        style_dir,
        save_dir,
        training_prefix,
        num_iterations,
        batch_size,
        base_lr,
        style_loss_coeff,
        log_interval,
        resume_training=False
):
    device = get_device()

    train_loader = create_dataloaders(content_dir, style_dir, batch_size)
    loss_fn = StyleTransferLoss(style_coeff=style_loss_coeff).to(device)
    model = Model().to(device)

    # Setup logging
    stats_file = os.path.join(save_dir, f'{training_prefix}_training_stats.json')
    logger = setup_logging(save_dir, training_prefix)

    # Training monitor setup
    monitor = TrainingMonitor(
        content_path='test_data/brad_pitt.jpg',
        style_path='test_data/vg.jpg',
        save_dir=save_dir,
        prefix=training_prefix
    )

    optimizer = torch.optim.Adam(
        model.decoder.parameters(),
        lr=base_lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=5,
        min_lr=5e-6,
        cooldown=2
    )

    # Training stats setup
    start_iter = 0
    stats = {
        'total_loss': [],
        'content_loss': [],
        'style_loss': [],
        'learning_rates': [],
        'times': [],
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    if resume_training:
        latest_checkpoint_path = find_latest_checkpoint(save_dir, training_prefix)
        if latest_checkpoint_path:
            print(f"Resuming from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path)

            model.decoder.load_state_dict(checkpoint['decoder'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            stats = checkpoint['stats']
            start_iter = checkpoint.get('iteration', 0)
            print(f"Resuming from iteration {start_iter}")
        else:
            print("No checkpoint found, starting from scratch")

    logger.info(f"{'Resuming' if resume_training else 'Starting'} training at iteration {start_iter}")
    logger.info(f"Training parameters: iterations={num_iterations}, batch_size={batch_size}")
    content_file_num = count_jpegs(content_dir)
    style_file_num = count_jpegs(style_dir)
    logger.info(f"Content images: {content_file_num}, Style images: {style_file_num}")
    logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

    data_iterator = iter(train_loader)
    iteration_start_time = time.time()
    current_iter = start_iter

    running_loss = running_content = running_style = 0.0

    while current_iter < num_iterations:
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            batch = next(data_iterator)

        content, style = batch

        content = content.to(device)
        style = style.to(device)

        # Regular forward pass
        generated, adain = model(content, style)

        total_loss, content_loss, style_loss = loss_fn(
            generated,
            adain,
            style
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += total_loss.item()
        running_content += content_loss.item()
        running_style += style_loss.item()

        if current_iter > 0 and current_iter % log_interval == 0:
            iteration_time = time.time() - iteration_start_time
            current_lr = optimizer.param_groups[0]['lr']

            avg_loss = running_loss / log_interval
            avg_content = running_content / log_interval
            avg_style = running_style / log_interval

            scheduler.step(avg_loss)

            log_msg = (f"Iteration {current_iter}/{num_iterations}"
                      f" | Loss: {avg_loss:.4f}"
                      f" [C: {avg_content:.4f}, S: {avg_style:.4f}]"
                      f" | LR: {current_lr:.6f}"
                      f" | Time: {iteration_time:.1f}s")
            logger.info(log_msg)

            # Update stats
            stats['total_loss'].append(avg_loss)
            stats['content_loss'].append(avg_content)
            stats['style_loss'].append(avg_style)
            stats['learning_rates'].append(current_lr)
            stats['times'].append(iteration_time)

            # Save checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                stats=stats,
                save_dir=save_dir,
                training_prefix=training_prefix,
                iteration=current_iter,
            )

            save_training_stats(stats, stats_file)
            iteration_start_time = time.time()
            running_loss = running_content = running_style = 0.0

        # Generate sample images
        monitor.generate_sample(
            model=model,
            device=device,
            iteration=current_iter
        )

        should_stop, reason = should_stop_training(stats)
        if should_stop:
            print(f"Early stopping triggered: {reason}")
            return model

        current_iter += 1

    # Save final stats
    stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_training_stats(stats, stats_file)
    logger.info("Training completed")
    logger.info(f"Final learning rate: {optimizer.param_groups[0]['lr']}")

    return model


def save_checkpoint(model, optimizer, scheduler, stats, save_dir, training_prefix, iteration):
    """Save a checkpoint of the model and training state."""
    checkpoint_path = os.path.join(save_dir, f'{training_prefix}_checkpoint_iter.pth')

    checkpoint = {
        'iteration': iteration,
        'decoder': model.decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'stats': stats
    }

    torch.save(checkpoint, checkpoint_path)

class TrainingMonitor:
    def __init__(
            self,
            content_path,
            style_path,
            save_dir,
            prefix,
            interval_minutes: int = 30
    ):
        self.content_path = content_path
        self.style_path = style_path
        self.save_dir = os.path.join(save_dir, 'progress_samples')
        self.prefix = prefix
        self.interval_seconds = interval_minutes * 60
        self.last_sample_time = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def should_generate_sample(self, iteration) -> bool:
        """Check if enough time has passed to generate a new sample"""
        current_time = time.time()
        if current_time - self.last_sample_time >= self.interval_seconds or iteration % 500 == 0:
            self.last_sample_time = current_time
            return True
        return False

    def generate_sample(self, model, device, iteration: int):
        """Generate and save a sample style transfer"""
        if not self.should_generate_sample(iteration):
            return

        timestamp = datetime.now().strftime('%H%M')
        filename = f'{self.prefix}_sample_{timestamp}_iter_{iteration:06d}.jpg'
        save_path = os.path.join(self.save_dir, filename)

        transfer_style(
            model=model,
            content_path=self.content_path,
            style_path=self.style_path,
            device=device,
            output_path=save_path,
            show_plot=False
        )

        print(f"Generated progress sample: {filename}")