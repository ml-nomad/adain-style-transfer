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
    log_file = os.path.join(save_dir, f'training_{prefix}.log')

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


def should_stop_training(stats, patience=5, min_improvement=0.0001, window_size=5, lookback_windows=10):
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


def train_model(
        content_dir,
        style_dir,
        save_dir,
        training_prefix,
        num_epochs,
        batch_size,
        base_lr,
        style_loss_coeff,
        log_interval
):
    device = get_device()

    train_loader = create_dataloaders(content_dir, style_dir, batch_size)
    loss_fn = StyleTransferLoss(style_coeff=style_loss_coeff).to(device)
    model = Model().to(device)

    # Setup logging
    stats_file = os.path.join(save_dir, f'{training_prefix}_training_stats.json')
    logger = setup_logging(save_dir, training_prefix)

    # Log training setup
    logger.info(f"Starting training for {num_epochs} epochs, batch size: {batch_size}, style loss coeff: {style_loss_coeff}")
    content_file_num = count_jpegs(content_dir)
    style_file_num = count_jpegs(style_dir)
    logger.info(f"Content images: {content_file_num}, Style images: {style_file_num}")
    logger.info(f"Initial learning rate: {base_lr}")

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
        patience=3,
        min_lr=1e-5,
        cooldown=1
    )

    # Training stats setup
    stats = {
        'epoch_times': [],
        'total_loss': [],
        'content_loss': [],
        'style_loss': [],
        'learning_rates': [],
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    for epoch in range(num_epochs):

        epoch_start = time.time()
        model.train()

        running_loss = running_content = running_style = 0.0

        for batch_idx, (content, style) in enumerate(train_loader):
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

            # Log progress
            if batch_idx % log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                avg_content = running_content / (batch_idx + 1)
                avg_style = running_style / (batch_idx + 1)

                log_msg = (f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx}"
                           f" | Loss: {avg_loss:.4f} [C: {avg_content:.4f}, S: {avg_style:.4f}]")
                logger.info(log_msg)

            monitor.generate_sample(
                model=model,
                device=device,
                epoch=epoch,
                current_iteration=batch_idx
            )

        # Epoch statistics
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        avg_content = running_content / len(train_loader)
        avg_style = running_style / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(avg_loss)

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f'{training_prefix}_checkpoint_epoch{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'encoder': model.encoder.state_dict(),
            'decoder': model.decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_loss,
            'content_loss': avg_content,
            'style_loss': avg_style,
            'stats': stats  # Save current training stats
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Record stats
        stats['epoch_times'].append(epoch_time)
        stats['total_loss'].append(avg_loss)
        stats['content_loss'].append(avg_content)
        stats['style_loss'].append(avg_style)
        stats['learning_rates'].append(current_lr)

        save_training_stats(stats, stats_file)
        logger.info(f"Training statistics saved to {stats_file}")

        # Log epoch summary
        log_msg = (f"Epoch {epoch + 1} completed in {epoch_time:.1f}s | "
                   f"Avg Loss: {stats['total_loss'][-1]:.4f} [C: {stats['content_loss'][-1]:.4f}, S: {stats['style_loss'][-1]:.4f}, LR: {current_lr:.6f}]")
        logger.info(log_msg)

        should_stop, reason = should_stop_training(stats)
        if should_stop:
            print(f"Early stopping triggered: {reason}")
            break

    # After training completion
    logger.info(f"Training completed. Final LR: {optimizer.param_groups[0]['lr']}")

    # Save final stats
    stats['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_training_stats(stats, stats_file)
    logger.info(f"Training statistics saved to {stats_file}")

    return model

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
        self.last_epoch = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def should_generate_sample(self, epoch, iter) -> bool:
        """Check if enough time has passed to generate a new sample"""
        current_time = time.time()
        if current_time - self.last_sample_time >= self.interval_seconds or iter % 500 == 0 or epoch > self.last_epoch:
            self.last_sample_time = current_time
            self.last_epoch = epoch
            return True
        return False

    def generate_sample(self, model, device, epoch: int, current_iteration: int):
        """Generate and save a sample style transfer"""
        if not self.should_generate_sample(epoch, current_iteration):
            return

        # Generate filename with timestamp and iteration
        timestamp = datetime.now().strftime('%H%M')
        filename = f'{self.prefix}_sample_{timestamp}_iter_{epoch:02d}_{current_iteration:04d}.jpg'
        save_path = os.path.join(self.save_dir, filename)

        # Run transfer_style function
        transfer_style(
            model=model,
            content_path=self.content_path,
            style_path=self.style_path,
            device=device,
            output_path=save_path,
            show_plot=False  # Don't show plots during training
        )

        print(f"Generated progress sample: {filename}")