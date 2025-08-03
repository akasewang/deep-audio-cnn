# Import necessary libraries
from pathlib import Path
import pandas as pd
import numpy as np

import modal  # For cloud execution
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm  # For progress bars
from torch.utils.tensorboard import SummaryWriter  # For logging

from src.components.model import AudioCNN  # Our custom CNN model

# Define Modal app and Docker image configuration
app = modal.App("audio-cnn")

# Create a Docker image with required dependencies
image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         # Audio processing tools
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands([
             # Download and extract ESC-50 dataset
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))  # Add our model code

# Define persistent volumes for data and model storage
volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


class ESC50Dataset(Dataset):
    """Dataset class for ESC-50 environmental sound classification dataset."""

    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        # Split data into train/test based on fold (fold 5 is test)
        if split == 'train':
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        # Create class mappings
        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata['label'] = self.metadata['category'].map(
            self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply transforms (e.g., convert to spectrogram)
        if self.transform:
            spectrogram = self.transform(waveform)
        else:
            spectrogram = waveform

        return spectrogram, row['label']


def mixup_data(x, y):
    """Implementation of mixup data augmentation."""
    lam = np.random.beta(0.2, 0.2)  # Mixup ratio
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(
        x.device)  # Random permutation of samples

    # Mix inputs
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]  # Get corresponding labels
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Special loss function for mixup augmentation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60 * 60 * 3)
def train():
    """Main training function running on Modal's cloud."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)  # TensorBoard logger

    esc50_dir = Path("/opt/esc50-data")

    # Define audio transforms for training data (with augmentation)
    train_transform = nn.Sequential(
        T.MelSpectrogram(  # Convert to mel-spectrogram
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),  # Convert to dB scale
        # Frequency masking augmentation
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)  # Time masking augmentation
    )

    # Validation transforms (no augmentation)
    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    # Create datasets
    train_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="train",
        transform=train_transform)

    val_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="test",
        transform=val_transform)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Set up device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    # Training configuration
    num_epochs = 100
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1)  # With label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    # One-cycle learning rate scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1  # Percentage of cycle spent increasing LR
    )

    best_accuracy = 0.0

    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # Training loop with progress bar
        progress_bar = tqdm(
            train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            # Apply mixup augmentation 30% of the time
            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(
                    criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # Log training metrics
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)
        writer.add_scalar(
            'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(test_dataloader)

        # Log validation metrics
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', accuracy, epoch)

        print(
            f'Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'classes': train_dataset.classes
            }, '/models/best_model.pth')
            print(f'New best model saved: {accuracy:.2f}%')

    writer.close()
    print(f'Training completed! Best accuracy: {best_accuracy:.2f}%')

# Local entry point for running the training remotely


@app.local_entrypoint()
def main():
    train.remote()
