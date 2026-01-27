"""
Sparse Autoencoder Feature Extractor (3-Channel Version)

This module provides a SparseAutoEncoder for 3-channel jet images (pT_norm, E, log(pT_norm))
with the ability to extract bottleneck features for downstream classifiers.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import spconv.pytorch as spconv
import numpy as np
import h5py
import hdf5plugin 
import sys
import time
from typing import Tuple, Optional


H5_FILE_PATH = '/home/asm/LHC-AD/Working files/3C_jet_images_parallel.h5'
IMAGE_KEY = 'images'     
LABEL_KEY = 'labels'
MJJ_KEY = 'MJJ'

# Model Hyperparams
INPUT_CHANNELS = 3        
BASE_CHANNELS = 32        
LATENT_DIM = 64
SPATIAL_SHAPE = [50, 50]

# Training params
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = '/home/asm/LHC-AD/Attempt 5/Results and wts 3/sparse_ae_3channel.pth'


class SparseJetImageDataset3C(Dataset):
    """
    Loads 3-channel sparse jet image data from HDF5 file.
    Images are (3, 50, 50) with channels: [pT_norm, E, log(pT_norm)]
    """

    def __init__(self, file_path, image_key=IMAGE_KEY, label_key=LABEL_KEY):
        self.file_path = file_path
        self.image_key = image_key
        self.label_key = label_key
        
        try:
            with h5py.File(self.file_path, 'r') as f:
                if self.image_key not in f or self.label_key not in f:
                    raise KeyError(f"Missing keys: expected '{self.image_key}' and '{self.label_key}'.")
                self.length = f[self.image_key].shape[0]
                self.n_channels = f[self.image_key].shape[1]
                print(f"Dataset: {self.length} samples, {self.n_channels} channels")
        except FileNotFoundError:
            print(f"ERROR: HDF5 file not found at {file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error accessing HDF5: {e}")
            sys.exit(1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            image = f[self.image_key][idx]  # Shape: (3, 50, 50)
            label = f[self.label_key][idx]
        
        # Sum across channels to find active pixels
        channel_sum = np.abs(image).sum(axis=0)  # Shape: (50, 50)
        non_zero_mask = (channel_sum > 1e-6)
        non_zero_indices = np.where(non_zero_mask) 
        
        n_active = len(non_zero_indices[0])
        
        if n_active == 0:
            # Empty jet - add a dummy point
            indices_2d = torch.tensor([[0, 0]], dtype=torch.int32)
            features = torch.tensor([[1e-12] * self.n_channels], dtype=torch.float32)
        else:
            # Extract features for all active pixels: shape (n_active, 3)
            indices_2d = torch.tensor(np.stack(non_zero_indices, axis=1), dtype=torch.int32)
            features = torch.tensor(
                image[:, non_zero_indices[0], non_zero_indices[1]].T,  # (n_active, 3)
                dtype=torch.float32
            )

        return features, indices_2d, label


def sparse_collate_fn_3c(batch_list, spatial_shape=SPATIAL_SHAPE):
    """
    Custom collate function for 3-channel sparse data.
    """
    features_list, indices_list, labels_list = [], [], []
    
    for i, (features, indices_2d, label) in enumerate(batch_list):
        batch_index = torch.full((indices_2d.shape[0], 1), i, dtype=torch.int32)
        indices_i = torch.cat((batch_index, indices_2d), dim=1)
        
        features_list.append(features)
        indices_list.append(indices_i)
        labels_list.append(label)

    batch_features = torch.cat(features_list, dim=0).to(DEVICE)
    batch_indices = torch.cat(indices_list, dim=0).to(DEVICE)
    batch_labels = torch.tensor(labels_list, dtype=torch.float32).to(DEVICE)

    sparse_batch = spconv.SparseConvTensor(
        features=batch_features,
        indices=batch_indices,
        spatial_shape=spatial_shape,
        batch_size=len(batch_list)
    )
    return sparse_batch, batch_labels



class SparseAutoEncoder3C(nn.Module):
    """
    Sparse Convolutional Autoencoder for 3-channel jet images.

    """
    
    def __init__(self, input_channels=INPUT_CHANNELS, base_channels=BASE_CHANNELS):
        super().__init__()
        
        C = base_channels      # 32
        C2 = C * 2             # 64
        C4 = C * 4             # 128 (bottleneck)
        
        self.bottleneck_channels = C4

        # --- ENCODER ---
        self.enc1_conv = spconv.SubMConv2d(input_channels, C, 3, padding=1, bias=False)
        self.enc1_bn = nn.BatchNorm1d(C)
        
        self.enc_down1_conv = spconv.SparseConv2d(C, C2, 3, stride=2, padding=1, bias=False, indice_key='down1')
        self.enc_down1_bn = nn.BatchNorm1d(C2)

        self.enc2_conv = spconv.SubMConv2d(C2, C2, 3, padding=1, bias=False)
        self.enc2_bn = nn.BatchNorm1d(C2)
        
        self.enc_down2_conv = spconv.SparseConv2d(C2, C4, 3, stride=2, padding=1, bias=False, indice_key='down2')
        self.enc_down2_bn = nn.BatchNorm1d(C4)

        # --- DECODER ---
        self.dec_up1_conv = spconv.SparseInverseConv2d(C4, C2, 3, bias=False, indice_key='down2')
        self.dec_up1_bn = nn.BatchNorm1d(C2)
        
        self.dec1_conv = spconv.SubMConv2d(C2, C2, 3, padding=1, bias=False)
        self.dec1_bn = nn.BatchNorm1d(C2)
        
        self.dec_up2_conv = spconv.SparseInverseConv2d(C2, C, 3, bias=False, indice_key='down1')
        self.dec_up2_bn = nn.BatchNorm1d(C)
        
        self.output_layer = spconv.SubMConv2d(C, input_channels, 3, padding=1)

    def encode(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.enc1_conv(x)
        x = x.replace_feature(torch.relu(self.enc1_bn(x.features)))

        x = self.enc_down1_conv(x)
        x = x.replace_feature(torch.relu(self.enc_down1_bn(x.features)))

        x = self.enc2_conv(x)
        x = x.replace_feature(torch.relu(self.enc2_bn(x.features)))

        x_bottleneck = self.enc_down2_conv(x)
        x_bottleneck = x_bottleneck.replace_feature(torch.relu(self.enc_down2_bn(x_bottleneck.features)))
        
        return x_bottleneck

    def decode(self, x_bottleneck: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.dec_up1_conv(x_bottleneck)
        x = x.replace_feature(torch.relu(self.dec_up1_bn(x.features)))

        x = self.dec1_conv(x)
        x = x.replace_feature(torch.relu(self.dec1_bn(x.features)))

        x = self.dec_up2_conv(x)
        x = x.replace_feature(torch.relu(self.dec_up2_bn(x.features)))

        return self.output_layer(x)

    def forward(self, x: spconv.SparseConvTensor, return_features: bool = False):
        x_bottleneck = self.encode(x)
        reconstruction = self.decode(x_bottleneck)
        
        if return_features:
            return reconstruction, x_bottleneck
        return reconstruction


#  Some helper functions

def aggregate_sparse_features(bottleneck: spconv.SparseConvTensor, batch_size: int) -> torch.Tensor:
    """
    Aggregate sparse bottleneck features into fixed-size per-sample vectors.
    Uses global average pooling over spatial dimensions.
    
    Returns:
        Tensor of shape [batch_size, C] with aggregated features
    """
    features = bottleneck.features
    indices = bottleneck.indices
    batch_indices = indices[:, 0]
    
    num_channels = features.shape[1]
    device = features.device
    
    aggregated = torch.zeros(batch_size, num_channels, device=device)
    
    for b in range(batch_size):
        mask = (batch_indices == b)
        if mask.any():
            aggregated[b] = features[mask].mean(dim=0)
    
    return aggregated


def calculate_sparse_loss(output: spconv.SparseConvTensor, target: spconv.SparseConvTensor):
    return nn.L1Loss(reduction='mean')(output.features, target.features)


# Train 
def train_autoencoder(max_train_samples: int = 300000, save_path: str = MODEL_SAVE_PATH):
    """
    Train the 3-channel sparse autoencoder on background data.
    
    Args:
        max_train_samples: Maximum number of background samples to use
        save_path: Path to save the trained model
    """
    
    print("Training 3-Channel Sparse Autoencoder")

    full_dataset = SparseJetImageDataset3C(H5_FILE_PATH)
    
    # background indices
    with h5py.File(H5_FILE_PATH, 'r') as f:
        all_labels = f[LABEL_KEY][:]
    
    bkg_indices = np.where(all_labels == 0)[0]
    print(f"Total background samples: {len(bkg_indices)}")
    
    if max_train_samples > 0 and max_train_samples < len(bkg_indices):
        print(f"Limiting to {max_train_samples} samples")
        bkg_indices = bkg_indices[:max_train_samples]
    
    bkg_dataset = Subset(full_dataset, bkg_indices)
    
    train_loader = DataLoader(
        bkg_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=sparse_collate_fn_3c,
        timeout=120.0,
        prefetch_factor=2
    )
    
    print(f"Training on {len(bkg_dataset)} background samples")
    print(f"Batches per epoch: {len(train_loader)}")
    
    model = SparseAutoEncoder3C().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nModel on {DEVICE}")
    print(f"Bottleneck channels: {model.bottleneck_channels}")
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        epoch_start = time.time()
        n_batches = len(train_loader)
        
        for batch_idx, (sparse_input, _) in enumerate(train_loader):
            optimizer.zero_grad()
            reconstruction = model(sparse_input)
            loss = calculate_sparse_loss(reconstruction, sparse_input)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Loss: {avg_loss:.6f} | Time: {epoch_time:.1f}s | {n_batches/epoch_time:.1f} batch/s")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")
    
    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
    
    return model


def load_pretrained_model(weights_path: str = MODEL_SAVE_PATH) -> SparseAutoEncoder3C:
    """Load a pretrained 3-channel sparse autoencoder."""
    model = SparseAutoEncoder3C().to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    print(f"Loaded weights from: {weights_path}")
    print(f"Device: {DEVICE}")
    print(f"Bottleneck channels: {model.bottleneck_channels}")
    
    model.eval()
    return model


def extract_features_from_dataset(
    model: SparseAutoEncoder3C,
    dataset: SparseJetImageDataset3C,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    num_workers: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract bottleneck features from the dataset.
    
    Args:
        model: Trained SparseAutoEncoder3C model
        dataset: Dataset to extract features from
        batch_size: Batch size for extraction
        max_samples: Optional limit on samples
        num_workers: DataLoader workers (use 0 in Jupyter, 4 in terminal)
    
    Returns:
        features: numpy array of shape [N, 128] (bottleneck features)
        labels: numpy array of shape [N]
    """
    if max_samples is not None and max_samples < len(dataset):
        dataset_to_use = Subset(dataset, list(range(max_samples)))
    else:
        dataset_to_use = dataset
    
    loader = DataLoader(
        dataset_to_use,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sparse_collate_fn_3c,
        timeout=120.0 if num_workers > 0 else 0
    )
    
    all_features = []
    all_labels = []
    n_batches = len(loader)
    
    print(f"Extracting features from {len(dataset_to_use)} samples ({n_batches} batches)...")
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (sparse_input, batch_labels) in enumerate(loader):
            _, bottleneck = model(sparse_input, return_features=True)
            batch_features = aggregate_sparse_features(bottleneck, len(batch_labels))
            
            all_features.append(batch_features.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
            
            # Print progress at 25%, 50%, 75%, 100%
            if (batch_idx + 1) % max(1, n_batches // 4) == 0:
                pct = 100 * (batch_idx + 1) / n_batches
                print(f"  Progress: {pct:.0f}% ({batch_idx + 1}/{n_batches} batches)")
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels: {np.sum(labels == 0)} background, {np.sum(labels == 1)} signal")
    
    return features, labels


if __name__ == '__main__':
    if torch.cuda.is_available():
        try:
            torch.multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    print(f"Device: {DEVICE}")
    train_autoencoder()
