"""
Pre-compute Augmented EMNIST for SuperFast Loading
--------------------------------------------------
HPC nodes often have immense GPU power but can be bottlenecked by CPU 
data-loading and image augmentation. This script pre-calculates 6 different 
random augmentations for every image and saves them to a Numpy file.
"""
import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATA_DIR="/home/shared/air/datasets/emnist"

def transpose_image(image, **kwargs):
    # EMNIST is rotated 90 degrees and flipped by default.
    return np.transpose(image, (1, 0))

def generate_variants(split='balanced', train=True, num_variants=6, save_dir='./data'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the augmentation pipeline
    transform = A.Compose([
        A.Lambda(image=transpose_image), 
        # Only apply random affine transforms to the training set
        A.Affine(translate_percent=(-0.08, 0.08), scale=(0.92, 1.08), rotate=(-15, 15), p=0.8) if train else A.NoOp(),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    print(f"Downloading/Loading EMNIST {split} (train={train})...")
    dataset = torchvision.datasets.EMNIST(root=DATA_DIR, split=split, train=train, download=True)
    raw_data = dataset.data.numpy()
    targets = dataset.targets.numpy()
    
    # We only need 1 variant for validation (no random augmentations)
    actual_variants = num_variants if train else 1
    
    # We will store the data as a massive numpy array: Shape (N_images, variants, Channels, H, W)
    # EMNIST is 28x28 grayscale (1 channel)
    N = len(raw_data)
    processed_data = np.zeros((N, actual_variants, 1, 28, 28), dtype=np.float32)
    
    print(f"Applying {actual_variants} transform variant(s) per image...")
    for i in tqdm(range(N), desc="Processing"):
        img = raw_data[i]
        for v in range(actual_variants):
            # Albumentations outputs a PyTorch tensor, convert it back to numpy for saving
            processed = transform(image=img)['image'].numpy()
            processed_data[i, v] = processed

    # Save to compressed Numpy file
    filename = os.path.join(save_dir, f"superfast_emnist_{'train' if train else 'val'}.npz")
    print(f"Saving to {filename}...")
    np.savez_compressed(filename, data=processed_data, targets=targets)
    print("Done!\n")

if __name__ == "__main__":
    generate_variants(train=True, num_variants=6)
    generate_variants(train=False, num_variants=1)