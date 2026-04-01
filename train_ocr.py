"""
Tiny EMNIST (28x28) CNN Trainer for HPC Workshops
-------------------------------------------------
A simple training script to test executing GPU workloads on HPC. Generated with various LLMs.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# EMNIST Balanced classes (47 alphanumeric characters)
CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a tiny OCR model on EMNIST.")
    parser.add_argument('--data-dir', type=str, default='/home/shared/air/datasets/emnist', help='Path to dataset')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--target-acc', type=float, default=0.90, help='Target validation accuracy')
    parser.add_argument('--disable-amp', action='store_true', help='Disable Automatic Mixed Precision (AMP)')
    
    # ---------------------------------------------------------
    # WORKSHOP CORE CONCEPT: DATA LOADING BOTTLENECKS
    # ---------------------------------------------------------
    parser.add_argument('--loader', type=str, choices=['normal', 'fast', 'superfast'], default='normal', 
                        help='Data strategy: normal (disk), fast (RAM), superfast (Pre-computed numpy file)')
    return parser.parse_args()

# ==========================================
# MODEL DEFINITION
# ==========================================
class TinyOCR(nn.Module):
    """A small, easy-to-train Convolutional Neural Network."""
    def __init__(self, num_classes=47):
        super(TinyOCR, self).__init__()
        
        # 1. Feature Extractor: Learns edges, curves, and shapes
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),          # Standard activation function
            nn.MaxPool2d(2, 2), # Shrinks spatial dimensions (16x14x14)
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Shrinks spatial dimensions (32x7x7)
        )
        
        # 2. Classifier: Takes extracted features and assigns a class probability
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # Prevents overfitting by randomly turning off neurons
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# DATASET STRATEGIES (Normal, Fast, SuperFast)
# ==========================================
def transpose_image(image, **kwargs):
    return np.transpose(image, (1, 0))

class StandardEMNIST(Dataset):
    """
    NORMAL: Standard PyTorch lazy-loading.
    Reads data continuously, which can starve the GPU if the CPU/Disk is slow.
    """
    def __init__(self, root, split, train, transform=None):
        self.dataset = torchvision.datasets.EMNIST(root=root, split=split, train=train, download=(not "/home/shared/" in root))
        self.transform = transform

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_np = np.array(img)
        if self.transform:
            img_np = self.transform(image=img_np)['image'].float()
        return img_np, label

class FastEMNIST(Dataset):
    """
    FAST: Loads the raw images into RAM.
    Avoids disk I/O bottlenecks, but the CPU still has to apply transforms 
    (rotations, normalization) every single step.
    """
    def __init__(self, root, split, train, transform=None):
        base_dataset = torchvision.datasets.EMNIST(root=root, split=split, train=train, download=(not "/home/shared/" in root))
        self.data = base_dataset.data.numpy()
        self.targets = base_dataset.targets.long()
        self.transform = transform
        
        ram_mb = (self.data.nbytes + self.targets.nbytes) / (1024 * 1024)
        print(f"--> [FAST] Raw data cached in RAM. Footprint: {ram_mb:.2f} MiB")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx] 
        label = self.targets[idx]
        if self.transform:
            img = self.transform(image=img)['image'].float()
        return img, label

class SuperFastEMNIST(Dataset):
    """
    SUPERFAST: Loads pre-computed, pre-augmented numpy files.
    Zero Disk I/O. Zero CPU augmentations during training. Pure GPU feeding.
    """
    def __init__(self, root, train):
        filename = f"superfast_emnist_{'train' if train else 'val'}.npz"
        filepath = os.path.join(root, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Cannot find {filepath}!")
            print("You must run 'generate_superfast_data.py' (and set the right dataset directory) first.")
            sys.exit(1)
            
        print(f"--> [SUPERFAST] Loading pre-computed arrays from {filepath}...")
        npz_file = np.load(filepath)
        
        # Convert numpy arrays to PyTorch tensors
        self.data = torch.from_numpy(npz_file['data'])
        self.targets = torch.from_numpy(npz_file['targets']).long()
        self.train = train
        self.num_variants = self.data.shape[1] # e.g., 6 variants for train
        
        # Array holding which variant to use for each image (updated per epoch)
        self.current_choices = torch.zeros(len(self.data), dtype=torch.long)
        print(f"--> [SUPERFAST] Loaded {self.num_variants} variant(s) per image into RAM.")

    def set_epoch(self):
        """Randomly pick 1 of the N pre-computed variants for this epoch."""
        if self.train:
            self.current_choices = torch.randint(0, self.num_variants, (len(self.data),))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        # O(1) Instant Memory Lookup: CPU does zero math here.
        chosen_variant_idx = self.current_choices[idx]
        return self.data[idx, chosen_variant_idx], self.targets[idx]

def main():
    args = parse_args()
    
    # ---------------------------------------------------------
    # HPC GUARDRAIL: Prevent users from training on the Login Node
    # ---------------------------------------------------------
    if not torch.cuda.is_available():
        print("\n" + "❌ " * 20)
        print("CRITICAL ERROR: CUDA IS NOT AVAILABLE!")
        print("You are attempting to run a GPU training script, but PyTorch cannot find a GPU.")
        print("Common causes:")
        print("  1. You are running this directly on the login node (Don't do this!).")
        print("  2. You submitted a SLURM job but forgot to request a GPU.")
        print("  3. Your CUDA drivers/modules are not loaded.")
        print("     Fix: Make sure your venv and enroot container are correct.")
        print("❌ " * 20 + "\n")
        sys.exit(1)
        
    DEVICE = torch.device("cuda")
    USE_AMP = not args.disable_amp # Mixed Precision speeds up modern GPUs via TensorCores
    
    # Define On-the-fly transforms (Used only for Normal and Fast loaders)
    train_transform = A.Compose([
        A.Lambda(image=transpose_image), 
        A.Affine(translate_percent=(-0.08, 0.08), scale=(0.92, 1.08), rotate=(-15, 15), p=0.8),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Lambda(image=transpose_image),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])
    
    print(f"\nInitializing {args.loader.upper()} Dataset Pipeline...")
    
    # Instantiate the correct Dataset based on user arguments
    if args.loader == 'normal':
        train_dataset = StandardEMNIST(args.data_dir, split='balanced', train=True, transform=train_transform)
        val_dataset = StandardEMNIST(args.data_dir, split='balanced', train=False, transform=val_transform)
    elif args.loader == 'fast':
        train_dataset = FastEMNIST(args.data_dir, split='balanced', train=True, transform=train_transform)
        val_dataset = FastEMNIST(args.data_dir, split='balanced', train=False, transform=val_transform)
    elif args.loader == 'superfast':
        train_dataset = SuperFastEMNIST(args.data_dir, train=True)
        val_dataset = SuperFastEMNIST(args.data_dir, train=False)

    # DataLoaders batch the images and use parallel workers to feed the GPU
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize Model, Loss Function, and Optimizer
    model = TinyOCR(num_classes=47).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # GradScaler is required to prevent underflow when using FP16 (AMP)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"\n🚀 Starting training on {DEVICE} [{torch.cuda.get_device_name(0)}]")
    print(f"⚙️  AMP Enabled: {USE_AMP} | Loader: {args.loader.upper()}\n")
    
    for epoch in range(1, args.epochs + 1):
        
        # If using SuperFast, "roll the dice" to pick new image variants for this epoch
        if args.loader == 'superfast':
            train_dataset.set_epoch()
            
        # ==================== TRAINING PHASE ====================
        model.train() # Turns on Dropout and BatchNorm tracking
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad() # Clear previous gradients
            
            # Autocast automatically mixes FP16 and FP32 for maximum speed
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Calculate gradients and update weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # ==================== VALIDATION PHASE ====================
        model.eval() # Turns off Dropout
        correct, total = 0, 0
        
        with torch.no_grad(): # Don't track gradients (saves memory/compute)
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP):
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total
        print(f"📈 Validation Accuracy: {val_acc:.4f} (Target: {args.target_acc})")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "model_best.pt")
            print(f"⭐️ Saved new best model!")
            
        # Early Stopping
        if val_acc >= args.target_acc:
            print("🎯 Target accuracy reached. Stopping early.")
            break

    print("\n✅ Training Complete. Best Validation Accuracy:", best_acc)

    # Load the best weights before exporting
    model.load_state_dict(torch.load("model_best.pt"))
    
    # ==========================================
    # EXPORT TO ONNX FOR BROWSER INFERENCE
    # ==========================================
    print("Exporting best model to ONNX for Web App...")
    model.eval().to("cpu")
    dummy_input = torch.randn(1, 1, 28, 28) # 1 image, 1 channel, 28x28
    
    torch.onnx.export(
        model, dummy_input, "browser_checkpoint_best.onnx",
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        # Dynamic axes allow the browser to send batches of >1 if needed later
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"📦 Export complete: 'browser_checkpoint_best.onnx' (Params: {sum(p.numel() for p in model.parameters()):,})")


if __name__ == "__main__":
    main()
