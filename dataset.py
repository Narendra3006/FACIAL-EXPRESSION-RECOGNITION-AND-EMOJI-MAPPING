import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

class FERDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(48, 48).astype(np.uint8)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0

        label = torch.tensor(label, dtype=torch.long)
        return image, label

def get_data_loaders(batch_size=64, test_size=0.2):
    # Load dataset
    df = pd.read_csv("fer2013.csv")
    pixels = df["pixels"].apply(lambda x: np.fromstring(x, dtype=int, sep=' ')).values
    labels = df["emotion"].values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        pixels, labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=42
    )

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets
    train_dataset = FERDataset(X_train, y_train, transform=train_transform)
    val_dataset = FERDataset(X_val, y_val, transform=val_transform)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, y_train  # ✅ Fixed variable name

