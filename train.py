import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from dataset import get_data_loaders
from model import EmotionCNN

def train_model(train_loader, val_loader, y_train, epochs=30, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class weights
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Initialize model
    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Training
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Update LR
        scheduler.step(val_loss)
        
        # Stats
        train_loss = running_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    torch.save(model.state_dict(), "emotion_model.pth")
    print("Training complete! Model saved.")

if __name__ == "__main__":
    train_loader, val_loader, y_train = get_data_loaders()
    train_model(train_loader, val_loader, y_train, epochs=30)