import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Improved Transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Load CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform_test, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4, pin_memory=True)

#Load ResNet-18 with CIFAR-100 Fixes
resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
resnet18.maxpool = nn.Identity()
resnet18.fc = nn.Linear(resnet18.fc.in_features, 100)
resnet18 = resnet18.to(device)

# Fine-tune only deeper layers
for param in resnet18.layer3.parameters():
    param.requires_grad = True
for param in resnet18.layer4.parameters():
    param.requires_grad = True

# Function to generate a random bounding box for CutMix
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# MixUp & CutMix Augmentations
def mixup_cutmix(x, y, alpha=0.8, cutmix_prob=0.3):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size(0)).to(x.device)
    if np.random.rand() < cutmix_prob:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    else:
        x = lam * x + (1 - lam) * x[rand_index, :]
    y_a, y_b = y, y[rand_index]
    return x, y_a, y_b, lam

def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training function with logging and loss plotting
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=400, patience=50, model_name="resnet18_cifar100.pth"):
    model.train()
    best_loss = float('inf')
    counter = 0
    scaler = GradScaler()
    accumulation_steps = 4  
    losses = [] 
    start_time = time.time() 

    for epoch in range(num_epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(train_loader):
            images, labels_a, labels_b, lam = mixup_cutmix(images, labels)
            images, labels_a, labels_b = images.to(device), labels_a.to(device), labels_b.to(device)

            with autocast():
                outputs = model(images)
                loss = mixup_cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), model_name)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    # Save training time
    end_time = time.time()
    training_time = (end_time - start_time) / 60  # Convert to minutes

    # Save results to a file
    with open("training_results.txt", "w") as f:
        f.write(f"Training Time: {training_time:.2f} minutes\n")

    # Plot and save training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, label="Training Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.savefig("training_loss_plot.png")

# Optimizer & Scheduler
optimizer = optim.SGD(resnet18.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=200, steps_per_epoch=len(train_loader))
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# Train and test
train_model(resnet18, train_loader, criterion, optimizer, scheduler)

#Load best model and apply TTA
resnet18.load_state_dict(torch.load("resnet18_cifar100.pth"))

#Test-time augmentation (TTA) and save results
def test_tta(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs += model(torch.flip(images, [3]))  
            outputs += model(torch.rot90(images, k=1, dims=[2, 3]))  
            outputs /= 3  
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy with TTA: {test_accuracy:.2f}%')

    with open("training_results.txt", "a") as f:
        f.write(f"Test Accuracy with TTA: {test_accuracy:.2f}%\n")

test_tta(resnet18, test_loader)

