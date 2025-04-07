import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import time
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from models import get_model, LSTMModel  
import matplotlib.pyplot as plt
from load_data import _load_data
from model import CNN, print_model_summary
import os 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wandb

# Argument parser for hyperparameters
parser = argparse.ArgumentParser(description='Model Training for MNIST and CIFAR-100')
parser.add_argument('--model', type=str, default='cnn', help='Choose model to train (cnn lstm)')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use (mnist or cifar100)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for optimizer')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
parser.add_argument('--activation_fn_lstm', type=str, default='relu', help='Activation function for LSTM')
parser.add_argument('--activation_fn', type=str, default='relu', help='Activation function (relu, elu, leaky_relu)')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
args = parser.parse_args()


wandb.init(project="mnist_cnn", reinit=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_results(hyperparams, train_time, accuracy):
    with open("results.txt", "a") as file:
        file.write(f"Hyperparameters: {hyperparams}\n")
        file.write(f"Training Time: {train_time:.2f} minutes\n")
        file.write(f"Accuracy: {accuracy:.2f}%\n")
        file.write("-----------------------------------------------------\n")

os.makedirs('graphs', exist_ok=True)


# Load Model
model = get_model(args.model, args.dataset).to(device)

if args.model.lower() == 'cnn':
    model = CNN(activation_fn=args.activation_fn).to(device)
    print_model_summary(model)
elif args.model.lower() == 'lstm':
    model = LSTMModel(input_size=28, hidden_size=128, num_layers=2, num_classes=10, activation_fn=args.activation_fn).to(device)
else:
    raise ValueError('Model not supported. Choose from: cnn, resnet18, lstm')



if args.dataset.lower() == 'mnist':
    train_loader, test_loader = _load_data(args.data_path, args.batch_size)
else:
    raise ValueError('Unsupported dataset! Choose from: mnist, cifar100')



# Optimizer & Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, epochs=args.epochs, steps_per_epoch=len(train_loader))
scaler = GradScaler()


def adjust_learning_rate(learning_rate, optimizer, epoch):
    lr = learning_rate
    if epoch > 5:
        lr = 0.001
    if epoch >= 10:
        lr = 0.0001
    if epoch > 20:
        lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



# Training Function

training_accuracies = []
epoch_losses = []


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, patience=args.patience, checkpoint_dir="checkpoints"):
    model.train()
    best_loss = float('inf')
    counter = 0
    scaler = GradScaler()
    losses = []
    accuracies = []
    val_losses = []
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('conv_visuals', exist_ok=True)

    # Track training time
    start_time = time.time()

    for epoch in range(num_epochs):
        adjust_learning_rate(args.learning_rate, optimizer, epoch)
        epoch_loss = 0
        correct = 0 
        total = 0 

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            if isinstance(model, LSTMModel):
                images = images.view(images.size(0), -1, 28)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images, visualize=(batch_idx == 0 and epoch % 5 == 0))
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            epoch_loss += loss.item()
            losses.append(loss.item())
            wandb.log({"Training Loss": loss.item()})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        accuracy = 100 * correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)

        training_accuracies.append(accuracy)
        epoch_losses.append(avg_loss)

        avg_val_loss = evaluate_model(model, test_loader, criterion) #remove if doesn't work
        val_losses.append(avg_val_loss)

        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        checkpoint_path = os.path.join(checkpoint_dir, f'cnn_mnist_epoch_{epoch+1}_loss_{avg_loss:.4f}.pth')
        torch.save(model.state_dict(), checkpoint_path)


        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), f'{args.model}_{args.dataset}.pth')
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Calculate total training time
    end_time = time.time()
    train_time = (end_time - start_time) / 60 
    logger.info(f'Total Training Time: {train_time:.2f} minutes')
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.savefig("training_loss.png")


    plt.plot(accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.savefig('graphs/training_accuracy.png')

    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig("mnist_training_validation_loss.png")



    return train_time



# Testing Function

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Check if the model is LSTM
            if isinstance(model, LSTMModel):
                images = images.view(images.size(0), -1, 28)  # Reshape for LSTM compatibility

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    logger.info(f'Test Accuracy: {accuracy:.2f}%')

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig('graphs/confusion_matrix.png')
    


    return accuracy 

def inference(model_path, test_loader):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)


    if isinstance(model, LSTMModel):
        # If model is LSTM, reshape the image for compatibility
        images = images.view(images.size(0), -1, 28)  

    with torch.no_grad():
        output = model(images[0].unsqueeze(0))  # Single image inference
        if isinstance(model, LSTMModel):
            output = model(images[0].unsqueeze(0).view(1, 28, 28))  # Reshape for LSTM
        predicted = torch.argmax(output, dim=1)

    logger.info(f'Predicted label: {predicted.item()}, Actual label: {labels[0].item()}')

def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)


def save_training_progress():
    # Plotting training loss and accuracy
    plt.figure()
    plt.plot(epoch_losses, label='Loss')
    plt.plot(training_accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('graphs/training_progress.png')
    #plt.show()

if __name__ == "__main__":
    train_time = train_model(model, train_loader, criterion, optimizer, scheduler)
    accuracy = test_model(model, test_loader)
    inference(f'{args.model}_{args.dataset}.pth', test_loader)

    hyperparams = {
        'model': args.model,
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'activation_fn': args.activation_fn
    }

    log_results(hyperparams, train_time, accuracy)

    save_training_progress()