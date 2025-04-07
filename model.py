import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
import os 

class CNN(nn.Module):
    def __init__(self, activation_fn='relu', save_dir='conv_visuals'):
        super(CNN, self).__init__()

        self.activation_fn = activation_fn.lower()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        # Input: [batch_size, 1, 28, 28] (MNIST images are 28x28, single channel)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: [batch_size, 32, 28, 28]
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: [batch_size, 64, 28, 28]
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: [batch_size, 128, 28, 28]
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)  # Reduces size by half [batch_size, 128, 14, 14]

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Input flattened from [batch_size, 128, 3, 3]
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)  # Output: [batch_size, 10] (10 classes)

    def forward(self, x, visualize=False):
        # Convolutional + Pooling Layers
        x = self.pool(self.apply_activation(self.bn1(self.conv1(x))))  # Output: [batch_size, 32, 14, 14]
        if visualize:
            self.visualize_convolutions(x, 'conv1')
        
        
        
        x = self.pool(self.apply_activation(self.bn2(self.conv2(x))))  # Output: [batch_size, 64, 7, 7]
        if visualize:
            self.visualize_convolutions(x, 'conv2')
        
        x = self.pool(self.apply_activation(self.bn3(self.conv3(x))))  # Output: [batch_size, 128, 3, 3]
        if visualize:
            self.visualize_convolutions(x, 'conv3')


        # Flatten the output from convolutional layers
        x = torch.flatten(x, 1)  # Output: [batch_size, 128*3*3 = 1152]

        # Fully connected layers
        x = self.apply_activation(self.fc1(x))  # Output: [batch_size, 256]
        x = self.dropout(x)
        x = self.fc2(x)  # Output: [batch_size, 10]

        return x


    def apply_activation(self, x):
        if self.activation_fn == 'relu':
            return F.relu(x)
        elif self.activation_fn == 'elu':
            return F.elu(x)
        elif self.activation_fn == 'leaky_relu':
            return F.leaky_relu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")

    def visualize_convolutions(self, feature_maps, layer_name):
        """Visualize and save the convolutional feature maps"""
        feature_maps = feature_maps.detach().cpu()
        num_filters = feature_maps.shape[1]

        # Plot feature maps
        fig, axes = plt.subplots(4, 8, figsize=(12, 6))
        fig.suptitle(f'Feature Maps of {layer_name}')
        
        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                ax.imshow(feature_maps[0, i, :, :], cmap='gray')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.savefig(f"{self.save_dir}/{layer_name}_feature_maps.png")

    
    
def print_model_summary(model):
    """Displays model architecture and layer sizes"""
    summary(model, (1, 28, 28))
