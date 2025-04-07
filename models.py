import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from model import CNN  # Importing your CNN model


def debug_model_freezing(model):
    """
    Print which layers are frozen and which are trainable for debugging purposes.
    """
    print("\nModel Layers Freezing Debug Information:")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Layer {name} is FROZEN")
        else:
            print(f"Layer {name} is TRAINABLE")
    print("\n")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, activation_fn='relu'):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.activation_fn = self.get_activation_function(activation_fn)

    def get_activation_function(self, activation_fn):
        if activation_fn == 'relu':
            return F.relu
        elif activation_fn == 'sigmoid':
            return torch.sigmoid
        elif activation_fn == 'tanh':
            return torch.tanh
        elif activation_fn == 'elu':
            return F.elu
        elif activation_fn == 'softmax':
            return F.softmax
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        if self.activation_fn == F.softmax:
            return self.activation_fn(out, dim=1)
        else:
            return self.activation_fn(out)


def get_model(model_name, dataset_name):
    if model_name == 'cnn':
        return CNN()  # Load your CNN model

    elif model_name in ['resnet18', 'resnet34']:
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_name == 'resnet34':
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        if dataset_name == 'cifar100':
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            model.fc = nn.Linear(model.fc.in_features, 100)

            for param in model.layer1.parameters():
                param.requires_grad = False
            for param in model.layer2.parameters():
                param.requires_grad = False

            debug_model_freezing(model)

        elif dataset_name == 'mnist':
            model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            model.fc = nn.Linear(model.fc.in_features, 10)

        return model

    elif model_name == 'lstm':
        return LSTMModel(input_size=28, hidden_size=128, num_layers=2, num_classes=10)

    else:
        raise ValueError(f"Model {model_name} not recognized")
