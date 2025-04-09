# HW3


This repository contains PyTorch implementations of:
- A **Convolutional Neural Network (CNN)** for classifying the MNIST dataset.
- A **ResNet-18** implementation for classifying the CIFAR-100 dataset (Bonus).

## HW Structure 

```
├── load_data.py          # Data loading and preprocessing for MNIST and CIFAR-100
├── main.py               # Main training and evaluation logic
├── model.py              # CNN architecture (3 conv + 2 FC layers)
├── models.py             # Model selector for CNN, LSTM, ResNet
├── organized_main.py     # A cleaner version of main.py 
├── resnet_bonus.py       # Bonus: ResNet-18 for CIFAR-100
├── report/               # Report write up, in zip file, unzip to edit the latex 
├── results/              # Output logs
├── run.sh                # Shell script to run training

```

Command args for trainning:

```
python main.py --model cnn --dataset mnist --epochs 20 --batch_size 64
```

```
python resnet_bonus.py 
```
