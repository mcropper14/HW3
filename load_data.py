import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def _load_data(DATA_PATH, batch_size):
    '''Data loader'''
    print("data_path: ", DATA_PATH)

    train_trans = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True, train=True, transform=train_trans)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, download=True, train=False, transform=test_trans)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
