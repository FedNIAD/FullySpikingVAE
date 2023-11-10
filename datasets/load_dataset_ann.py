import torch
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
import PIL

def load_mnist(data_path, batch_size):
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.MNIST(data_path,
                                            train=True,
                                            transform=transform,
                                            download=True)

    testset = torchvision.datasets.MNIST(data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader


def load_fashionmnist(data_path,batch_size):
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.FashionMNIST(data_path,
                                            train=True,
                                            transform=transform,
                                            download=True)

    testset = torchvision.datasets.FashionMNIST(data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def load_celeba(data_path,batch_size):
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.CelebA(data_path,
                                            split='train',
                                            transform=transform,
                                            download=True)
    testset = torchvision.datasets.CelebA(data_path,
                                            split='test',
                                            transform=transform,
                                            download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, testloader

def load_cifar10(data_path,batch_size):
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.CIFAR10(data_path,
                                            train=True,
                                            transform=transform,
                                            download=True)

    testset = torchvision.datasets.CIFAR10(data_path,
                                            train=False,
                                            transform=transform,
                                            download=True)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

class mvtecImageDataset(Dataset):
    def __init__(self, img_dir, split, transform):
        self.img_lables = pd.read_csv(img_dir+'/mvtec/transistor/'+split+'/'+split+'.csv')
        self.img_dir = img_dir
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.img_lables)

    def __getitem__(self, idx):
        img_path = self.img_dir+'/mvtec/transistor/'+self.split+'/'+self.img_lables.iloc[idx, 0]
        image = PIL.Image.open(img_path)
        label = self.img_lables.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_mvtec(data_path, batch_size):

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = mvtecImageDataset(img_dir=data_path, split='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=0, pin_memory=True)

    testset = mvtecImageDataset(img_dir=data_path, split='test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=0, pin_memory=True)
    return trainloader, testloader