import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch
import global_v as glv
from torch.utils.data.dataset import Dataset
import pandas as pd
import PIL
from data.sampler import FederatedSampler

def load_mnist(data_path):
    print("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return trainloader, testloader

def load_fashionmnist(data_path):
    print("loading Fashion MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.FashionMNIST(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_cifar10(data_path):
    print("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    transform_test = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True)
    return trainloader, testloader

def load_celebA(data_path):
    print("loading CelebA")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']
    
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        SetRange])

    trainset = torchvision.datasets.CelebA(root=data_path, 
                                            split='train', 
                                            download=True, 
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True, num_workers=8, pin_memory=True)

    testset = torchvision.datasets.CelebA(root=data_path, 
                                            split='test', 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=8, pin_memory=True)
    return trainloader, testloader

class mvtecImageDataset(Dataset):
    def __init__(self, img_dir, split, transform):
        self.img_labels = pd.read_csv(img_dir+'/mvtec/transistor/'+split+'/'+split+'.csv')
        self.img_dir = img_dir
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir+'/mvtec/transistor/'+self.split+'/'+self.img_labels.iloc[idx, 0]
        image = PIL.Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_mvtec(data_path, non_iid, n_clients, n_shards):
    print("loading mvtec")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = glv.network_config['batch_size']
    input_size = glv.network_config['input_size']

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.CenterCrop(148),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        SetRange
    ])

    # trainset = torchvision.datasets.CelebA(root=data_path,
    #                                        split='train',
    #                                        download=True,
    #                                        transform=transform)
    trainset = mvtecImageDataset(img_dir=data_path, split='train', transform=transform)
    sampler = FederatedSampler(
        trainset, non_iid=non_iid, n_clients=n_clients, n_shards=n_shards
    )
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              # shuffle=True,
                                              num_workers=0, pin_memory=True, sampler=sampler)

    # testset = torchvision.datasets.CelebA(root=data_path,
    #                                       split='test',
    #                                       download=True,
    #                                       transform=transform)
    testset = mvtecImageDataset(img_dir=data_path, split='test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False, num_workers=0, pin_memory=True)
    return trainloader, testloader

