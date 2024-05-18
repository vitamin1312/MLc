import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from hyperparameters import device


def get_transforms():

    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
                                    ])
    to_pil_image = transforms.ToPILImage()

    return {'transform': transform,
            'to_pil_image': to_pil_image}


def get_dataset(transform):

    dataset = datasets.MNIST(
        root='../input/data',
        train=True,
        download=True,
        transform=transform
    )

    return dataset


def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)

def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)

def create_noise(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)
