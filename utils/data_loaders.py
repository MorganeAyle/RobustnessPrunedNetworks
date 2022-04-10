import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
import pickle

from utils.constants import NUM_WORKERS, FLIP_CHANCE, DATASET_PATH

"""
Handles loading datasets
"""


def get_mnist_loaders(arguments, mean=(0.5,), std=(0.5,), val=False):
    print("Using mean", mean)
    transformers = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.MNIST(
        DATASET_PATH,
        train=True,
        download=True,
        transform=transformers
    )
    test_set = datasets.MNIST(
        DATASET_PATH,
        train=False,
        download=True,
        transform=transformers
    )
    return load(arguments, test_set, train_set, val)


def get_unnormliazed_mnist_loaders(arguments, mean=(0.5,), std=(0.5,), val=False):
    print("Using mean", mean)
    transformers = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        DATASET_PATH,
        train=True,
        download=True,
        transform=transformers
    )
    test_set = datasets.MNIST(
        DATASET_PATH,
        train=False,
        download=True,
        transform=transformers
    )
    return load(arguments, test_set, train_set, val), mean, std


def get_fashionmnist_loaders(arguments, mean=(0.5,), std=(0.5,), val=False):
    print("Using mean", mean)
    transformers = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean, std)
                                       ])
    train_set = datasets.FashionMNIST(
        DATASET_PATH,
        train=True,
        download=True,
        transform=transformers
    )
    test_set = datasets.FashionMNIST(
        DATASET_PATH,
        train=False,
        download=True,
        transform=transformers
    )
    return load(arguments, test_set, train_set, val)


def get_unnormalized_fashionmnist_loaders(arguments, mean=(0.5,), std=(0.5,), val=False):
    print("Using mean", mean)
    transformers = transforms.Compose([transforms.ToTensor()
                                       ])
    train_set = datasets.FashionMNIST(
        DATASET_PATH,
        train=True,
        download=True,
        transform=transformers
    )
    test_set = datasets.FashionMNIST(
        DATASET_PATH,
        train=False,
        download=True,
        transform=transformers
    )
    return load(arguments, test_set, train_set, val), mean, std


def get_kmnist_loaders(arguments, mean=(0.5,), std=(0.5,), val=False):
    print("Using mean", mean)
    # (0.1918,), (0.3483,)
    transformers = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean, std)
                                       ])
    train_set = datasets.KMNIST(
        DATASET_PATH,
        train=True,
        download=True,
        transform=transformers
    )
    test_set = datasets.KMNIST(
        DATASET_PATH,
        train=False,
        download=True,
        transform=transformers
    )
    return load(arguments, test_set, train_set, val)


def get_omniglot_loaders(arguments, mean=(0.5,), std=(0.5,), val=False):
    print("Using mean", mean)
    # (0.92206,), (0.08426,)
    if arguments['preload_all_data']: raise NotImplementedError

    train_set = datasets.Omniglot(DATASET_PATH, background=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std)
                                  ]))

    test_set = datasets.Omniglot(DATASET_PATH, background=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
                                 ]))

    return load(arguments, test_set, train_set, val)


def get_cifar10_loaders(arguments, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), val=False):
    print("Using mean", mean)
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_set = datasets.CIFAR10(DATASET_PATH, train=True, transform=transform, download=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_set = datasets.CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)
    return load(arguments, test_set, train_set, val)


def get_unnormalized_cifar10_loaders(arguments, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), val=False):
    print("Using mean", mean)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_set = datasets.CIFAR10(DATASET_PATH, train=True, transform=transform, download=True)
    test_set = datasets.CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)
    return load(arguments, test_set, train_set, val), mean, std


def get_cifar100_loaders(arguments, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), val=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_training = torchvision.datasets.CIFAR100(root=DATASET_PATH, train=True, download=True,
                                                      transform=transform_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    cifar100_test = torchvision.datasets.CIFAR100(root=DATASET_PATH, train=False, download=True,
                                                  transform=transform_test)

    return load(arguments, cifar100_test, cifar100_training, val)


def get_unnormalized_cifar100_loaders(arguments, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), val=False):
    print("Using mean", mean)
    if arguments['preload_all_data']: raise NotImplementedError

    train_set = datasets.CIFAR100(DATASET_PATH, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(32, padding=4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                  ]))

    test_set = datasets.CIFAR100(DATASET_PATH, train=False, download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))

    return load(arguments, test_set, train_set, val), mean, std


class MyData(Dataset):
    def __init__(self, mean, std):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.transformations = [None, transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0),
                                "rotation90", "rotation180", "rotation270"]
        self.images = datasets.CIFAR10(DATASET_PATH, train=True, download=True)

    def __getitem__(self, index):
        image, y = self.images[index]
        transformation = random.choice(self.transformations)
        if transformation is not None:
            if type(transformation) == str:
                image = self.transform(image)
                image = torch.rot90(image, 0, [1, 2])
                if not transformation.endswith("90"):
                    image = torch.rot90(image, 0, [1, 2])
                    if transformation.endswith("270"):
                        image = torch.rot90(image, 0, [1, 2])
            else:
                image = transformation(image)
                image = self.transform(image)
        else:
            image = self.transform(image)

        return image.to(torch.float32), torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.images)


def get_custom_cifar10_loaders(arguments, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), val=False):
    print("Using mean", mean)
    train_set = MyData(mean, std)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_set = datasets.CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)
    return load(arguments, test_set, train_set, val)


def get_svhn_loaders(arguments, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), val=False):
    print("Using mean", mean)
    # (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
    transformers = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean, std)
                                       ])
    train_set = datasets.SVHN(
        DATASET_PATH,
        split='train',
        download=True,
        transform=transformers
    )
    test_set = datasets.SVHN(
        DATASET_PATH,
        split='test',
        download=True,
        transform=transformers
    )
    return load(arguments, test_set, train_set, val)


def get_lsun_loaders(arguments, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), val=False):
    print("Using mean", mean)
    transformers = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_set = datasets.LSUN(root=DATASET_PATH, classes='test',
                             transform=transformers)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=arguments['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    if val:
        return None, None, test_loader
    return None, test_loader


def preloading(arguments, test_set, train_set):
    print("preloading data")
    train_images, train_labels = zip(*train_set)
    test_images, test_labels = zip(*test_set)
    train_images = torch.stack(train_images, dim=0).to(arguments['device'])
    train_labels = torch.tensor(train_labels).to(arguments['device'])
    test_images = torch.stack(test_images, dim=0).to(arguments['device'])
    test_labels = torch.tensor(test_labels).to(arguments['device'])
    # noinspection PyTypeChecker
    train_loader, test_loader = PersonalDataLoader(train_images, train_labels, arguments['batch_size'],
                                                   horizontal_flips=True, device=arguments['device']), \
                                PersonalDataLoader(test_images, test_labels, arguments['batch_size'],
                                                   device=arguments['device'])
    return test_loader, train_loader


def load(arguments, test_set, train_set, val):
    if arguments['preload_all_data']:
        if val:
            raise NotImplementedError
        test_loader, train_loader = preloading(arguments, test_set, train_set)
        return train_loader, test_loader
    else:
        if not val:
            test_loader, train_loader = traditional_loading(arguments, test_set, train_set, val)
            return train_loader, test_loader
        else:
            test_loader, val_loader, train_loader = traditional_loading(arguments, test_set, train_set, val)
            return train_loader, val_loader, test_loader


def traditional_loading(arguments, test_set, train_set, val):
    workers = NUM_WORKERS
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=arguments['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=workers
    )

    if val:
        train_set_size = len(train_set)
        new_train_set_size = int(len(train_set) * 0.8)
        new_train_set, val_set = torch.utils.data.random_split(train_set, [new_train_set_size, train_set_size - new_train_set_size])

        train_loader = torch.utils.data.DataLoader(
            new_train_set,
            batch_size=arguments['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=arguments['batch_size'],
            shuffle=False,
            pin_memory=True,
            num_workers=workers
        )
        return test_loader, val_loader, train_loader

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=arguments['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=workers
    )

    return test_loader, train_loader


def get_rubbish_loaders(arguments=None):
    bs = 10 if arguments is None else arguments['batch_size']
    train_loader = torch.utils.data.DataLoader(
        RubbishSet(),
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS

    )

    test_loader = torch.utils.data.DataLoader(
        RubbishSet(),
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS

    )

    return train_loader, test_loader


def get_gaussian_noise_loaders(arguments=None):
    bs = arguments['batch_size']
    train_loader = torch.utils.data.DataLoader(
        GaussianNoise(arguments, train=True),
        batch_size=bs,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS

    )

    test_loader = torch.utils.data.DataLoader(
        GaussianNoise(arguments, train=False),
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS

    )

    return train_loader, test_loader


def change_range(x):
    return x * 255


def get_oodomain_loaders(arguments=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    print("In OOD domain loader")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(change_range),
        ]
    )

    if (arguments["data_set"] == "CIFAR10") or (arguments["data_set"] == "CIFAR100"):
        train_set = datasets.SVHN(
            DATASET_PATH,
            split='train',
            download=True,
            transform=transform
        )
        test_set = datasets.SVHN(
            DATASET_PATH,
            split='test',
            download=True,
            transform=transform
        )
    elif arguments["data_set"] == "FASHION":
        train_set = datasets.MNIST(
            DATASET_PATH,
            train=True,
            download=True,
            transform=transform
        )
        test_set = datasets.MNIST(
            DATASET_PATH,
            train=False,
            download=True,
            transform=transform
        )
    else:
        raise NotImplementedError(f"OODomain loader not implemented for {arguments['data_set']}")
    return load(arguments, test_set, train_set)
    # bs = arguments['batch_size']
    # train_loader = torch.utils.data.DataLoader(
    #     OODomain(arguments, train=True),
    #     batch_size=bs,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=NUM_WORKERS
    #
    # )
    #
    # test_loader = torch.utils.data.DataLoader(
    #     OODomain(arguments, train=False),
    #     batch_size=bs,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=NUM_WORKERS
    #
    # )

    # return train_loader, test_loader


#### Classes

class CIFAR10C(Dataset):

    def __init__(self, images, labels, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        self.images = images
        self.labels = labels
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ]
        )

    def __getitem__(self, item):
        image = self.images[item] / 255
        image = self.transforms(image.transpose((1, 2, 0)))
        return image.to(torch.float32), torch.tensor(self.labels[item], dtype=torch.float32)

    def __len__(self):
        return len(self.images)


class CIFAR10CU(Dataset):

    def __init__(self, images, labels, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
        self.images = images
        self.labels = labels
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, item):
        image = self.images[item] / 255
        image = self.transforms(image.transpose((1, 2, 0)))
        return image.to(torch.float32), torch.tensor(self.labels[item], dtype=torch.float32)

    def __len__(self):
        return len(self.images)


class RubbishSet(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, item):
        class_ = random.choice([0, 1])
        tensor = np.random.normal(class_, 0.2, (3, 3))
        return tensor, class_

    def __len__(self):
        return 10000


class GaussianNoise(Dataset):

    def __init__(self, arguments, train):
        self.args = arguments
        self.train = train

        if self.args['input_dim'] == [1, 28, 28] and self.args['output_dim'] == 10:
            path = '/nfs/students/ayle/guided-research/gitignored/data/random_1x28x28x10'
            self.reshape = [1, 1, 1]
        elif self.args['input_dim'] == [3, 32, 32] and self.args['output_dim'] == 10:
            path = '/nfs/students/ayle/guided-research/gitignored/data/random_3x32x32x10'
            self.reshape = [3, 1, 1]
        else:
            raise NotImplementedError

        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        if self.train:
            self.set = dataset['train_set']
        else:
            self.set = dataset['test_set']

        self.mean = dataset['mean']
        self.std = dataset['std']

    def __getitem__(self, item):
        tensor, class_ = self.set[item]

        tensor = (tensor - np.reshape(self.mean, self.reshape)) / np.reshape(self.std, self.reshape)
        tensor = tensor.astype(float)

        return tensor, class_

    def __len__(self):
        return len(self.set)


class OODomain(Dataset):

    def __init__(self, arguments, train):
        self.args = arguments
        self.train = train

        if self.args['input_dim'] == [1, 28, 28] and self.args['output_dim'] == 10:
            path = '/nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/data/random_1x28x28x10'
            self.reshape = [1, 1, 1]
        elif self.args['input_dim'] == [3, 32, 32] and self.args['output_dim'] == 10:
            path = '/nfs/homedirs/ayle/guided-research/SNIP-it/gitignored/data/random_3x32x32x10'
            self.reshape = [3, 1, 1]
        else:
            raise NotImplementedError

        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        if self.train:
            self.set = dataset['train_set']
        else:
            self.set = dataset['test_set']

        self.mean = dataset['mean']
        self.std = dataset['std']

    def __getitem__(self, item):
        tensor, class_ = self.set[item]

        tensor = tensor.astype(float)

        return tensor, class_

    def __len__(self):
        return len(self.set)


class PersonalDataLoader:

    # def _reset(self):
    def __init__(self, data, labels, batch_size: int, device="cuda", horizontal_flips=False):
        self.horizontal_flips = horizontal_flips
        self.labels = labels
        self.batch_size = batch_size
        self.data = data
        self.device = device
        self.length = torch.LongTensor([(len(data) // batch_size) + 1]).to(device)
        self.length_orgi = len(data)
        self.indices = None
        self.flips = None
        self.counter = torch.zeros([1], device=self.device).long()
        self.one = torch.ones([1], device=self.device).long()

    def __len__(self):
        return self.length

    def __iter__(self):
        self.indices = (
                torch.randperm(self.length.item() * self.batch_size, device=self.device) % self.length_orgi).view(
            -1, self.batch_size)
        if self.horizontal_flips:
            self.flips = torch.bernoulli(torch.empty(self.data.shape[0]).to(self.device), p=FLIP_CHANCE).bool()
            self.data[self.flips] = self.data[self.flips].flip(-1)
        self.counter -= self.counter
        return self

    def __next__(self):
        if self.counter >= self.length:
            if self.horizontal_flips:
                self.data[self.flips] = self.data[self.flips].flip(-1)
            raise StopIteration
        else:
            batch = self.__getitem__(self.counter)
            self.counter += self.one
            return batch

    def __getitem__(self, item):
        return self.data[
                   self.indices[item].squeeze()
               ], \
               self.labels[
                   self.indices[item].squeeze()
               ]
