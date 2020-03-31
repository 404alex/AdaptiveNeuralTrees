"""Data loader"""
import torch
import torchvision
from torchvision import datasets, transforms
from ops import ChunkSampler
from imbalanced import ImbalancedDatasetSampler
import numpy as np
from torch.utils import data as tu
from torch._utils import _accumulate
from torch import randperm
from sklearn.preprocessing import normalize
from helper import Subset



def random_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(_accumulate(lengths), lengths)]


def get_dataloaders(
        dataset='mnist',
        batch_size=128,
        augmentation_on=False,
        cuda=False, num_workers=0,
):
    # TODO: move the dataloader to data.py
    kwargs = {
        'num_workers': num_workers, 'pin_memory': True,
    } if cuda else {}

    if dataset == 'mnist':
        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=2),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )

        mnist_train = datasets.MNIST(
            '../data', train=True, download=True, transform=transform_train,
        )
        mnist_valid = datasets.MNIST(
            '../data', train=True, download=True, transform=transform_test,
        )
        mnist_test = datasets.MNIST(
            '../data', train=False, transform=transform_test,
        )

        TOTAL_NUM = 60000
        NUM_VALID = int(round(TOTAL_NUM * 0.1))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))

        train_loader = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)

        valid_loader = torch.utils.data.DataLoader(
            mnist_valid,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)

        test_loader = torch.utils.data.DataLoader(
            mnist_test,
            batch_size=1000,
            shuffle=False,
            **kwargs)

    elif dataset == 'cifar10':
        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )

        cifar10_train = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True,
            transform=transform_train,
        )
        cifar10_valid = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform_test,
        )
        cifar10_test = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True,
            transform=transform_test,
        )

        TOTAL_NUM = 50000
        NUM_VALID = int(round(TOTAL_NUM * 0.02))
        NUM_TRAIN = int(round(TOTAL_NUM - NUM_VALID))
        train_loader = torch.utils.data.DataLoader(
            cifar10_train,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            cifar10_valid,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            cifar10_test,
            batch_size=1000,
            shuffle=False,
            **kwargs)
    elif dataset == 'iot':
        if augmentation_on:
            transform_train = transforms.Compose(
                [
                    transforms.Resize((40, 40)),
                    transforms.RandomCrop(40, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize((40, 40)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
                    ),
                ],
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.Resize((40, 40)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )
            transform_test = transforms.Compose(
                [
                    transforms.Resize((40, 40)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )
        train_data = datasets.ImageFolder(root="../img", transform=transform_train)
        test_data = datasets.ImageFolder(root="../img", transform=transform_test)
        val_data = datasets.ImageFolder(root="../img", transform=transform_test)

        TOTAL_NUM = 551
        NUM_VALID = 11
        NUM_TRAIN = TOTAL_NUM - NUM_VALID

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_TRAIN, 0, shuffle=True),
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            sampler=ChunkSampler(NUM_VALID, NUM_TRAIN, shuffle=True),
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=256,
            shuffle=False,
            **kwargs)

        # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
        # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,  shuffle=False, **kwargs)
        # valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,  shuffle=True, **kwargs)

    elif dataset == 'batadal':
        filename = '../data/ICS/BATADAL/Physical_BATADAL.npy'
        data = np.load(filename, allow_pickle=True)
        datalen = len(data[0])
        X = data[:, 1:datalen - 2]
        X = normalize(X, axis=0)
        y = data[:, datalen - 1]
        re_X = []
        for item in X:
            z = np.zeros(6)
            item = np.concatenate((item, z), axis=0)
            temp = np.reshape(item, (1, 7, 7)).tolist()
            re_X.append(np.array(temp))
        re_y = []
        for item in y:
            if item == 'Normal':
                temp = 0
            else:
                temp = 1
            re_y.append(temp)

        tensor_x = torch.Tensor(re_X)
        tensor_y = torch.Tensor(re_y).long()

        all_dataset = tu.TensorDataset(tensor_x, tensor_y)
        total_num = len(y)
        val_num = int(round(total_num * 0.1))
        train_num = total_num - val_num - 2000
        tran_set, test_set, val_set = random_split(all_dataset, [train_num, 2000, val_num])


        NUM_VALID = len(val_set)
        NUM_TRAIN = len(tran_set)

        train_loader = torch.utils.data.DataLoader(
            tran_set,
            batch_size=batch_size,

            sampler=ImbalancedDatasetSampler(tran_set),
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            sampler=ImbalancedDatasetSampler(val_set),
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=2000,
            shuffle=False,
            **kwargs)

    elif dataset == 'wadi':
        filename = '../data/ICS/WADI/Physical_WADI.npy'
        data = np.load(filename, allow_pickle=True)
        datalen = len(data[0])
        X = data[:, 3:datalen - 2]
        X = normalize(X, axis=0)
        y = data[:, datalen - 1]
        re_X = []
        for item in X:
            z = np.zeros(22)
            item = np.concatenate((item, z), axis=0)
            temp = np.reshape(item, (1, 12, 12)).tolist()
            re_X.append(np.array(temp))
        re_y = []
        for item in y:
            if item == 'Normal':
                temp = 0
            else:
                temp = 1
            re_y.append(temp)

        tensor_x = torch.Tensor(re_X)
        tensor_y = torch.Tensor(re_y).long()

        all_dataset = tu.TensorDataset(tensor_x, tensor_y)
        total_num = len(y)
        val_num = int(round(total_num * 0.2))
        train_num = total_num - val_num - val_num
        tran_set, test_set, val_set = random_split(all_dataset, [train_num, val_num, val_num])


        NUM_VALID = len(val_set)
        NUM_TRAIN = len(tran_set)

        train_loader = torch.utils.data.DataLoader(
            tran_set,
            batch_size=batch_size,
            sampler=ImbalancedDatasetSampler(tran_set),
            **kwargs)
        valid_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            sampler=ImbalancedDatasetSampler(val_set),
            **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=val_num,
            shuffle=False,
            **kwargs)


    else:
        raise NotImplementedError("Specified data set is not available.")

    return train_loader, valid_loader, test_loader, NUM_TRAIN, NUM_VALID


def get_dataset_details(dataset):
    if dataset == 'mnist':
        input_nc, input_width, input_height = 1, 28, 28
        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    elif dataset == 'cifar10':
        input_nc, input_width, input_height = 3, 32, 32
        classes = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck',
        )
    elif dataset == 'iot':
        input_nc, input_width, input_height = 3, 40, 40
        classes = (
            'benign_img', 'malicious_img'
        )
    elif dataset == 'batadal':
        input_nc, input_width, input_height = 1, 7, 7
        classes = (
            'Normal', 'Attack'
        )
    elif dataset == 'wadi':
        input_nc, input_width, input_height = 1, 12, 12
        classes = (
            'Normal', 'Attack'
        )
    else:
        raise NotImplementedError("Specified data set is not available.")

    return input_nc, input_width, input_height, classes
