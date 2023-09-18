import os
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_dataset(data_name, path='./data'):
    if not data_name in ['mnist', 'cifar10']:
        raise TypeError('data_name should be a string, including mnist,cifar10. ')

    # model: 2 conv. layers followed by 2 FC layers
    if (data_name == 'mnist'):
        trainset = datasets.MNIST(path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
        testset = datasets.MNIST(path, train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))

    # model: ResNet-50
    elif (data_name == 'cifar10'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root=path, train=True,
                                    download=True, transform=transform)
        testset = datasets.CIFAR10(root=path, train=False,
                                   download=True, transform=transform)
    return trainset, testset


def get_dataloader(trainset, testset, batch_size, device):
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def split_class_data(dataset, forget_class, num_forget):
    forget_index = []
    class_remain_index = []
    remain_index = []
    sum = 0
    for i, (data, target) in enumerate(dataset):
        if target == forget_class and sum < num_forget:
            forget_index.append(i)
            sum += 1
        elif target == forget_class and sum >= num_forget:
            class_remain_index.append(i)
            remain_index.append(i)
            sum += 1
        else:
            remain_index.append(i)
    return forget_index, remain_index, class_remain_index


# def split_dataset(dataset, forget_class):


def get_unlearn_loader(trainset, testset, forget_class, batch_size, num_forget, repair_num_ratio=0.01):
    train_forget_index, train_remain_index, class_remain_index = split_class_data(trainset, forget_class,
                                                                                  num_forget=num_forget)
    test_forget_index, test_remain_index, _ = split_class_data(testset, forget_class, num_forget=len(testset))

    repair_class_index = random.sample(class_remain_index, int(repair_num_ratio * len(class_remain_index)))

    train_forget_sampler = SubsetRandomSampler(train_forget_index)  # 5000
    train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 45000

    repair_class_sampler = SubsetRandomSampler(repair_class_index)

    test_forget_sampler = SubsetRandomSampler(test_forget_index)  # 1000
    test_remain_sampler = SubsetRandomSampler(test_remain_index)  # 9000

    train_forget_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_forget_sampler)
    train_remain_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=train_remain_sampler)

    repair_class_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                                      sampler=repair_class_sampler)

    test_forget_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_forget_sampler)
    test_remain_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                                     sampler=test_remain_sampler)

    return train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
           train_forget_index, train_remain_index, test_forget_index, test_remain_index


def get_forget_loader(dt, forget_class):
    idx = []
    els_idx = []
    count = 0
    for i in range(len(dt)):
        _, lbl = dt[i]
        if lbl == forget_class:
            # if forget:
            #     count += 1
            #     if count > forget_num:
            #         continue
            idx.append(i)
        else:
            els_idx.append(i)
    forget_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(idx), drop_last=True)
    remain_loader = torch.utils.data.DataLoader(dt, batch_size=8, shuffle=False,
                                                sampler=torch.utils.data.SubsetRandomSampler(els_idx), drop_last=True)
    return forget_loader, remain_loader
