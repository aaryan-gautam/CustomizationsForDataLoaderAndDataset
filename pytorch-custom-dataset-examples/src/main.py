import os

import torch
import torchvision.transforms as tt
import torch.nn as nn
import time
import opendatasets as od
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset  # For custom datasets

from src.custom_dataset_from_http import CustomDatasetFromHTTP
from src.custom_dataset_from_remote_cifar import CustomDatasetFromFetchCifar
from src.custom_dataset_from_remote_cifar_test import CustomDatasetFromFetchCifarTest
from src.cifar_model import ResNet

# in-repo imports
from custom_dataset_from_csv import CustomDatasetFromCsvLocation, CustomDatasetFromCsvData
from custom_dataset_from_file import CustomDatasetFromFile
from custom_dataset_from_remote_server import CustomDatasetFromFetch
from cnn_model import MnistCNNModel
from torchvision.datasets import ImageFolder
from odd_even_sampler import OddEvenSampler
from random_sampler import RandomSampler

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])

    custom_random_sampler = RandomSampler('dataset')


    # train_mnist = datasets.MNIST("", train=True, download=True, transform=transforms.ToTensor(), )

    data_dir = './archive/cifar-100-images/CIFAR100'

    custom_mnist_from_file = \
        CustomDatasetFromFile('../fetched/mnist_with_class/')


    # dataset_url = 'https://www.kaggle.com/minbavel/cifar-100-images'
    # od.download(dataset_url)
    data_dir = 'archive/CIFAR100'
    classes = os.listdir(data_dir + "/TRAIN")

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                             tt.RandomHorizontalFlip(),
                             tt.ToTensor(),
                             tt.Normalize(*stats, inplace=True)
                             ])
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)
                             ])

    # PyTorch datasets
    train_ds = ImageFolder(data_dir + '/TRAIN', train_tfms)
    # valid_ds = ImageFolder(data_dir + '/TEST', valid_tfms)

    # Let's define batch size
    batch_size = 128

    # mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_file,
    #                                                 batch_size=10,
    #                                                 shuffle=True, prefetch_factor=10, num_workers=1)

    # reusing metadata by calling init for http
    custom_cifar_from_server_train = CustomDatasetFromFetchCifar('archive/CIFAR100/TRAIN')
    custom_http_server = CustomDatasetFromHTTP('archive/CIFAR100/TRAIN', custom_random_sampler.shuffler_order_tracker)
    # custom_cifar_from_server_test = CustomDatasetFromFetchCifarTest('archive/CIFAR100TEST')


    # PyTorch data loaders
    # local file system trained with train_ds
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=60, shuffle=True, num_workers=3, pin_memory=True)
    # valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size * 2, num_workers=3, pin_memory=True)
    # http server trained with httpserver
    train_dl = torch.utils.data.DataLoader(dataset=custom_http_server, batch_size=60, sampler=custom_random_sampler)
    # valid_dl = torch.utils.data.DataLoader(dataset=custom_cifar_from_server_test, batch_size=20,
    #                                        pin_memory=True)

    model = ResNet(3, 100)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



    @torch.no_grad()
    def evaluate(model, train_dl):
        model.eval()
        outputs = [model.validation_step(batch) for batch in train_dl]
        return model.validation_epoch_end(outputs)


    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)


    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                      weight_decay=0, grad_clip=None, opt_func=torch.optim.Adam):
        torch.cuda.empty_cache()
        history = []

        # Set up cutom optimizer with weight decay
        optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
        # Set up one-cycle learning rate scheduler
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))

        for epoch in range(epochs):
            # print(epoch)
            # print("epoch started")
            # Training Phase
            # model.train()
            train_losses = []
            lrs = []
            training_start_time = time.time()
            counter_for_batch = 0
            print(train_loader.__len__())
            batch_getting_time = time.time()
            for batch in train_loader:
                print('batch loading took {:.2f}s', format(time.time() - batch_getting_time))
                print("batch_counter=")
                print(counter_for_batch)
                counter_for_batch += 1
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # Gradient clipping
                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Record & update learning rate
                lrs.append(get_lr(optimizer))
                sched.step()
                print('one iteration took {:.2f}s', format(time.time() - training_start_time))
            print('Training finished, took {:.2f}s',format(time.time() - training_start_time))

            # Validation phase
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['lrs'] = lrs
            model.epoch_end(epoch, result)
            history.append(result)
        return history

    history = []

    epochs = 1
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    history += fit_one_cycle(epochs, max_lr, model, train_dl, train_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)



    print('program has completed')
