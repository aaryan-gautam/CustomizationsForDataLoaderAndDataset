import http
import ssl
import time
from io import BytesIO

import numpy as np
import requests
from PIL import Image
import base64
import os
import paramiko
import csv
from stat import S_ISDIR as isdir
import sys

import torch



from torchvision import transforms
from torch.utils.data.dataset import Dataset, TensorDataset  # For custom datasets
from torchvision.datasets import ImageFolder

from src.custom_dataset_from_remote_cifar import CustomDatasetFromFetchCifar
from src.random_sampler import RandomSampler


class CustomDatasetFromHTTP(Dataset):

    def __init__(self, path):
        self.all_dir = all_dir = ['apple', 'aquarium_fish','baby' ,'bear' ,'beaver' ,'bed' ,'bee' ,'beetle' ,'bicycle' ,'bottle' ,'bowl' ,'boy' ,'bridge' ,'bus' ,'butterfly' ,'camel' ,'can' ,'castle' ,'caterpillar' ,'cattle' ,'chair' ,'chimpanzee' ,'clock' ,'cloud' ,'cockroach' ,'couch' ,'crab' ,'crocodile' ,'cup' ,'dinosaur' ,'dolphin' ,'elephant' ,'flatfish' ,'forest' ,'fox' ,'girl' ,'hamster' ,'house' ,'kangaroo' ,'keyboard' ,'lamp' ,'lawn_mower' ,'leopard' ,'lion' ,'lizard' ,'lobster' ,'man' ,'maple_tree' ,'motorcycle' ,'mountain' ,'mouse' ,'mushroom' ,'oak_tree' ,'orange' ,'orchid' ,'otter' ,'palm_tree' ,'pear' ,'pickup_truck' ,'pine_tree' ,'plain' ,'plate' ,'poppy' ,'porcupine' ,'possum' ,'rabbit' ,'raccoon' ,'ray' ,'road' ,'rocket' ,'rose' ,'sea' ,'seal' ,'shark' ,'shrew' ,'skunk' ,'skyscraper' ,'snail' ,'snake' ,'spider' ,'squirrel' ,'streetcar' ,'sunflower' ,'sweet_pepper' ,'table' ,'tank' ,'telephone' ,'television' ,'tiger' ,'tractor' ,'train' ,'trout' ,'tulip' ,'turtle' ,'wardrobe' ,'whale' ,'willow_tree' ,'wolf' ,'woman' ,'worm']
        # start_time = time.time()
        # response = requests.get("http://127.0.0.1:8000/")
        # print('it took took {:.2f}s', format(time.time() - start_time))
        obj1 = CustomDatasetFromFetchCifar('filepath')
        self.filepaths = obj1.file_record
        # stores index val and image object pairs (can be changed to reduce storage space)
        self.fetched_list = dict()

    def __getitem__(self, index):
        hostname = "http://127.0.0.1:8000/"
        print(RandomSampler.get_tracker())
        # meta data strings are parsed from character 39 onwards since they contain info regarding the ssh server location
        # prior to character 39 which is not needed for fetching requests.
        # Currently only the sftp init method is used to reuse metadata
        if self.fetched_list.__len__() == 0 or not index in self.fetched_list:
            print(RandomSampler.shuffler_order_tracker)
            for i in range(25):
                # it has the list of records that are going to be fetched
                curr_index_position = RandomSampler.shuffler_order_tracker.index(index)
                if curr_index_position + i + 1 >= RandomSampler.shuffler_order_tracker.__len__():
                    break
                index_to_fetch = RandomSampler.shuffler_order_tracker[curr_index_position + i + 1]
                self.fetched_list[index_to_fetch] = requests.get(hostname + self.filepaths[index][39:])
        # response = requests.get(hostname + self.filepaths[index][39:])
        response = self.fetched_list.get(index)
        img = Image.open(BytesIO(response.content))
        im_as_im = img
        im_as_np = np.asarray(im_as_im) / 255
        directory_name = self.filepaths[index][39:].split('/')[0]
        label = self.all_dir.index(directory_name)
        im_as_ten = torch.from_numpy(im_as_np).float()
        im_as_ten = torch.reshape(im_as_ten, [3, 32, 32])

        return im_as_ten, label


    def __len__(self):
        return 50000
