import http
import queue
import ssl
import threading
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

    def __init__(self, path, RandomSamplerOrder):
        self.all_dir = all_dir = ['apple', 'aquarium_fish','baby' ,'bear' ,'beaver' ,'bed' ,'bee' ,'beetle' ,'bicycle' ,'bottle' ,'bowl' ,'boy' ,'bridge' ,'bus' ,'butterfly' ,'camel' ,'can' ,'castle' ,'caterpillar' ,'cattle' ,'chair' ,'chimpanzee' ,'clock' ,'cloud' ,'cockroach' ,'couch' ,'crab' ,'crocodile' ,'cup' ,'dinosaur' ,'dolphin' ,'elephant' ,'flatfish' ,'forest' ,'fox' ,'girl' ,'hamster' ,'house' ,'kangaroo' ,'keyboard' ,'lamp' ,'lawn_mower' ,'leopard' ,'lion' ,'lizard' ,'lobster' ,'man' ,'maple_tree' ,'motorcycle' ,'mountain' ,'mouse' ,'mushroom' ,'oak_tree' ,'orange' ,'orchid' ,'otter' ,'palm_tree' ,'pear' ,'pickup_truck' ,'pine_tree' ,'plain' ,'plate' ,'poppy' ,'porcupine' ,'possum' ,'rabbit' ,'raccoon' ,'ray' ,'road' ,'rocket' ,'rose' ,'sea' ,'seal' ,'shark' ,'shrew' ,'skunk' ,'skyscraper' ,'snail' ,'snake' ,'spider' ,'squirrel' ,'streetcar' ,'sunflower' ,'sweet_pepper' ,'table' ,'tank' ,'telephone' ,'television' ,'tiger' ,'tractor' ,'train' ,'trout' ,'tulip' ,'turtle' ,'wardrobe' ,'whale' ,'willow_tree' ,'wolf' ,'woman' ,'worm']
        # start_time = time.time()
        # response = requests.get("http://127.0.0.1:8000/")
        # print('it took took {:.2f}s', format(time.time() - start_time))
        self.data_order = RandomSamplerOrder
        obj1 = CustomDatasetFromFetchCifar('filepath')
        self.filepaths = obj1.file_record
        # stores index val and image object pairs (can be changed to reduce storage space)
        self.position_tracker = 0
        self.fetched_dict = list()
        self.total_time = 0
        print("running this")
        for i in range(175):
            self.fetched_dict.append('a')
        self.thread_queue = queue.Queue()


    def __getitem__(self, index):
        self.part_time = time.time()
        self.hostname = "http://127.0.0.1:8000/"
        # print(self.data_order)
        # meta data strings are parsed from character 39 onwards since they contain info regarding the ssh server location
        # prior to character 39 which is not needed for fetching requests.
        # Currently only the sftp init method is used to reuse metadata
        # if self.fetched_dict.__len__() == 0 or not index in self.fetched_dict:
        list_time = time.time()
        # if not [item for item in self.fetched_dict if not item[0] == index]:
        #     print('hola')
        temp_flag = False
        # if self.fetched_dict.__len__() == 0 or not [item for item in self.fetched_dict if item[0] == index]:
        # print("size of queue is:", format(self.thread_queue.qsize()))
        if self.thread_queue.empty() or self.thread_queue.qsize() % 25 == 0:
            self.position_tracker = 0
            a_thread = threading.Thread(target=self.fetch_from_thread, args=(index, ))
            a_thread.start()
            if self.thread_queue.empty():
                a_thread.join()
        response = self.thread_queue.get(block=True, timeout=0)
        self.position_tracker += 1
        # comment above for no prefetch
        # response = requests.get(hostname + self.filepaths[index][39:]).content
        img = Image.open(BytesIO(response))
        im_as_im = img
        im_as_np = np.asarray(im_as_im) / 255
        directory_name = self.filepaths[index][39:].split('/')[0]
        label = self.all_dir.index(directory_name)
        im_as_ten = torch.from_numpy(im_as_np).float()
        im_as_ten = torch.reshape(im_as_ten, [3, 32, 32])

        return im_as_ten, label


    def __len__(self):
        return 50000

    #async fetch function run with another thread
    def fetch_from_thread(self, current_index):
        curr_index_position = self.data_order.index(current_index)
        for i in range(50):
            if curr_index_position + i >= self.data_order.__len__():
                break
            index_to_fetch = self.data_order[curr_index_position + i]
            response = requests.get(self.hostname + self.filepaths[index_to_fetch][39:]).content
            self.thread_queue.put(response)

