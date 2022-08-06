import glob
import io
import time
from os.path import isdir

import pandas as pd
import numpy as np
import pdfplumber as pdfplumber
from paramiko import sftp
from paramiko.client import WarningPolicy
from sklearn import preprocessing

from PIL import Image
import base64
import os
import paramiko
import csv
from stat import S_ISDIR as isdir
import torch


from torchvision import transforms
from torch.utils.data.dataset import Dataset, TensorDataset  # For custom datasets
from torchvision.datasets import ImageFolder


class CustomDatasetFromFetchCifar(Dataset):
    global ssha
    global sftpa

    def __init__(self, csv_path):
        training_time = time.time()
        host = "mc21.cs.purdue.edu"
        port = 22
        username = "gautam10"
        password = "randompassword3"
        directories = ['archive/CIFAR100/TRAIN']
        remote_images_path = '/homes/gautam10/'
        local_path = '../fetched/CIFAR100/TRAIN'

        self.file_record = []

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=username, password=password)
        sftp = ssh.open_sftp()
        self.file_remote_paths = []
        file_count = 0
        self.setssh(ssh)
        self.track = 0

        for directory in directories:
            command = "ls " + remote_images_path + directory
            stdin, stdout, stderr = ssh.exec_command(command)
            lines = stdout.readlines()
            for line in lines:
                self.file_remote_paths.append(line)
                command = "ls " + remote_images_path + directory + "/" + line
                stdin, stdout, stderr = ssh.exec_command(command)
                files = stdout.readlines()
                for file in files:
                    file_count += 1
                    self.file_record.append(remote_images_path + directory + "/" + line[:len(line) - 1] + "/" +
                                            file[:len(file) - 1])

        self.data_len = file_count
        self.dir_count = 0
        self.file_number = 0

    def __getitem__(self, index):
        host = "mc21.cs.purdue.edu"
        port = 22
        username = "gautam10"
        password = "randompassword3"
        directories = ['archive/CIFAR100/TRAIN']
        all_dir = ['apple', 'aquarium_fish','baby'
,'bear'
,'beaver'
,'bed'
,'bee'
,'beetle'
,'bicycle'
,'bottle'
,'bowl'
,'boy'
,'bridge'
,'bus'
,'butterfly'
,'camel'
,'can'
,'castle'
,'caterpillar'
,'cattle'
,'chair'
,'chimpanzee'
,'clock'
,'cloud'
,'cockroach'
,'couch'
,'crab'
,'crocodile'
,'cup'
,'dinosaur'
,'dolphin'
,'elephant'
,'flatfish'
,'forest'
,'fox'
,'girl'
,'hamster'
,'house'
,'kangaroo'
,'keyboard'
,'lamp'
,'lawn_mower'
,'leopard'
,'lion'
,'lizard'
,'lobster'
,'man'
,'maple_tree'
,'motorcycle'
,'mountain'
,'mouse'
,'mushroom'
,'oak_tree'
,'orange'
,'orchid'
,'otter'
,'palm_tree'
,'pear'
,'pickup_truck'
,'pine_tree'
,'plain'
,'plate'
,'poppy'
,'porcupine'
,'possum'
,'rabbit'
,'raccoon'
,'ray'
,'road'
,'rocket'
,'rose'
,'sea'
,'seal'
,'shark'
,'shrew'
,'skunk'
,'skyscraper'
,'snail'
,'snake'
,'spider'
,'squirrel'
,'streetcar'
,'sunflower'
,'sweet_pepper'
,'table'
,'tank'
,'telephone'
,'television'
,'tiger'
,'tractor'
,'train'
,'trout'
,'tulip'
,'turtle'
,'wardrobe'
,'whale'
,'willow_tree'
,'wolf'
,'woman'
,'worm']
        remote_images_path = '/homes/gautam10/'
        local_path = '../fetched/'

        ssh = self.getssh()
        sftp = ssh.open_sftp()

        image_to_fetch = self.file_record[index]

        file_to_fetch = image_to_fetch.rsplit('/', 2)[1]

        file_remote = remote_images_path + 'archive/CIFAR100/TRAIN' + "/" + file_to_fetch

        file_local = local_path + 'CIFAR100/TRAIN' + "/" + file_to_fetch
        # inbound_files = sftp.listdir(file_remote)
        filepath = file_remote + "/" + image_to_fetch.rsplit('/', 2)[2]
        # download file
        file_start_time = time.time()
        sftp.get(filepath, "../fetched/CIFAR100/TRAIN/" + image_to_fetch.rsplit('/', 2)[1] + "/" + image_to_fetch.rsplit('/', 2)[2])
        print('file recieved in {:.2f}s', format(time.time() - file_start_time))

        print(filepath)
        # test fetching on commands
        file_content = self.fetch_file_as_bytesIO(sftp, filepath)
        # command = "echo " + "$(python3 -c \"from PIL import Image;print(Image.open(" + "\'" + filepath + "\'" + "))\")"
        # command = "cat " + filepath
        # stdin, stdout, stderr = ssh.exec_command(command)
        # echo $(python3 - c "from PIL import Image;print(Image.open('/homes/gautam10/archive/CIFAR100/TRAIN/bowl/bowl_s_
        # 000718.png'))")
        # lines = stdout.readlines()
        # print('file image loaded in {:.2f}s', format(time.time() - file_opening_time))
        # fetch from bytes
        # im = Image.frombuffer("I;16", (50, 100), file_content.getvalue(), "raw", "I;12")

        ncfile = sftp.open(filepath)
        ncfile.prefetch()
        b_ncfile = ncfile.read()
        print(type(b_ncfile))
        self.track = self.track + 1

        single_image_path = file_local + "/" + image_to_fetch.rsplit('/', 2)[2]
        im_as_im = Image.open(single_image_path)

        # Convert to numpy
        im_as_np = np.asarray(im_as_im) / 255

        # Transform image to tensor, change data type
        im_as_ten = torch.from_numpy(im_as_np).float()

        # return ImageFolder(single_image_path)
        # training_start_time = time.time()
        im_as_ten = torch.reshape(im_as_ten, [3, 32, 32])

        label = all_dir.index(file_to_fetch)

        if self.track == self.data_len:
            sftp.close()
            ssh.close()
        return im_as_ten, label


    def __len__(self):
        return self.data_len

    def getssh(self):
        return self.ssha

    def setssh(self, ssh):
        self.ssha = ssh

    def fetch_file_as_bytesIO(self, sftpb, path):
        """
        Using the sftp client it retrieves the file on the given path by using pre fetching.
        :param sftp: the sftp client
        :param path: path of the file to retrieve
        :return: bytesIO with the file content
        """
        with sftpb.file(path, mode='rb') as file:
            file_size = file.stat().st_size
            file.prefetch(file_size)
            file.set_pipelined()
            return io.BytesIO(file.read(file_size))
        # flo = BytesIO()
        # sftp.getfo(fullpath, flo)
        # flo.seek(0)
        # pdfplumber.load(flo)

    def get_all_files(self):
        return self.file_record