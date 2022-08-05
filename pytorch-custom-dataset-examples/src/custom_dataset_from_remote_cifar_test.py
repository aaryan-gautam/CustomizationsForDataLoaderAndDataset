import glob
from os.path import isdir

import pandas as pd
import numpy as np
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


class CustomDatasetFromFetchCifarTest(Dataset):

    def __init__(self, csv_path):
        host = "mc22.cs.purdue.edu"
        port = 22
        username = "gautam10"
        password = "randompassword3"
        directories = ['archive/CIFAR100/TEST']
        remote_images_path = '/homes/gautam10/'

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=username, password=password)
        sftp = ssh.open_sftp()
        self.file_remote_paths = []
        file_count = 0

        for directory in directories:
            command = "ls " + remote_images_path + directory
            stdin, stdout, stderr = ssh.exec_command(command)
            lines = stdout.readlines()
            # print(lines)
            # print("end")
            for line in lines:
                self.file_remote_paths.append(line)
                file_count += 1

        sftp.close()
        ssh.close()

        self.data_len = file_count

    def __getitem__(self, index):
        host = "mc22.cs.purdue.edu"
        port = 22
        username = "gautam10"
        password = "randompassword3"
        directories = ['archive/CIFAR100/TEST']
        remote_images_path = '/homes/gautam10/'
        local_path = '../fetched/'

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=username, password=password)
        sftp = ssh.open_sftp()

        file_to_fetch = self.file_remote_paths[index]
        print(file_to_fetch)

        file_remote = remote_images_path + 'archive/CIFAR100/TEST' + "/" + file_to_fetch[:len(file_to_fetch) - 1]
        print(file_remote)

        file_local = local_path + 'archive/CIFAR100/TEST' + "/" + file_to_fetch[:len(file_to_fetch) - 1]

        # sftp.get(file_remote, file_local)
        inbound_files = sftp.listdir(file_remote)
        for file in inbound_files:
            filepath = file_remote + "/" + file
            localpath = file_local
            print(filepath)
            sftp.get(filepath, localpath)
        # print(file_remote + '>>>' + file_local)
        print(index)
        sftp.close()
        ssh.close()

        # Get image name from the pandas df
        single_image_path = file_local
        return ImageFolder(single_image_path)


    def __len__(self):
        return self.data_len