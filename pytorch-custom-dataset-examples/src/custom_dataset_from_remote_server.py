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

class CustomDatasetFromFetch(Dataset):

    def __init__(self, csv_path):
        host = "mc21.cs.purdue.edu"
        port = 22
        username = "gautam10"
        password = "randompassword3"
        directories = ['mnist_with_class']
        remote_images_path = '/homes/gautam10/data/'
        local_path = '../fetched/'

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
            for line in lines:
                self.file_remote_paths.append(line)
                file_count += 1

        sftp.close()
        ssh.close()

        self.data_len = file_count

    def __getitem__(self, index):
        host = "mc21.cs.purdue.edu"
        port = 22
        username = "gautam10"
        password = "randompassword1"
        directories = ['mnist_with_class']
        remote_images_path = '/homes/gautam10/data/'
        local_path = '../fetched/'

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=username, password=password)
        sftp = ssh.open_sftp()

        file_to_fetch = self.file_remote_paths[index]

        file_remote = remote_images_path + 'mnist_with_class' + "/" + file_to_fetch[:len(file_to_fetch) - 1]

        file_local = local_path + 'mnist_with_class' + "/" + file_to_fetch[:len(file_to_fetch) - 1]

        sftp.get(file_remote, file_local)
        print(file_remote + '>>>' + file_local)
        sftp.close()
        ssh.close()

        # Get image name from the pandas df
        single_image_path = file_local
        # Open image
        im_as_im = Image.open(single_image_path)

        # Do some operations on image
        # Convert to numpy, dim = 28x28
        im_as_np = np.asarray(im_as_im) / 255
        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        im_as_np = np.expand_dims(im_as_np, 0)

        # Transform image to tensor, change data type
        im_as_ten = torch.from_numpy(im_as_np).float()

        # Get label(class) of the image based on the file name
        class_indicator_location = single_image_path.rfind('_c')
        label = int(single_image_path[class_indicator_location + 2:class_indicator_location + 3])
        return (im_as_ten, label)


    def __len__(self):
        return self.data_len