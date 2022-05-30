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
        password = "randompassword1"
        files = ['mnist_in_csv.csv', 'mnist_labels.csv']
        directories = ['mnist_images', 'mnist_with_class']
        remote_images_path = '/homes/gautam10/data/'
        local_path = '../fetched/'

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(
            paramiko.AutoAddPolicy())
        ssh.connect(hostname=host, port=port, username=username, password=password)
        sftp = ssh.open_sftp()

        for file in files:
            file_remote = remote_images_path + file
            file_local = local_path + file
            sftp.get(file_remote, file_local)
            print(file_remote + '>>>' + file_local)
        for directory in directories:
            command = "ls " + remote_images_path + directory
            stdin, stdout, stderr = ssh.exec_command(command)
            lines = stdout.readlines()
            for line in lines:
                file_remote = remote_images_path + directory + "/" + line[:len(line) - 1]
                file_local = local_path + directory + "/" +line[:len(line) - 1]
                sftp.get(file_remote, file_local)
                print(file_remote + '>>>' + file_local)

        sftp.close()
        ssh.close()

        self.data_len = 5

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Check if there is an operation
        some_operation = self.operation_arr[index]
        # If there is an operation
        if some_operation:
            # Do some operation on image
            # ...
            # ...
            pass
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len