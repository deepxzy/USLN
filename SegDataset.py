import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


def read_file_list(type='train'):


    path_list_images_train = os.listdir(r"datasets/images_train")
    path_list_labels_train = os.listdir(r"datasets/labels_train")
    path_list_images_val = os.listdir(r"datasets/images_val")
    path_list_labels_val = os.listdir(r"datasets/labels_val")
    path_list_images_test = os.listdir(r"datasets/images_test")

    images_train = [os.path.join(r"datasets/images_train", i) for i in path_list_images_train]
    labels_train = [os.path.join(r"datasets/labels_train", i) for i in path_list_labels_train]
    images_val = [os.path.join(r"datasets/images_val", i) for i in path_list_images_val]
    labels_val = [os.path.join(r"datasets/labels_val", i) for i in path_list_labels_val]
    images_test = [os.path.join(r"datasets/images_test", i) for i in path_list_images_test]


    if type == 'train':
        return images_train, labels_train  # 两者路径的列表
    elif type == 'val':
        return images_val, labels_val
    elif type == 'test':
        return images_test, path_list_images_test


def preprocess_input(image):
    image /= 255.0
    return image



class SegDataset(torch.utils.data.Dataset):
    def __init__(self, type):



        images, labels = read_file_list(type=type)
        self.images = images
        self.labels = labels
        print('Read ' + str(len(self.images)) + ' valid examples')


    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a



    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image).convert('RGB')
        label = Image.open(label).convert('RGB')

        image = np.transpose(preprocess_input(np.array(image, np.float64)), [2, 0, 1])
        image = torch.from_numpy(image).type(torch.FloatTensor)
        label = np.transpose(preprocess_input(np.array(label, np.float64)), [2, 0, 1])
        label = torch.from_numpy(label).type(torch.FloatTensor)

        return image, label  # float32 tensor, uint8 tensor

    def __len__(self):
        return len(self.images)




