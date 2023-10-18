from os import listdir
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from Distribution import get_leaves_imgs
import numpy as np
from torchvision.transforms import v2

class Leaves_dataset(Dataset):
    
    def __init__(self, fruit_directory_path):
        self.fruit_directory_path = fruit_directory_path
        self.sub_directories = listdir(fruit_directory_path)
        self.images_per_leaf_type = [listdir(fruit_directory_path + '/' + type_dir) for type_dir in self.sub_directories]
        self.len = sum(len(images_of_type) for images_of_type in self.images_per_leaf_type)
        self.transform = v2.Resize(size=128)


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        i = index
        for sub_dir_index, leaf_type_images in enumerate(self.images_per_leaf_type):
            # print(sub_directory)
            if index < len(leaf_type_images):
                path = self.fruit_directory_path + '/' + self.sub_directories[sub_dir_index] + '/' + leaf_type_images[index]
                img = self.transform(read_image(path).float())
                # print(path)
                return img, sub_dir_index
            index -= len(leaf_type_images)
        raise IndexError(f"Index {i} out of range, len = {self.len}, subdir len = {len(leaf_type_images)}")
