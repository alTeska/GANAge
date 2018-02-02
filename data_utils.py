"""
Created on Sun Jan 21 21:20:32 2018
@author: Monkey-PC
Modified for ConvAge project
"""

"""Data utility functions."""
import os
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

class FaceWiki(data.Dataset):

    def __init__(self, image_paths_file):
        self.root_dir_name = os.path.dirname(image_paths_file)
        with open(image_paths_file) as f:
            self.image_folders = f.read().splitlines()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_folders)

    def get_item_from_index(self, index):
        to_tensor = transforms.ToTensor()
        img_folder = self.image_folders[index]

        img = Image.open(os.path.join(self.root_dir_name, img_folder)).convert('RGB')
        grayscale = transforms.Grayscale()
        resize = transforms.Resize((200,200))

        img = resize(img)
        img = to_tensor(img)
        return img, img_folder
