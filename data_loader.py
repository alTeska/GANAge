"""
Created for ConvAge project
"""
import torch
from data_utils import FaceWiki

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    old = FaceWiki(image_paths_file='LAG/train/younglist.txt')
    young = FaceWiki(image_paths_file='LAG/train/oldlist.txt')

    svhn_loader = torch.utils.data.DataLoader(old,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(young,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return svhn_loader, mnist_loader
