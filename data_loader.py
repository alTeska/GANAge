import torch
from torchvision import datasets
from torchvision import transforms
from data_utils import FaceData


def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
#    transform = transforms.Compose([
#                    transforms.Scale(config.image_size),
#                    transforms.ToTensor(),
#                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#    
#    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform)
#    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform)

    old = FaceData(image_paths_file='LAG/train/train.txt', young=False)
    young = FaceData(image_paths_file='LAG/train/train.txt')

    svhn_loader = torch.utils.data.DataLoader(old,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(young,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return svhn_loader, mnist_loader
