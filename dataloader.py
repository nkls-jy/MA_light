"""
Copyright (c) 2020 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
All rights reserved. This work should only be used for nonprofit purposes.
 
"""

import torch
from dataset import folders_data
from dataset import sar_dataset
from torchvision.transforms import Compose

# paths for home machine
#train_path = "/home/niklas/Documents/test_data"
#valid_path = ""

# paths for uni machine
train_path = "/home/niklas/Documents/train_data"
valid_path = "/home/niklas/Documents/valid_data"

def create_train_realsar_dataloaders(patchsize, batchsize, trainsetiters):
    transform_train = Compose([
        sar_dataset.RandomCropNy(patchsize),
        sar_dataset.Random8OrientationNy(),
        sar_dataset.NumpyToTensor(),
    ])

    trainset = sar_dataset.PlainSarFolder(dirs=train_path, transform=transform_train, cache=True)
    trainset = torch.utils.data.ConcatDataset([trainset]*trainsetiters)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=12)

    return trainloader


def create_valid_realsar_dataloaders(patchsize, batchsize):
    transform_valid = Compose([
        sar_dataset.CenterCropNy(patchsize),
        sar_dataset.NumpyToTensor(),
    ])

    validset = sar_dataset.PlainSarFolder(dirs=valid_path, transform=transform_valid, cache=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batchsize, shuffle=False, num_workers=12)

    return validloader

class PreprocessingInt:
    def __call__(self, batch):
        #print(f'preprocessing input: {batch.shape}')
        
        tl = torch.split(batch, 1, dim=1)
        noisy = tl[0]
        target = tl[1]

        #print(f'noisy shape: {noisy.shape}')
        #print(f'taget shape: {target.shape}')
        #noisy = images[:, :0, :, :]
        #target = images[:, 1:, :, :]
        
        if batch.is_cuda:
            noisy = noisy.cuda()
            target = target.cuda()
        
        # returns 2 tensors with all noisy/target images from batch
        return noisy, target
'''
if __name__ == '__main__':

    data_loader = create_train_realsar_dataloaders(25, 5, 1)
    data_preprocessing = PreprocessingBatch(); flag_log = False

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import numpy as np
    import torchvision.transforms.functional as F

    data_iter = iter(data_loader)
    images = data_iter.next()

    noisy, target = data_preprocessing(images)

    print(noisy.shape)
    print(type(noisy))
'''