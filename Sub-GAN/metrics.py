from __future__ import print_function
from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
# from trainer import SubGAN_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os
import yaml
import torch_fidelity

from networks import AdaINGen, MsImageDis, StyleGenerator, StyleDiscriminator
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

from torchvision import transforms
from data import ImageFilelist, ImageFolder
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--folder1', type=str, help="folder 1")
parser.add_argument('--folder2', type=str, help="folder 2")

opts = parser.parse_args()

def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, num_workers=4, crop=True):
#     transform_list = [transforms.ToTensor(),
#                       transforms.Normalize((0.5, 0.5, 0.5),
#                                            (0.5, 0.5, 0.5))]
#     transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
#     transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
#     transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
#     transform = transforms.Compose(transform_list)
    transform_list = [transforms.PILToTensor()]
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder,transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=1)
    return loader

# cn = './configs/selfie2anime.yaml'
# config = get_config(cn)
# trainer = SubGAN_Trainer(config)
# trainer = resume(trainer, 'outputs/selfie2anime/checkpoints', '00200000')
# folder = '../datasets/horse2zebra/testB'
# folder2 = 'touts_00'

folder = opts.folder1
folder2 = opts.folder2

data = get_data_loader_folder(folder, 1, False)
data2 = get_data_loader_folder(folder2, 1, False)

metrics = torch_fidelity.calculate_metrics(
            input2=data2.dataset,
            input1=data.dataset,
            isc=True,
            fid=True,
            kid=True,
            kid_subset_size=100,
            ppl_epsilon=1e-2,
            batch_size=1
            # ppl_sample_similarity_resize=64,
        )
        
# log metrics
for k, v in metrics.items():
    print(f'metrics/{k}', v)