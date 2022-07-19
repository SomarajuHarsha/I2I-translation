# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

from networks import ResnetGenerator

# from ugatit_pytorch import Generator
from dataset import ImageFolder

parser = argparse.ArgumentParser(description="PyTorch Generate Realistic Animation Face")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("--dataset", type=str, default="selfie2anime",
                    help="dataset name.  (default:`selfie2anime`)"
                         "Option: [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, "
                         "cezanne2photo, ukiyoe2photo, vangogh2photo, selfie2anime]")
parser.add_argument("--cuda", action="store_true", help="Enables cuda")
parser.add_argument("--model", type=str, help="model path")
parser.add_argument("--a2b", type=int, default=1, help="model path")
parser.add_argument("--outf", default="./results",
                    help="folder to output images. (default: `./results`).")
parser.add_argument("--image-size", type=int, default=256,
                    help="size of the data crop (squared assumed). (default:256)")
parser.add_argument("--manualSeed", type=int,
                    help="Seed for initializing training. (default:none)")

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Dataset
dataset = ImageFolder(root=os.path.join(args.dataroot, args.dataset),
                       transform=transforms.Compose([
                           transforms.Resize(args.image_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                       ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

try:
    os.makedirs(os.path.join(args.outf, str(args.dataset)))
except OSError:
    pass

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
# netG_A2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256, light=True).to(device)
# netG_B2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256, light=True).to(device)
netG = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256, light=True).to(device)

params = torch.load(args.model, map_location=torch.device(device))

# Load state dicts
# netG_A2B.load_state_dict(params['genA2B'])
# netG_B2A.load_state_dict(params['genB2A'])
netG.load_state_dict(params['genA2B'] if args.a2b else args.genB2A)

# Set model mode
# netG_A2B.eval()
# netG_B2A.eval()
netG.eval()

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

for i, data in progress_bar:
    # get batch size data
    # real_image_A = data["A"].to(device)
    # real_image_B = data["B"].to(device)
    # print(data)
    real_image = data[0].to(device)

    # Generate output
    # fake_image_B, _ = netG_A2B(real_image_A)
    # fake_image_A, _ = netG_B2A(real_image_B)
    fake_image, _, _ = netG(real_image)

    # Save image files
    # vutils.save_image(fake_image_A.detach(), f"{args.outf}/{args.dataset}/A/{i + 1:04d}.png", normalize=True)
    # vutils.save_image(fake_image_B.detach(), f"{args.outf}/{args.dataset}/B/{i + 1:04d}.png", normalize=True)
    vutils.save_image(fake_image.detach(), f"{args.outf}/{args.dataset}/{i + 1:04d}.png", normalize=True)

    progress_bar.set_description(f"Generated images {i + 1} of {len(dataloader)}")