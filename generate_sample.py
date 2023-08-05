import argparse
import os 
import numpy as np
from model import Generator
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid




parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default= 0, help='Seed for generate images')
parser.add_argument('--out_dir', type=str, default='./results', help='Directory for the output images')
parser.add_argument('--num_imgs', type=int, default=1, help='Number of images to generate')
parser.add_argument('--weight', type=str, help='Generator weight')
parser.add_argument('--out_res', type=int, default=128, help='The resolution of final output image')
parser.add_argument('--cuda', action='store_true', help='Using GPU to train')

opt = parser.parse_args()

if not os.path.exists(opt.out_dir):
	os.makedirs(opt.out_dir)
device = torch.device('cuda:0' if (torch.cuda.is_available() and opt.cuda)  else 'cpu')
latent_size = 512


G_net = Generator(latent_size, opt.out_res).to(device)


G_net.load_state_dict(torch.load(opt.weight))
G_net.depth = int(np.log2(opt.out_res)) - 1
noise = torch.randn(opt.num_imgs, latent_size, 1, 1, device=device)

G_net.eval()
out_imgs = G_net(noise)
save_image(out_imgs, 'out_grid.png', normalize=True)













