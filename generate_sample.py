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

resume = 40
G_net = Generator(latent_size, opt.out_res).to(device)
check_point = torch.load('/media/ziwei/Dell Portable Hard Drive/PGGAN/check_points/check_point_epoch_%i.pth' % resume)
G_net.load_state_dict(check_point['G_net'])
G_net.depth = check_point['depth']
G_net.alpha = check_point['alpha']

# G_net.load_state_dict(torch.load(opt.weight))

# noise = torch.randn(opt.num_imgs, latent_size, 1, 1, device=device)
noise = check_point['fixed_noise']
G_net.eval()
out_imgs = G_net(noise)
print(G_net.alpha)
save_image(out_imgs[0], 'out_grid.png', normalize=True)













