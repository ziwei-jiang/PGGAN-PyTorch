import argparse
import os 
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import torch.optim as optim

from model import Generator, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./', help='directory contrains the data and outputs')
parser.add_argument('--epochs', type=int, default=40, help='training epoch number')
parser.add_argument('--out_res', type=int, default=128, help='The resolution of final output image')
parser.add_argument('--resume', type=int, default=0, help='continues from epoch number')
parser.add_argument('--cuda', action='store_true', help='Using GPU to train')


opt = parser.parse_args()

root = opt.root
data_dir = root + 'dataset/'
check_point_dir = root + 'check_points/'
output_dir = root + 'output/'
weight_dir = root+ 'weight/'
if not os.path.exists(check_point_dir):
	os.makedirs(check_point_dir)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
if not os.path.exists(weight_dir):
	os.makedirs(weight_dir)

## The schedule contains [num of epoches for starting each size][batch size for each size][num of epoches]
schedule = [[5, 15, 25 ,35, 40],[16, 16, 16, 8, 4],[5, 5, 5, 1, 1]]
batch_size = schedule[1][0]
growing = schedule[2][0]
epochs = opt.epochs
latent_size = 512
out_res = opt.out_res
lr = 1e-4
lambd = 10

device = torch.device('cuda:0' if (torch.cuda.is_available() and opt.cuda)  else 'cpu')

transform = transforms.Compose([
			transforms.Resize(out_res),
			transforms.CenterCrop(out_res),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])


D_net = Discriminator(latent_size, out_res).to(device)
G_net = Generator(latent_size, out_res).to(device)

fixed_noise = torch.randn(16, latent_size, 1, 1, device=device)
D_optimizer = optim.Adam(D_net.parameters(), lr=lr, betas=(0, 0.99))
G_optimizer = optim.Adam(G_net.parameters(), lr=lr, betas=(0, 0.99))


D_running_loss = 0.0
G_running_loss = 0.0
iter_num = 0

D_epoch_losses = []
G_epoch_losses = []

if torch.cuda.device_count() > 1:
	print('Using ', torch.cuda.device_count(), 'GPUs')
	D_net = nn.DataParallel(D_net)
	G_net = nn.DataParallel(G_net)

if opt.resume != 0:
	check_point = torch.load(check_point_dir+'check_point_epoch_%i.pth' % opt.resume)
	fixed_noise = check_point['fixed_noise']
	G_net.load_state_dict(check_point['G_net'])
	D_net.load_state_dict(check_point['D_net'])
	G_optimizer.load_state_dict(check_point['G_optimizer'])
	D_optimizer.load_state_dict(check_point['D_optimizer'])
	G_epoch_losses = check_point['G_epoch_losses']
	D_epoch_losses = check_point['D_epoch_losses']
	G_net.depth = check_point['depth']
	D_net.depth = check_point['depth']
	G_net.alpha = check_point['alpha']
	D_net.alpha = check_point['alpha']


try:
	c = next(x[0] for x in enumerate(schedule[0]) if x[1]>opt.resume)-1
	batch_size = schedule[1][c]
	growing = schedule[2][c]
	dataset = datasets.ImageFolder(data_dir, transform=transform)
	# dataset = datasets.CelebA(data_dir, split='all', transform=transform)
	data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

	tot_iter_num = (len(dataset)/batch_size)
	G_net.fade_iters = (1-G_net.alpha)/(schedule[0][c+1]-opt.resume)/(2*tot_iter_num)
	D_net.fade_iters = (1-D_net.alpha)/(schedule[0][c+1]-opt.resume)/(2*tot_iter_num)


except:
	print('Fully Grown\n')
	c = -1
	batch_size = schedule[1][c]
	growing = schedule[2][c]

	dataset = datasets.CelebA(data_dir, split='all', transform=transform)
	data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

	tot_iter_num = (len(dataset)/batch_size)
	print(schedule[0][c], opt.resume)

	if G_net.alpha < 1:
		G_net.fade_iters = (1-G_net.alpha)/(opt.epochs-opt.resume)/(2*tot_iter_num)
		D_net.fade_iters = (1-D_net.alpha)/(opt.epochs-opt.resume)/(2*tot_iter_num)


size = 2**(G_net.depth+1)
print("Output Resolution: %d x %d" % (size, size))

for epoch in range(1+opt.resume, opt.epochs+1):
	G_net.train()
	D_epoch_loss = 0.0
	G_epoch_loss = 0.0
	if epoch-1 in schedule[0]:

		if (2 **(G_net.depth +1) < out_res):
			c = schedule[0].index(epoch-1)
			batch_size = schedule[1][c]
			growing = schedule[2][0]
			data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)
			tot_iter_num = tot_iter_num = (len(dataset)/batch_size)
			G_net.growing_net(growing*tot_iter_num)
			D_net.growing_net(growing*tot_iter_num)
			size = 2**(G_net.depth+1)
			print("Output Resolution: %d x %d" % (size, size))

	
	print("epoch: %i/%i" % (int(epoch), int(epochs)))
	databar = tqdm(data_loader)

	for i, samples in enumerate(databar):
		##  update D
		if size != out_res:
			samples = F.interpolate(samples[0], size=size).to(device)
		else:
			samples = samples[0].to(device)
		D_net.zero_grad()
		noise = torch.randn(samples.size(0), latent_size, 1, 1, device=device)
		fake = G_net(noise)
		fake_out = D_net(fake.detach())
		real_out = D_net(samples)

		## Gradient Penalty

		eps = torch.rand(samples.size(0), 1, 1, 1, device=device)
		eps = eps.expand_as(samples)
		x_hat = eps * samples + (1 - eps) * fake.detach()
		x_hat.requires_grad = True
		px_hat = D_net(x_hat)
		grad = torch.autograd.grad(
									outputs = px_hat.sum(),
									inputs = x_hat, 
									create_graph=True
									)[0]
		grad_norm = grad.view(samples.size(0), -1).norm(2, dim=1)
		gradient_penalty = lambd * ((grad_norm  - 1)**2).mean()

		###########

		D_loss = fake_out.mean() - real_out.mean() + gradient_penalty

		D_loss.backward()
		D_optimizer.step()

		##	update G

		G_net.zero_grad()
		fake_out = D_net(fake)

		G_loss = - fake_out.mean()

		G_loss.backward()
		G_optimizer.step()

		##############

		D_running_loss += D_loss.item()
		G_running_loss += G_loss.item()

		iter_num += 1


		if i % 500== 0:
			D_running_loss /= iter_num
			G_running_loss /= iter_num
			print('iteration : %d, gp: %.2f' % (i, gradient_penalty))
			databar.set_description('D_loss: %.3f   G_loss: %.3f' % (D_running_loss ,G_running_loss))
			iter_num = 0
			D_running_loss = 0.0
			G_running_loss = 0.0

		
	D_epoch_losses.append(D_epoch_loss/tot_iter_num)
	G_epoch_losses.append(G_epoch_loss/tot_iter_num)


	check_point = {'G_net' : G_net.state_dict(), 
				   'G_optimizer' : G_optimizer.state_dict(),
				   'D_net' : D_net.state_dict(),
				   'D_optimizer' : D_optimizer.state_dict(),
				   'D_epoch_losses' : D_epoch_losses,
				   'G_epoch_losses' : G_epoch_losses,
				   'fixed_noise': fixed_noise,
				   'depth': G_net.depth,
				   'alpha':G_net.alpha
				   }
	with torch.no_grad():
		G_net.eval()
		torch.save(check_point, check_point_dir + 'check_point_epoch_%d.pth' % (epoch))
		torch.save(G_net.state_dict(), weight_dir + 'G_weight_epoch_%d.pth' %(epoch))
		out_imgs = G_net(fixed_noise)
		out_grid = make_grid(out_imgs, normalize=True, nrow=4, scale_each=True, padding=int(0.5*(2**G_net.depth))).permute(1,2,0)
		plt.imshow(out_grid.cpu())
		plt.savefig(output_dir + 'size_%i_epoch_%d' %(size ,epoch))




