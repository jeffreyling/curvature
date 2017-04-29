import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from models import AlexNet, MLP, Inception
from transforms import per_image_whiten
from util import *

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Plot curvature around model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='mlp',
                    help='model architecture')
# parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    # help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model-path', required=True, type=str, metavar='PATH',
                    help='path to model')
parser.add_argument('--num-plots', default=5, type=int,
                    help='number of data points to plot')
parser.add_argument('--fig-names', required=True, type=str, help='fig names')

parser.add_argument('--eps', type=float, default=.25,
                    help='eps of gradient, chosen uniformly')
parser.add_argument('--step', type=float, default=.1,
                    help='step size for figures')
parser.add_argument('--fig-size', type=float, default=20,
                    help='max perturbation in positive and negative directions')
parser.add_argument('--seed', type=int, default=3435, help='random seed')

def load_model(model_path, model):
    assert os.path.isfile(model_path), 'no file found at {}'.format(model_path)
    print("=> loading model '{}'".format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_path, checkpoint['epoch']))

models = {'alexnet': AlexNet, 'mlp': MLP, 'inception': Inception}

def main():
    global args
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models[args.arch]()
    model = model.cuda()
    load_model(args.model_path, model)

    # load data
    normalize = transforms.Lambda(per_image_whiten)
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.CenterCrop(28),
                            transforms.ToTensor(),
                            normalize,
                        ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data/', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.CenterCrop(28),
                            transforms.ToTensor(),
                            normalize,
                        ])),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True)

    model.eval()
    dir1, dir2 = get_directions(torch.Size((1, 3, 28, 28)), args.eps)

    print('plotting for train...')
    for i, (input, target) in enumerate(train_loader):
        if i >= args.num_plots: break
        filename = '{}_train_eps{:.2f}_size{}_step{}_{}.png'.format(
                args.fig_names,
                args.eps,
                int(args.fig_size),
                args.step,
                i)
        plot_curvature(input, model, filename, dir1, dir2, size=args.fig_size, step=args.step)

    print('plotting for val...')
    for i, (input, target) in enumerate(val_loader):
        if i >= args.num_plots: break
        filename = '{}_val_eps{:.2f}_size{}_step{}_{}.png'.format(
                args.fig_names,
                args.eps,
                int(args.fig_size),
                args.step,
                i)
        plot_curvature(input, model, filename, dir1, dir2, size=args.fig_size, step=args.step)

def plot_curvature(input, model, filename, dir1, dir2, size=20, step=0.1):
    xx, yy = np.meshgrid(np.arange(-size, size, step), np.arange(-size, size, step))
    xx_flat, yy_flat = xx.flatten(), yy.flatten()
    all_input = input.repeat(xx_flat.shape[0], 1, 1, 1)
    for i in range(len(xx_flat)):
        x,y = xx_flat[i], yy_flat[i]
        all_input[i].add_(dir1 * x).add_(dir2 * y)

    all_preds = torch.LongTensor(all_input.size(0), 1)
    for i in range(0, all_input.size(0), args.batch_size):
        input_var = Variable(all_input[i:i + args.batch_size].cuda(), volatile=True)
        labels = model(input_var)
        _, preds = torch.max(labels.data, 1)
        all_preds[i:i + args.batch_size].copy_(preds)

    all_preds = all_preds.numpy().reshape((xx.shape[0], xx.shape[1]))

    plt.figure()
    plt.axis('equal')
    plt.pcolormesh(xx, yy, all_preds) # colors?
    plt.savefig(filename)

def get_directions(size, eps):
    # Return random noise for now
    dir1 = torch.Tensor(size).uniform_(-eps, eps)
    dir2 = torch.Tensor(size).uniform_(-eps, eps)
    return dir1, dir2


if __name__ == '__main__':
    main()
