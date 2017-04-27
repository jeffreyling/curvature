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
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model-path', required=True, type=str, metavar='PATH',
                    help='path to model')
parser.add_argument('--num-plots', default=5, type=int,
                    help='number of data points to plot')
parser.add_argument('--fig-names', required=True, type=str, help='fig names')

torch.manual_seed(3435)
torch.cuda.manual_seed(3435)

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

    for i, (input, target) in enumerate(train_loader):
        if i >= args.num_plots: break
        filename = '{}_{}.png'.format(args.fig_names, i)
        plot_curvature(input, model, filename)

def plot_curvature(input, model, filename, size=20, step=0.1, num_classes=10):
    # TODO take random directions
    dir1, dir2 = get_directions(input.size())

    xx, yy = np.meshgrid(np.arange(-size, size+1, step), np.arange(-size, size+1, step))
    xx_flat, yy_flat = xx.flatten(), yy.flatten()
    all_input = input.repeat(xx_flat.shape[0], 1, 1, 1)
    for i in range(len(xx_flat)):
        x,y = xx_flat[i], yy_flat[i]
        all_input[i] += dir1 * x + dir2 * y

    all_preds = torch.LongTensor(all_input.size(0), 1)
    for i in range(0, all_input.size(0), args.batch_size):
        input_var = Variable(all_input[i:i + args.batch_size].cuda())
        labels = model(input_var)
        _, preds = torch.max(labels.data, 1)
        all_preds[i:i + args.batch_size].copy_(preds)

    all_preds = all_preds.numpy().reshape((xx.shape[0], xx.shape[1]))

    plt.figure()
    plt.pcolormesh(xx, yy, all_preds) # colors?
    plt.savefig(filename)

def get_directions(size, eps=1.):
    # Return random noise for now
    dir1 = torch.Tensor(size).uniform_(-eps, eps)
    dir2 = torch.Tensor(size).uniform_(-eps, eps)
    return dir1, dir2


if __name__ == '__main__':
    main()
