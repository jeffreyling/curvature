import argparse
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import AlexNet, MLP, Inception
from util import *

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from sklearn.neighbors import NearestNeighbors

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

parser.add_argument('--seed', type=int, default=3435, help='random seed')

parser.add_argument('--neighbors', type=int, default=5, help='number of neighbors')

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
    train_loader, val_loader = load_datasets()

    # annoying
    X_transformed = np.array([input.numpy() for input, _ in train_loader])
    X_transformed = X_transformed.reshape((X_transformed.shape[0], 3*28*28))
    print("constructing nearest neighbors...")
    knn = NearestNeighbors(args.neighbors).fit(X_transformed)
    print("done constructing")

    for i, (input, label) in enumerate(train_loader):
        if i >= args.num_plots: break
        idxs = knn.kneighbors(input.numpy().flatten(), n_neighbors=args.neighbors + 1, return_distance=False) # +1 since it will return itself
        idxs = idxs.flatten()[1:]
        print("==> sample {} nearest neighbors:".format(i))
        print(idxs)
        closest = [X_transformed[idx].reshape((3,28,28)) for idx in idxs]
        dirs = [get_dir(input.numpy(), x) for x in closest]

        # directional derivatives
        input_var = Variable(input.cuda(), requires_grad=True)
        model.zero_grad() # irrelevant
        output = model(input_var)
        partials = []
        for j in range(10): # num classes
            dout = torch.zeros(output.size())
            dout[0,j] = 1
            output.backward(dout.cuda(), retain_variables=True)
            partials.append(input_var.grad.data.clone())

        for j,d in zip(idxs, dirs):
            deriv = [torch.sum(p * d) for p in partials]
            print("==> directional derivative for {} to {}: ".format(i, j) + str(["{:2f}".format(a) for a in deriv]).replace("'", ""))

def get_dir(input, x):
    vec = x - input
    vec = vec / np.linalg.norm(vec)
    return torch.FloatTensor(vec).cuda()

if __name__ == '__main__':
    main()
