import torch
import numpy as np

def per_image_whiten(image):
    mean = image.mean(); stddev = image.std()
    adjusted_stddev = max(stddev, 1.0/np.sqrt(image.numel()))
    return (image - mean) / adjusted_stddev

def random_labels(data_loader, num_labels=10):
    dataset = data_loader.dataset
    if dataset.train:
        labels = dataset.train_labels
    else:
        labels = dataset.test_labels

    for i in range(len(labels)):
        labels[i] = np.random.choice(num_labels)
