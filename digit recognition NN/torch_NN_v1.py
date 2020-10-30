"""
Building digit recognition neural network using pytorch
We try to make the code as light as possible
"""

import struct
import array
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
import torch

# ===================
# Dataset preparation
# ===================


DATA_TYPES = {0x08: 'B',  # unsigned byte
              0x09: 'b',  # signed byte
              0x0b: 'h',  # short (2 bytes)
              0x0c: 'i',  # int (4 bytes)
              0x0d: 'f',  # float (4 bytes)
              0x0e: 'd'}  # double (8 bytes)

FILE_NAMES = ['train-images.idx3-ubyte',
              'train-labels.idx1-ubyte',
              't10k-images.idx3-ubyte',
              't10k-labels.idx1-ubyte']

def mnist_data(i):
    filename = 'database//' + FILE_NAMES[i]
    fd = open(filename, 'rb')
    header = fd.read(4)
    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
    data_type = DATA_TYPES[data_type]
    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions, fd.read(4 * num_dimensions))
    data = array.array(data_type, fd.read())
    data.byteswap()
    return np.array(data).reshape(dimension_sizes)

TRAIN_IMAGES = mnist_data(0)
TRAIN_LABELS = mnist_data(1)
TEST_IMAGES = mnist_data(2)
TEST_LABELS = mnist_data(3)

LEN_TRAIN = len(TRAIN_LABELS)
LEN_TEST = len(TEST_LABELS)

# ===================
# Network
# ===================

model = torch.nn.Sequential(
    torch.nn.Linear(784, 16),
    torch.nn.Linear(16, 10)
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

nb_epochs = 2
batch_size = 100
nb_batches = LEN_TRAIN // batch_size

def train(model=model):
    liste_loss = []
    
    for epoch in range(nb_epochs):
        random_shuffle = torch.randperm(LEN_TRAIN)
        x_train = torch.tensor(TRAIN_IMAGES[random_shuffle]/255, dtype=torch.float32)
        x_train = torch.reshape(x_train, (nb_batches, batch_size, 784))
        x_labels = torch.tensor(TRAIN_LABELS[random_shuffle], dtype=torch.int64)
        x_labels = torch.reshape(x_labels, (nb_batches, batch_size))
        
        for i in range(nb_batches):
            optimizer.zero_grad()
            x_input = x_train[i]
            y_target = x_labels[i]
            y_pred = model(x_input)
            y_loss = criterion(y_pred, y_target)
            y_loss.backward()
            optimizer.step()
            
        print(epoch, y_loss)
    return
    

def test(model=model):
    with torch.no_grad():
        precision = 0
        torch_images = torch.tensor(TEST_IMAGES/255, dtype=torch.float32)
        torch_images = torch.reshape(torch_images, (LEN_TEST, 784))
        predictions = torch.argmax(model(torch_images), dim=1)
        nb_correct = (predictions == torch.tensor(TEST_LABELS)).sum()
    return nb_correct.item() / LEN_TEST