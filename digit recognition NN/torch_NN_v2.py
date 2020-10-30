import struct
import array
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
import torch

# ===================
# Dataset
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


class Model(torch.nn.Module):
    def __init__(self, p=0, epochs=2):
        super(Model, self).__init__()
        self.d1 = torch.nn.Dropout(p)
        self.l1 = torch.nn.Linear(784,16)
        self.d2 = torch.nn.Dropout(p)
        self.l2 = torch.nn.Linear(16,10)
        
        self.total_epochs = 0
        self.total_iter = 0
        self.loss_evol = []
        self.precision_evol = []
        self.list = []
        
        self.epochs_per_train=epochs


    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.l1(x1)
        x3 = self.d2(x2)
        return self.l2(x3)
    
    def train_it(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.05)

        nb_epochs = self.epochs_per_train
        batch_size = 100
        nb_batches = LEN_TRAIN // batch_size
        
        self.total_epochs += nb_epochs
        self.total_iter += nb_epochs * nb_batches

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

                # forward + backward + optimize
                y_pred = self(x_input)
                y_loss = criterion(y_pred, y_target)
                y_loss.backward()
                optimizer.step()
#                 self.loss_evol.append(y_loss.item())
#                 if (epoch==0 and i<=15) or (epoch == 0 and 15<i<71 and i%5 ==0) or (i%100 == 0):
#                     self.list.append(epoch*nb_batches + i)
#                     self.precision_evol.append(self.precision())
                if (i==nb_batches-1):
                    print(epoch, y_loss)
    
    def precision(self):
        with torch.no_grad():
            precision = 0
            torch_images = torch.tensor(TEST_IMAGES/255, dtype=torch.float32)
            torch_images = torch.reshape(torch_images, (LEN_TEST, 784))
            predictions = torch.argmax(self(torch_images), dim=1)
            nb_correct = (predictions == torch.tensor(TEST_LABELS)).sum()
        return nb_correct.item() / LEN_TEST
    
    def test(self):
        i = np.random.randint(LEN_TEST)
        plt.subplot(121)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(TEST_IMAGES[i], 'Greys')
        with torch.no_grad():
            torch_image = torch.tensor(TEST_IMAGES[i]/255, dtype=torch.float32)
            torch_image = torch.reshape(torch_image, (1, 784))
            y_pred = torch.softmax(self(torch_image), dim=1)
            prediction = y_pred.argmax().item()
            
        if TEST_LABELS[i] == prediction:
            plt.suptitle('CORRECT', color='g', fontsize=16, fontweight='bold')
        else:
            plt.suptitle('INCORRECT', color='r', fontsize=14, fontweight='bold')
        plt.title('Label = ' + str(TEST_LABELS[i]))
        plt.subplot(122)
        plt.bar(range(10), y_pred[0])
        plt.ylim(0, 1)
        plt.xticks(range(10))
        plt.grid(axis='y')
        plt.xlabel('Output values')
        plt.title('Guessed digit = ' + str(prediction))