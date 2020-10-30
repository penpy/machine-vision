# coding: utf-8

"""
Convolution neural network trained to recognize handwritten digits from the MNIST dataset
The network architecture is inspired from Yann LeCun's LeNet5
Ref: Y. Lecun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition", 1998
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
import torchvision


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # In case GPU is available
print('device: ' + str(device))

n_epochs = 2
train_batch_size = 100
n_iter = 60000 // train_batch_size

path = ''
train_data = torchvision.datasets.MNIST(root=path, train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root=path, train=False, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10000, shuffle=False)


class ConvNeuralNet(nn.Module):
    
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 =nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        
        self.lin_1 = nn.Linear(120, 84)
        self.lin_2 = nn.Linear(84, 10)
        
        # Optional dropout layers (set p = 0. if not wanted)
        self.drop_input = nn.Dropout2d(p=0.2)
        self.drop_conv1 = nn.Dropout2d(p=0.1)
        self.drop_conv3 = nn.Dropout(p=0.2)
        self.drop_lin1 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.drop_input(x)
        
        x = torch.relu(self.conv1(x))
        x = self.drop_conv1(x)
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 120)
        x = self.drop_conv3(x)
        
        x = torch.relu(self.lin_1(x))
        x = self.drop_lin1(x)
        x = self.lin_2(x)
        return x
    

model_conv = ConvNeuralNet()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_conv.parameters(), lr=0.001)

def train(model=model_conv, criterion=criterion, optimizer=optimizer, numEpochs=n_epochs):
    for epoch in range(numEpochs):
        for i, batch in enumerate(train_loader):
            images, labels = batch
            y_pred = model(images).to(device)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100==0:
                txt = 'epoch: ' + str(epoch + 1) + '/' + str(numEpochs)
                txt += ' batch: ' + str(i) + '/' + str(n_iter)
                txt += ' loss = ' + str(loss.item())
                print(txt)
    print('# done')

    
def accuracy():
    model_conv.eval()
    with torch.no_grad():
        for test_batch in test_loader:
            images, labels = test_batch
            y_pred = model_conv(images).to(device)
            max_inidices = torch.max(y_pred, dim=1)[1]
            n_correct = (labels == max_inidices).sum()
    return n_correct.item() / 10000

                
def test():
    i = np.random.randint(10000)
    image = test_data[i][0]
    label = test_data[i][1]
    plt.imshow(image[0], cmap='Greys')
    plt.title(str(i) + ': label = ' + str(label))
    with torch.no_grad():
        img = torch.reshape(image, (1,1,28,28))
        y_pred = model(img)
        max_index = torch.argmax(y_pred).item()
        print('prediction:', max_index)


def main():
    train()
    return accuracy()