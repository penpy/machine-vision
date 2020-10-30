import struct
import array
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import image as img

# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------

def mnist_data(i):
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
    
    filename = FILE_NAMES[i]
    fd = open(filename, 'rb')
    header = fd.read(4)
    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)
    data_type = DATA_TYPES[data_type]
    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions, fd.read(4 * num_dimensions))
    data = array.array(data_type, fd.read())
    data.byteswap()
    
    return np.array(data).reshape(dimension_sizes)


train_images = mnist_data(0)
train_labels = mnist_data(1)
test_images = mnist_data(2)
test_labels = mnist_data(3)


# -------------------------------------------------------------
# Training
# -------------------------------------------------------------
'''
Notations :

w1, w2, w3 : weight matrices 1, 2 and 3
b1, b2, b3 : bias vectors 1, 2 and 3
a : activation function
da : derivative of the activation function

y0 : entry vector (stage 0)
x1, x2, x3 : vectors of stages 1, 2 and 3 before activation function
y1, y2, y3 : vectors of stages 1, 2 and 3 after activation function
t : target
E : error

du_dv : derivative of u with respect to v (for any u,v above)
'''


class NN:
    
    def __init__(self):
        
        self.error = []
        self.inputs = []
        self.outputs = []
        self.iteration = 0
        
        self.len_y0 = 28 * 28
        self.len_y1 = 300
        self.len_y2 = 10
        self.n_epochs = 300
        self.batch_size = 50

        self.batch_shape = (self.batch_size, self.len_y2)
        self.w1_shape = (self.len_y0, self.len_y1)
        self.w2_shape = (self.len_y1, self.len_y2)
        
        self.b1 = np.full(self.len_y1, 0.0)
        self.b2 = np.full(self.len_y2, 0.0)

        self.w1 = np.random.uniform(0, 1, self.w1_shape) / self.len_y1
        self.w2 = np.random.uniform(0, 1, self.w2_shape) / self.len_y2
        
        self.test_len = 10

    def update_ws(self, images, labels):
        
        lr = 0.005
        
        error_batch = np.zeros(self.batch_shape)

        target_batch = np.zeros(self.batch_shape)
    
        sum_db1 = np.zeros(self.len_y1)
        sum_db2 = np.zeros(self.len_y2)
        sum_dw1 = np.zeros(self.w1_shape)
        sum_dw2 = np.zeros(self.w2_shape)
        
        for i in range(self.batch_size):
            
            # Stages
            y0 = images[i].flatten() /255
            x1 = np.matmul(y0, self.w1) + self.b1
            y1 = 1 / (1 + np.exp(-x1))
            x2 = np.matmul(y1, self.w2) + self.b2
            y2_bis = np.exp(x2)
            y2 = y2_bis / y2_bis.sum()
            # Softmax function
            
            # Target
            t = np.zeros(self.len_y2)
            t[labels[i]] = 1
    
            target_batch[i] = t
            
            
            self.inputs.append(images[i])
            self.outputs.append(y2)
            error_batch[i] = -np.log(y2[labels[i]])
            
            
            # Derivatives
            # derivative of Softmax: dS/dx = S(1-S)

            dE_db2 = y2
            dE_db2[labels[i]] -= 1
            dE_dw2 = np.outer(y1, dE_db2)
            P0 = np.matmul(self.w2, dE_db2)
            dE_db1 = P0 * y1 * (1 - y1)
            dE_dw1 = np.outer(y0, dE_db1)

            sum_db1 += dE_db1
            sum_db2 += dE_db2
            sum_dw1 += dE_dw1
            sum_dw2 += dE_dw2
            
        # update ws an bs
        self.b1 -= lr * sum_db1
        self.b2 -= lr * sum_db2

        self.w1 -= lr * sum_dw1
        self.w2 -= lr * sum_dw2
        
        self.error.append(error_batch.mean())
        
    
    
    
    
    def train(self, train_images, train_labels):
        for k in range(self.n_epochs):
            self.iteration += 1
            
            print('Starting round', k+1, '-----------------------')
            
            start = k * self.batch_size
            stop = start + self.batch_size
            images = train_images[start: stop]
            labels = train_labels[start: stop]
            self.update_ws(images, labels)


            # if k == self.n_batches -1 or k==0 or k==100:
            #     print('output =', outputs.mean(0).round(2))
            #     print('target =', targets.mean(0).round(2), '\n')
            # print('Error = \n', E, '\n')
            # print('Average error =', E.mean().round(3), '\n')
            # print('Ending round', k+1, '----------------------- \n \n \n')
         
            
    def test(self):
        test_results = np.zeros(self.test_len)
        
        for i in range(self.test_len):
            j = i + np.random.randint(9000)
            image = test_images[j]
            y0 = image.flatten() / 255
            x1 = np.matmul(y0, self.w1) + self.b1
            y1 = 1 / (1 + np.exp(-x1))
            x2 = np.matmul(y1, self.w2) + self.b2
            y2_bis = np.exp(x2)
            y2 = y2_bis / y2_bis.sum()
            result = y2.argmax()
            
            plt.figure(j+1)
            plt.subplot(121)
            plt.imshow(image, 'Greys')
            plt.subplot(122)
            plt.bar(range(10), y2)
            plt.xticks(range(10), range(10))
            print('result', j+1, '=', result)
            
            
    
    def run(self):
        self.train(train_images, train_labels)




nn = NN()
one = nn.w1
nn.run()
two = nn.w1
# nn.run()
three = nn.w1
con = np.array_equal(two, three)

nn.error = np.array(nn.error)
nn.inputs = np.array(nn.inputs)
nn.outputs = np.array(nn.outputs)

e = nn.error
inp = nn.inputs
out = nn.outputs

plt.figure(0)
plt.plot(nn.error, '.-')

# for i in range(len(e)):
#     if i in [0, 9, 49, 99, len(e)]:
#         plt.figure(i+1)
#         plt.subplot(121)
#         plt.imshow(inp[i], 'Greys')
#         plt.subplot(122)
#         plt.bar(range(10), out[i])






















d













