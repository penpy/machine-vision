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
        
        self.len_y0 = 28 * 28
        self.len_y1 = 256
        self.len_y2 = 128
        self.len_y3 = 10
        self.learn_r = 0.001
        self.batch_size = 100
        self.n_epochs = 10
        self.batch_shape = (self.batch_size, self.len_y3)
        
        self.w1_shape = (self.len_y0, self.len_y1)
        self.w2_shape = (self.len_y1, self.len_y2)
        self.w3_shape = (self.len_y2, self.len_y3)
        
        self.b1 = np.full((self.len_y1), 0.0)
        self.b2 = np.full((self.len_y2), 0.0)
        self.b3 = np.full((self.len_y3), 0.0)

        self.w1 = np.random.uniform(0, 1, self.w1_shape) / np.sqrt(self.len_y0)
        self.w2 = np.random.uniform(0, 1, self.w2_shape) / np.sqrt(self.len_y1)
        self.w3 = np.random.uniform(0, 1, self.w3_shape) / np.sqrt(self.len_y2)
        
        self.a = lambda x: 1 / (1 + np.exp(-x))
        self.da = lambda x: self.a(x) * (1 - self.a(x))
    
    
    
    
    
    
    def update_ws(self, images, labels):

        target_batch = np.zeros(self.batch_shape)
    
        sum_db1 = np.zeros(self.len_y1)
        sum_db2 = np.zeros(self.len_y2)
        sum_db3 = np.zeros(self.len_y3)
        
        sum_dw1 = np.zeros(self.w1_shape)
        sum_dw2 = np.zeros(self.w2_shape)
        sum_dw3 = np.zeros(self.w3_shape)
        
        for i in range(self.batch_size):
            
            # Stages
            
            y0 = images[i].flatten() * 0.99 / 255 + 0.01
            x1 = np.matmul(y0, self.w1) + self.b1
            y1 = self.a(x1)
            x2 = np.matmul(y1, self.w2) + self.b2
            y2 = self.a(x2)
            x3 = np.matmul(y2, self.w3) + self.b3
            y3_bis = np.exp(x3)
            y3 = y3_bis / y3_bis.sum()
            # Softmax function
            
            
            # Target
            t = np.zeros(self.len_y3)
            t[labels[i]] = 1
    
            target_batch[i] = t
            
            
            self.inputs.append(images[i])
            self.outputs.append(y3)
            self.error.append((.5 * (t - y3) ** 2).mean())
            
            # Derivatives
            # derivative of Softmax: dS/dx = S(1-S)

            dE_db3 = (y3 - t) * y3 * (1 - y3)
            P0 = np.matmul(self.w3, dE_db3)
            da_x2 = self.da(x2)
            dE_db2 = P0 * da_x2
            P1 = np.matmul(self.w3, dE_db3)
            P2 = np.matmul(self.w2, P1 * da_x2)
            dE_db1 = P2 * self.da(x1)
            
            dE_dw3 = np.outer(y2, dE_db3)
            dE_dw2 = np.outer(y1, dE_db2)
            dE_dw1 = np.outer(y0, dE_db1)
            
            sum_db1 += dE_db1
            sum_db2 += dE_db2
            sum_db3 += dE_db3
            
            sum_dw1 += dE_dw1
            sum_dw2 += dE_dw2
            sum_dw3 += dE_dw3
        
        # update ws an bs
        # self.b1 = learning_rate * sum_db1 / batch_size
        # self.b2 = learning_rate * sum_db2 / batch_size
        # self.b3 = learning_rate * sum_db3 / batch_size
        
        # print('sum dw1 =', sum_dw1.mean())
        # print('sum dw2 =', sum_dw2[4,6])
        # print('sum dw3 =', sum_dw3[4, 6])
        #
        # print('b1[2] =', self.b1[2])
        # print('b2[2] =', self.b2[2])
        # print('b3[2] =', self.b3[2])

        self.w1 -= self.learn_r * sum_dw1 / self.batch_size
        self.w2 -= self.learn_r * sum_dw2 / self.batch_size
        self.w3 -= self.learn_r * sum_dw3 / self.batch_size
        
    
        # print('w1[4,6] =', self.w1[4,6])
        # print('w2[4,6] =', self.w2[4,6])
        # print('w3[4,6] =', self.w3[4,6])
    
    
    
    
    def train(self, train_images, train_labels):
        for k in range(self.n_epochs):
            
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

for i in range(len(e)):
    if i in [0, 9, 49, 99, len(e)]:
        plt.figure(i+1)
        plt.subplot(121)
        plt.imshow(inp[i], 'Greys')
        plt.subplot(122)
        plt.bar(range(10), out[i])











# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

'''
my_image = img.imread('number_photo.jpg', format='jpg')
my_image = np.array(my_image)
new_size = 28

hight, length, color = my_image.shape

if hight < length :
    pixel_size = hight // new_size
    i_start = 0
    j_start = int((length - hight) / 2)
else:
    pixel_size = length // new_size
    i_start = int((length - hight) / 2)
    j_start = 0




def entry_func(my_image):
    # Convert 720 * 1080 rvb image into 784 black&white

    for index in range(entry_size):
        sum = 0
        for i in range(pixel_size):
            for j in range(pixel_size):
                I = i + i_start + pixel_size * (index // new_size)
                J = j + j_start + pixel_size * (index % new_size)
                for K in range(3):
                    sum += my_image[I][J][K]
        entry[index] = sum
    return entry / entry.max()



test_entry = entry_func(my_image)
'''
