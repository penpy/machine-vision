import struct
import array
import numpy as np
from matplotlib import pyplot as plt

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
    filename = FILE_NAMES[i]
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
# Neural network
# ===================


class NN:
    
    def __init__(self):
        # 3 layers: input x, hidden h, output y
        
        self.ee = []
        self.ya_list = []
        self.all = []
        
        self.n_x = 28 * 28
        self.n_h1 = 16
        self.n_h2 = 16
        self.n_y = 10
        
        self.shape_w1 = (self.n_x, self.n_h1)
        self.shape_w2 = (self.n_h1, self.n_h2)
        self.shape_w3 = (self.n_h2, self.n_y)
        
        self.w1 = np.random.uniform(-1, 1, self.shape_w1) / 28
        self.w2 = np.random.uniform(-1, 1, self.shape_w2) / 4
        self.w3 = np.random.uniform(-1, 1, self.shape_w3) / 3
        
        self.b1 = np.full(self.n_h1, 0.0)
        self.b2 = np.full(self.n_h2, 0.0)
        self.b3 = np.full(self.n_y, 0.0)
        
        self.n_iter = 200
        self.batch_size = 16
        self.learn_r = 0.01
        self.n_tests = 100
        self.train_indices = np.arange(LEN_TRAIN)
        
        if self.n_x != 28 * 28:
            raise ValueError('input vector size must be equal to the numbers of pixels in one image')
        if self.n_y != 10:
            raise ValueError('output vector size must be equal to 10')
        if LEN_TRAIN < self.n_iter * self.batch_size:
            raise ValueError('Too many iterations for 60 000 training images')
    
    def forward_propagation(self, image):
        # Activation: sigmoid for ha2, softmax for ya
        x = image.flatten() / 255
        
        h1 = np.dot(x, self.w1) + self.b1
        ha1 = h1 * (0 < h1)
        
        h2 = np.dot(ha1, self.w2) + self.b2
        ha2 = h2 * (0 < h2)
        
        y = np.dot(ha2, self.w3) + self.b3
        exp_y = np.exp(y - y.max())
        self.ya = exp_y / exp_y.sum()
        
        if self.ya.min() < 0:
            raise ValueError('WTF')
        
        return x, ha1, ha2, self.ya
    
    
    def backpropagation(self, x, ha1, ha2, ya, t):
        # Derivatives d_u of the cross-entropy loss with respect to each parameter u
        d_b3 = ya - t
        d_w3 = np.outer(ha2, d_b3)
        
        d_b2 = np.dot(self.w3, d_b3)
        d_w2 = np.outer(ha1, d_b2)
        
        d_b1 = np.dot(self.w2, d_b2)
        d_w1 = np.outer(x, d_b1)
        
        return d_w1, d_w2, d_w3, d_b1, d_b2, d_b3
    
    def train(self):
        
        np.random.shuffle(self.train_indices)
        
        for k in range(self.n_iter):
            
            # Initialization of the derivatives for the batch
            sum_d_w1 = np.zeros(self.shape_w1)
            sum_d_w2 = np.zeros(self.shape_w2)
            sum_d_w3 = np.zeros(self.shape_w3)
            sum_d_b1 = np.zeros(self.n_h1)
            sum_d_b2 = np.zeros(self.n_h2)
            sum_d_b3 = np.zeros(self.n_y)
            
            for i in range(self.batch_size):
                index = self.train_indices[k * self.batch_size + i]
                self.index_image = index
                image = TRAIN_IMAGES[index]
                t = np.zeros(10)
                t[TRAIN_LABELS[index]] = 1
                
                x, ha1, ha2, ya = self.forward_propagation(image)
                d_w1, d_w2, d_w3, d_b1, d_b2, d_b3 = self.backpropagation(x, ha1, ha2, ya, t)
                
                # assert ya.min() > 0
                # self.ee.append(-np.log(ya[TRAIN_LABELS[index]]))

                sum_d_w1 += d_w1
                sum_d_w2 += d_w2
                sum_d_w3 += d_b3
                sum_d_b1 += d_b1
                sum_d_b2 += d_b2
                sum_d_b3 += d_b3
            
            a = sum_d_w1.sum().round(2)
            b = sum_d_w2.sum().round(2)
            c = sum_d_w3.sum().round(2)
            d = sum_d_b1.sum().round(2)
            e = sum_d_b2.sum().round(2)
            f = sum_d_b3.sum().round(2)
        
            self.all.append(np.array([a,b,c,d,e,f]))
            
            self.w1 -= self.learn_r * sum_d_w1
            self.w2 -= self.learn_r * sum_d_w2
            self.w3 -= self.learn_r * sum_d_w3
            self.b1 -= self.learn_r * sum_d_b1
            self.b2 -= self.learn_r * sum_d_b2
            self.b3 -= self.learn_r * sum_d_b3
    
    def test(self, test_index):
        image = TEST_IMAGES[test_index]
        label = TEST_LABELS[test_index]
        x, ha1, ha2, ya = self.forward_propagation(image)
        result = ya.argmax()
        return result == label
    
    def accuracy(self):
        # Returns the proportion of correctly guessed digits by the network
        acc = 0
        for i in range(self.n_tests):
            if self.test(i):
                acc += 1
        return acc / self.n_tests
    
    def loss(self, test_index):
        image = TEST_IMAGES[test_index]
        label = TEST_LABELS[test_index]
        x, ha1, ha2, ya = self.forward_propagation(image)
        return -np.log(ya[label])










# plt.figure(0)
# nn = NN()
# nn.train()
# plt.plot(range(len(nn.ee)), nn.ee, '.-')

# ===================
# Plot performance
# ===================

def perf():
    nn = NN()
    nn.n_iter = 5
    y = [nn.accuracy()]
    x = [0]
    tot = 20
    string = (tot - 12) * ' ' + '| end'
    print('Progression:' + string)
    for i in range(tot):
        print('=', end='')
        nn.train()
        y.append(nn.accuracy())
        x.append(nn.n_iter + x[-1])
    print('\nFinished.')
    plt.plot(x, y, 'o-', lw=1)
    plt.grid()
    plt.ylabel('Proportion of correct guesses')
    plt.xlabel('Number of iterations')
    plt.title('Accuracy during training process')










class Check:
    
    def __init__(self):
        self.n_epochs = 5
        self.nn = NN()
    
    def train_nn(self):
        print('Starting training process...')
        for i in range(self.n_epochs):
            self.nn.train()
            i += 1
            str_1 = 'Epoch ' + str(i) + '/' + str(self.n_epochs) + ' completed'
            str_2 = '|' + '=' * i + ' ' * (self.n_epochs - i) + '|'
            print(str_1, str_2)
        print('Training process finished.')
        self.example()
    
    def example(self):
        random_n = np.random.randint(10000)
        image = TEST_IMAGES[random_n]
        label = TEST_LABELS[random_n]
        x, h, ha, y, ya = self.nn.forward_propagation(image)
        plt.figure(random_n)
        if label == ya.argmax():
            plt.suptitle('CORRECT', color='g', fontsize=16, fontweight='bold')
        else:
            plt.suptitle('INCORRECT', color='r', fontsize=16, fontweight='bold')
        plt.subplot(121)
        plt.imshow(image, 'Greys')
        plt.title('Label = ' + str(label))
        plt.subplot(122)
        plt.bar(range(10), ya)
        plt.xticks(range(10), range(10))
        plt.ylim(0, 1)
        plt.xlim(-.5, 9.5)
        plt.grid(axis='y')
        plt.title('Guessed digit = ' + str(ya.argmax()))