


import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
import mnist

# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()


# -------------------------------------------------------------
# Training
# -------------------------------------------------------------
'''
Notations :

w1, w2, w3 : weight matrices 1, 2 and 3
b1, b2, b3 : bias vectors 1, 2 and 3
a : activation functi
da : derivative of the activation function

y0 : entry vector (stage 0)
x1, x2, x3 : vectors of stages 1, 2 and 3 before activation function
y1, y2, y3 : vectors of stages 1, 2 and 3 after activation function
t : target

E : error

du_dv : derivative of u with respect to v
'''


entry_size = 28 * 28
entry = np.zeros(entry_size)
stages_size = 16
exit_size = 10
learning_rate = 0.00000000001
batch_size = 100
n_batches = 100

w1_shape = (entry_size, stages_size)
w2_shape = (stages_size, stages_size)
w3_shape = (stages_size, exit_size)

global b1, b2, b3, w1, w2, w3

w1 = np.random.uniform(-1, 1, w1_shape)
w2 = np.random.uniform(-1, 1, w2_shape)
w3 = np.random.uniform(-1, 1, w3_shape)


b1 = np.full((stages_size), 0.)
b2 = np.full((stages_size), 0.)
b3 = np.full((exit_size), 0.)

print('b1[2] =', b1[2])
print('b2[2] =', b2[2])
print('b3[2] =', b3[2])

print('w1[4,6] =', w1[4, 6])
print('w2[4,6] =', w2[4, 6])
print('w3[4,6] =', w3[4, 6])


# f_activ = np.tanh
# df_activ = lambda x: 1 - np.tanh(x)**2
# f_activ = lambda x: np.fmax(0, 1 - np.exp(-10 * x))

a = lambda x: 1 / (1 + np.exp(-x))
da = lambda x: a(x) * (1 - a(x))

'''
def f_error(target, output):
    error = 0.5 * (target - output) ** 2
    return error.sum()
'''

def f_E(output, target):
    return .5 * (target - output) ** 2


def update_ws(images, labels):
    
    global b1, b2, b3, w1, w2, w3
    
    output_batch = np.zeros((batch_size, exit_size))
    target_batch = np.zeros((batch_size, exit_size))

    sum_db1 = np.zeros(stages_size)
    sum_db2 = np.zeros(stages_size)
    sum_db3 = np.zeros(exit_size)
    
    sum_dw1 = np.zeros(w1_shape)
    sum_dw2 = np.zeros(w2_shape)
    sum_dw3 = np.zeros(w3_shape)
    
    for i in range(batch_size):
        
        # Stages
        y0 = images[i].flatten() / 255
        x1 = np.matmul(y0, w1) + b1
        y1 = a(x1)
        x2 = np.matmul(y1, w2) + b2
        y2 = a(x2)
        x3 = np.matmul(y2, w3) + b3
        y3_bis = np.exp(x3 - max(x3))
        y3 = y3_bis / y3_bis.sum()
        output_batch[i] = y3
        
        # Target
        t = np.zeros(exit_size)
        t[labels[i]] = 1

        target_batch[i] = t
        
        # Derivatives
        
        dE_db3 = (y3 - t) * da(x3)
        da_x2 = da(x2)
        P0 = np.matmul(w3, dE_db3)
        P1 = np.matmul(w3, dE_db3)
        P2 = np.matmul(P1 * da_x2, w2)
        dE_db2 = P0 * da_x2
        dE_db1 = P2 * da(x1)
        
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
    # b1 = learning_rate * sum_db1 / batch_size
    # b2 = learning_rate * sum_db2 / batch_size
    # b3 = learning_rate * sum_db3 / batch_size
    
    # print('sum dw1 =', sum_dw1.mean())
    # print('sum dw2 =', sum_dw2[4,6])
    # print('sum dw3 =', sum_dw3[4, 6])
    #
    # print('b1[2] =', b1[2])
    # print('b2[2] =', b2[2])
    # print('b3[2] =', b3[2])

    w1 = learning_rate * sum_dw1 / batch_size
    w2 = learning_rate * sum_dw2 / batch_size
    w3 = learning_rate * sum_dw3 / batch_size

    # print('w1[4,6] =', w1[4,6])
    # print('w2[4,6] =', w2[4,6])
    # print('w3[4,6] =', w3[4,6])
    
    return output_batch, target_batch


def training(train_images, train_labels):
    for k in range(n_batches):
        
        print('Starting round', k+1, '-----------------------')
        start = k * batch_size
        stop = start + batch_size
        images = train_images[start : stop]
        labels = train_labels[start : stop]
        output_batch, target_batch = update_ws(images, labels)
        print('output_batch 3 =', output_batch[3].round(2))
        print('target_batch 3 =', target_batch[3].round(2))
        E = f_E(output_batch, target_batch)
        #print('Error = \n', E, '\n')
        print('Average error =', E.mean().round(2))
        print('Ending round', k+1, '----------------------- \n \n \n')
        
        
def main():
    training(train_images, train_labels)



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