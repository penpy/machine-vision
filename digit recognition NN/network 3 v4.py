


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
a : activation function
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
learning_rate = 0.05
batch_size = 10
n_batches = 10

w1_shape = (entry_size, stages_size)
w2_shape = (stages_size, stages_size)
w3_shape = (stages_size, exit_size)

global w1, w2, w3

w1 = np.random.uniform(-1, 1, w1_shape)
w2 = np.random.uniform(-1, 1, w2_shape)
w3 = np.random.uniform(-1, 1, w3_shape)


b1 = np.full((stages_size), 0.1)
b2 = np.full((stages_size), 0.1)
b3 = np.full((exit_size), 0.1)

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


def f_dE_dy3(output, target):
    # error = 1/2 * (targets - outputs) ** 2
    # derror_doutput = - (target - output)
    return output - target


def f_dE_dx3(x3, y3, t):
    # derror_doutput = f_derror_doutput(output, target)
    # doutput_dstage3 = df_activ(stage_3)
    # derror_dstage3 = derror_doutput * doutput_dstage3
    return f_dE_dy3(y3, t) * da(x3)


# dE_dw3 = dE_dx3 * dx3_dw3

def f_dE_dw3(dE_dx3, y2, x3, y3, t):
    dE_dw3 = np.zeros(w3_shape)
    for i, dE_dw3i in enumerate(dE_dw3):
        for j, dE_dw3ij in enumerate(dE_dw3i):
            dE_dw3ij = dE_dx3[j] * y2[i]
    return dE_dw3

    


# dE_dw2 = sum dEk_dw2
# dEk_dw2 = dEk_dx3k * dx3k_dy2j * dy2j_dx2j * dx2j_dw3

def f_dE_dw2(dy2_dx2, dE_dx3, y1, x2, y2, x3, y3, t):
    dE_dw2 = np.zeros(w2_shape)
    for i, dE_dw2i in enumerate(dE_dw2):
        for j, dE_dw2ij in enumerate(dE_dw2i):
            for k in range(exit_size):
                dEk_dw2 = dE_dx3[k] * w3[j][k] * dy2_dx2[j] * y1[i]
                dE_dw2ij += dEk_dw2
    return dE_dw2



# dE_dw1 = sum dEl_dw1
# dEl_dw1 = dEl_dx3l * sum(dx3l_dy2k * dy2k_dx2k * dx2k_dy1j * dy1j_dx1j * dx1j_dw1)

def f_dE_dw1(dy2_dx2, dE_dx3, y0, x1, y1, x2, y2, x3, y3, t):
    dE_dw1 = np.zeros(w1_shape)
    for i, dE_dw1i in enumerate(dE_dw1):
        for j, dE_dw1ij in enumerate(dE_dw1i):
            dEl_dw1 = 0
            for l in range(exit_size):
                for k in range(stages_size):
                    sum = w3[k][l] * dy2_dx2[k] * w2[j][k] * da(x1[j]) * y0[i]
                dE_dw1ij += dE_dx3[l] * sum
    return dE_dw1





def update_ws(images, labels):
    
    global w1, w2, w3
    
    output_batch = np.zeros((batch_size, exit_size))
    target_batch = np.zeros((batch_size, exit_size))
    
    sum_dw1 = np.zeros(w1_shape)
    sum_dw2 = np.zeros(w2_shape)
    sum_dw3 = np.zeros(w3_shape)
    
    for i in range(batch_size):
        
        # Stages
        y0 = images[i].flatten() / 255
        x1 = np.dot(entry, w1) + b1
        y1 = a(x1)
        x2 = np.dot(y1, w2) + b2
        y2 = a(x2)
        x3 = np.dot(y2, w3) + b3
        y3 = a(x3)

        output_batch[i] = y3
        
        # Target
        t = np.zeros(exit_size)
        t[labels[i]] = 1

        target_batch[i] = t
        
        # Derivatives
        dE_dx3 = f_dE_dx3(x3, y3, t)
        dy2_dx2 = da(x2)
        sum_dw1 += f_dE_dw1(dy2_dx2, dE_dx3, y0, x1, y1, x2, y2, x3, y3, t)
        sum_dw2 += f_dE_dw2(dy2_dx2, dE_dx3, y1, x2, y2, x3, y3, t)
        sum_dw3 += f_dE_dw3(dE_dx3, y2, x3, y3, t)
    
    # update w
    w1 = learning_rate * sum_dw1 / batch_size
    w2 = learning_rate * sum_dw2 / batch_size
    w3 = learning_rate * sum_dw3 / batch_size
    
    return output_batch, target_batch


def training(train_images, train_labels):
    for k in range(n_batches):
        
        print('Starting round', k+1, '----------------------- \n')
        start = k * batch_size
        stop = start + batch_size
        images = train_images[start : stop]
        labels = train_labels[start : stop]
        output_batch, target_batch = update_ws(images, labels)
        E = f_E(output_batch, target_batch)
        print('Error = \n', E, '\n')
        print('Average error =', E.mean(), '\n')
        print('Ending round', k+1, '----------------------- \n \n')
        
        
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