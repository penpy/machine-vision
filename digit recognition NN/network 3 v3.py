


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

entry_size = 28 * 28
entry = np.zeros(entry_size)
stages_size = 16
exit_size = 10
learning_rate = 0.1
batch_size = 3
n_batches = 2

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

f_activ = lambda x: 1 / (1 + np.exp(-x))
df_activ = lambda x: f_activ(x) * (1 - f_activ(x))

'''
def f_error(target, output):
    error = 0.5 * (target - output) ** 2
    return error.sum()
'''

def error(targets, outputs):
    return .5 * (targets - outputs) ** 2


def f_derror_doutput(output, target):
    # error = 1/2 * (targets - outputs) ** 2
    # derror_doutput = - (target - output)
    return output - target


def f_derror_dstage3(stage_3, output, target):
    # derror_doutput = f_derror_doutput(output, target)
    # doutput_dstage3 = df_activ(stage_3)
    # derror_dstage3 = derror_doutput * doutput_dstage3
    return f_derror_doutput(output, target) * df_activ(stage_3)


def de_dw(entry, stage_1, stage_2, stage_3, output, target):
    
    derror_dstage3 = f_derror_dstage3(stage_3, output, target)
        
    # matrix_3 --------------------------------
    de_dw3 = np.zeros(w3_shape)
    for i in range(stages_size):
        for j in range(exit_size):
            de_dw3[i][j] = stage_2[i] * derror_dstage3[j]
            
    # matrix_2 --------------------------------
    # de_dw2 = sum dek_dw2
    # dek_dw2 = dek_dstg3k * dstg3k_dstg2j * dstg2j_dw2
    de_dw2 = np.zeros(w2_shape)
    for i in range(stages_size):
        for j in range(stages_size):
            for k in range(exit_size):
                # dek_dstg3k = derror_dstage3[k]
                # dstg3k_dstg2j = w3[j][k]
                # dstg2j_dw2 = stage_1[i]
                dek_dw2 = derror_dstage3[k] * w3[j][k] * stage_1[i]
                de_dw2[i][j] += dek_dw2

    # matrix_1 --------------------------------
    # de_dw1 = sum dek_dw1
    # dek_dw1 = dek_dstg3k * sum(dstg3k_dstg2l * dstg2_dstg1 * dstg1_dw1)
    # --------------------------
    de_dw1 = np.zeros(w1_shape)
    for i in range(entry_size):
        for j in range(stages_size):
            for k in range(exit_size):
                sum = 0
                for l in range(stages_size):
                    # dstg3k_dstg2l = w3[l][k]
                    # dstg2l_dstg1j = w2[j][l]
                    # dstg1_dw1 = entry[i]
                    sum += w3[l][k] * w2[j][l] * entry[i]
                # dek_dstg3k = derror_dstage3[k]
                # dek_dw1 = derror_dstage3[k] * sum
                de_dw1[i][j] += derror_dstage3[k] * sum
            
    # return --------------------------------
    return de_dw1, de_dw2, de_dw3
    


def update_ws(images, labels):
    global w1, w2, w3
    sum1 = np.zeros((entry_size, stages_size))
    sum2 = np.zeros((stages_size, stages_size))
    sum3 = np.zeros((stages_size, exit_size))
    for i in range(batch_size):
        entry = images[i].flatten() / 255
        stage_1 = np.dot(entry, w1) + b1
        stage_2 = np.dot(stage_1, w2) + b2
        stage_3 = np.dot(stage_2, w3) + b3
        output = f_activ(stage_3)
        target = np.zeros(exit_size)
        target[labels[i]] = 1
        
        my_error_vector = error(target, output)
        my_error = round(sum(my_error_vector), 3)
        print('error =', my_error)
        
        de_dw1, de_dw2, de_dw3 = de_dw(entry, stage_1, stage_2, stage_3, output, target)
        
        sum1 +=  de_dw1
        sum2 += de_dw2
        sum3 += de_dw3
    
    w1 = learning_rate * sum1 / batch_size
    w2 = learning_rate * sum2 / batch_size
    w3 = learning_rate * sum3 / batch_size
        


def training(train_images, train_labels):
    for k in range(n_batches):
        
        print('Starting round', k+1, '-----------------------')
        start = k * batch_size
        stop = start + batch_size
        images = train_images[start : stop]
        labels = train_labels[start : stop]
        update_ws(images, labels)
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