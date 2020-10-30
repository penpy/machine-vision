# -*- coding: utf_8 -*-
"""
Make sure you have installed numpy and mnist in your python environmemt
"""

import numpy as np
import mnist

# =====================
# Collecting the MNIST datasets of handwritten digits
# =====================

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()


# =====================
# Neural network
# =====================

# Length of each layers (input x, hidden h, output y)
len_x = 28 * 28
len_h = 16
len_y = 10

# Shapes of weight matrices
shape_w1 = (len_x, len_h)
shape_w2 = (len_h, len_y)

# Initialization of weight matrices (w1 and w2) with random numbers
w1 = np.random.uniform(-1, 1, shape_w1) / np.sqrt(len_x)
w2 = np.random.uniform(-1, 1, shape_w2) / np.sqrt(len_h)

# Initialization of bias vectors (b1 and b2) with zeros
b1 = np.full(len_h, 0.)
b2 = np.full(len_y, 0.)

# Training parameters
n_iterations = 200
batch_size = 32
learn_r = 0.05


def forward_propagation(image):
    # Returns the vectors of each layers for a given image
    
    # Input layer
    x = image.flatten() / 255
    
    # Hidden layer (activation with sigmoid function)
    h = np.dot(x, w1) + b1
    ha = 1 / (1 + np.exp(-h))
    
    # Output layer (activation with softmax function)
    y = np.dot(ha, w2) + b2
    exp_y = np.exp(y)
    ya = exp_y / exp_y.sum()
    
    return x, h, ha, y, ya


def backpropagation(x, h, ha, ya, t):
    # Derivatives of the cross entropy loss with respect to each parameter
    d_b2 = ya
    d_b2[t] -= 1
    d_w2 = np.outer(ha, d_b2)
    d_b1 = np.dot(w2, d_b2) * ha * (1 - ha)
    d_w1 = np.outer(x, d_b1)
    return d_w1, d_w2, d_b1, d_b2


def train():
    # This function updates the weights and biases to try to minimize the loss

    for k in range(n_iterations):
        
        # Initialization of the derivatives for the batch
        sum_d_w1 = np.zeros(shape_w1)
        sum_d_w2 = np.zeros(shape_w2)
        sum_d_b1 = np.zeros(len_h)
        sum_d_b2 = np.zeros(len_y)
        
        for i in range(batch_size):
            
            # index of the training image and label
            index = k * batch_size + i
            image = train_images[index]
            t = train_labels[index]
            
            x, h, ha, y, ya = forward_propagation(image)
            d_w1, d_w2, d_b1, d_b2 = backpropagation(x, h, ha, ya, t)
            
            sum_d_w1 += d_w1
            sum_d_w2 += d_w2
            sum_d_b1 += d_b1
            sum_d_b2 += d_b2

        # Updating weights and biases
        w1[:] -= learn_r * sum_d_w1
        w2[:] -= learn_r * sum_d_w2
        b1[:] -= learn_r * sum_d_b1
        b2[:] -= learn_r * sum_d_b2
        # The [:] notation is used to modify w1, w2, b1 and b2
        # Without this notation they are considered as undefined local variables
        

def accuracy():
    # Returns the proportion of correctly guessed digits in all the test set
    acc = 0
    len_tests = len(test_images)
    for i in range(len_tests):
        x, h, ha, y, ya = forward_propagation(test_images[i])
        if ya.argmax() == test_labels[i]:
            acc += 1
    return acc / len_tests