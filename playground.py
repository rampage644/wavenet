#%%
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import sys
import scipy.misc
import numpy as np

import matplotlib.pyplot as plt

import chainer
import chainer.training
import chainer.training.extensions as extensions
import chainer.links as L

import wavenet.utils as utils


#%%
def simple(image):
    return (image > 0.5).astype('f')

#%%
test, train = chainer.datasets.get_mnist(False, ndim=2)

def plot(sample):
    plt.subplot(141)
    plt.imshow(sample, cmap='Greys', interpolation='none')
    plt.subplot(142)
    plt.imshow(utils.binarize(sample), cmap='Greys', interpolation='none')
    plt.subplot(143)
    plt.imshow(utils.quantisize(sample, 2), cmap='Greys', interpolation='none')
    plt.subplot(144)
    plt.imshow(simple(sample), cmap='Greys', interpolation='none')


plot(test[0])
plot(test[1])

#%%
B, CHANNELS, DIM, H, W = 16, 256, 3, 27, 26
input = np.zeros([B, CHANNELS * DIM, H, W])
indices = np.arange(CHANNELS * DIM)
indices % 3 == 2

input[:, indices % 3 == 0, :, :] = 1.0
input[:, indices % 3 == 1, :, :] = 2.0
input[:, indices % 3 == 2, :, :] = 3.0

r = np.reshape(input, [B, CHANNELS, DIM, H, W])
rt = np.transpose(r, [0, 2, 1, 3, 4])

r.shape
r[0, :, 2, 0, 0]

rt.shape


#%%
Cin, Cout, kh, kw = 6, 10, 5, 5
mask = np.ones([Cout, Cin, kh, kw])

yc, xc = kh // 2, kw // 2

mask[:, :, yc+1:, :] = 0.0
mask[:, :, yc:, xc+1:] = 0.0

mtype = 'B'
value = 0.0 if mtype == 'A' else 1.0




a1 * a2

mask[a1 * a2] = 0
cout_idx.shape
cin_idx.shape

mask

def bmask(i_out, i_in):
    cout_idx = np.expand_dims(np.arange(Cout) % 3 == i_out, 1)
    cin_idx = np.expand_dims(np.arange(Cin) % 3 == i_in, 0)
    a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
    return a1 * a2


for j in range(3):
    mask[bmask(j, j), yc, xc] = value

mask[bmask(1, 0), yc, xc] = 0.0
mask[bmask(2, 0), yc, xc] = 0.0
mask[bmask(2, 1), yc, xc] = 0.0




mask[:, :, yc, xc]



