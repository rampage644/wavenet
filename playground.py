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
import wavenet.models as models


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

mask = np.ones([Cout, Cin, kh, kw])

yc, xc = kh // 2, kw // 2

mask[:, :, yc+1:, :] = 0.0
mask[:, :, yc:, xc+1:] = 0.0

mtype = 'B'
value = 0.0 if mtype == 'A' else 1.0




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


mask[bmask(0, 1), yc, xc] = .0
mask[bmask(0, 2), yc, xc] = .0

mask[bmask(1, 2), yc, xc] = .0

mask[:, :, yc, xc]


#%%
import importlib
importlib.reload(models)

Cin, Cout, kh, kw = 5, 12, 3, 3
link = models.MaskedConvolution2D(Cin, Cout, 3, mask='B', pad=1)
link.W.data = np.ones_like(link.W.data)

# model = models.PixelCNN(3, 16, 5, 8, 100)
# zeros = np.zeros([1, 3, 11, 13], dtype='f')

img = np.ones([1, 5, 6, 7]).astype('f')
img[:, 1, :, :] *= 10
img[:, 2, :, :] *= 100
img[:, 4, :, :] *= 10
img[:, 5, :, :] *= 100
img[:, 7, :, :] *= 10
img[:, 8, :, :] *= 100

h = link(img)


out = link(img).data.astype('i')
batch_size, channels, height, width = out.shape
print(batch_size, channels, height, width)
print(out[0, 0], out[0, 1], out[0, 2], out[0, 3], out[0, 11],  sep='\n')
out = np.reshape(out, [batch_size, 4, 3, height, width])
out = np.transpose(out, [0, 2, 1, 3, 4])
print(out[0, :, 0], out[0, :, 1], out[0, :, 2], sep='\n')

out.shape

122*4

#%%
import chainer.links as L

N = 3
conv1 = L.Convolution2D(1, 1, )
conv1.W.data

conv2 = L.Convolution2D(1, 1, [1, N//2+1], pad=[0, N//2+1], initialW=1.0)
conv2.W.data

# this filter rank is 1, so this convolution is separable
conv_combined = L.Convolution2D(1, 1, N, pad=1, initialW=1.0)
conv_combined.W.data


dims = 1, 1, 5, 6  # B, Cin, H, W
input = np.arange(np.prod(dims)).astype('f').reshape(dims)

input
conv2(conv1(input)).data
conv_combined(input).data

(conv2(conv1(input)) - conv_combined(input)).data
(conv1(conv2(input)) - conv_combined(input)).data


import wavenet.models as models
import importlib
importlib.reload(models)

conv1 = models.CroppedConvolution(1, 1, [1, 2], pad=[0, 2], initialW=1.0)
conv1.pad, conv1.ksize

input, conv1(input).data

#
