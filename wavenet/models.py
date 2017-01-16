'''Models.'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy

import chainer
import chainer.functions as F
import chainer.links as L


class MaskedConvolution2D(L.Convolution2D):
    def __init__(self, *args, mask='B', **kwargs):
        super(MaskedConvolution2D, self).__init__(
            *args, **kwargs
        )

        kh, kw = (self.ksize, ) * 2
        self.mask = self.xp.ones([kh, kw]).astype('f')
        yc, xc = kh // 2, kw // 2

        self.mask[yc+1:, :] = 0.0
        self.mask[yc:, xc+1:] = 0.0
        if mask == 'A':
            self.mask[yc, xc] = 0.0

        self.mask = F.broadcast_to(chainer.Variable(self.mask), self.W.shape)
        self.add_persistent('convmask', self.mask)

    def __call__(self, x):
        if self.has_uninitialized_params:
            with chainer.cuda.get_device(self._device_id):
                self._initialize_params(x.shape[1])

        return chainer.functions.connection.convolution_2d.convolution_2d(
            x, self.W * self.convmask, self.b, self.stride, self.pad, self.use_cudnn,
            deterministic=self.deterministic)


class ResidualBlock(chainer.Chain):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__(
            conv1=L.Convolution2D(in_channels, in_channels // 2, 1),
            conv2=MaskedConvolution2D(in_channels // 2, in_channels // 2, 3, pad=1),
            conv3=L.Convolution2D(in_channels // 2, in_channels, 1)
        )

    def __call__(self, x):
        # XXX: relu before convolution?
        h = self.conv1(F.relu(x))
        h = self.conv2(F.relu(h))
        h = self.conv3(F.relu(h))
        return x + h


class ResidualBlockList(chainer.ChainList):
    def __init__(self, block_num, hidden_dims, mask='B'):
        blocks = [ResidualBlock(hidden_dims) for _ in range(block_num)]
        super(ResidualBlockList, self).__init__(*blocks)

    def __call__(self, x):
        h = x
        for block in self:
            # XXX: do we need relu here?
            # magenta repo review says yes, no relu in carpedm20 implemention, nothing in paper
            h = F.relu(h)
        return h


class PixelCNN(chainer.Chain):
    def __init__(self, in_channels, hidden_dims, block_num):
        super(PixelCNN, self).__init__(
            conv1=MaskedConvolution2D(in_channels, hidden_dims, 7, pad=3),
            blocks=ResidualBlockList(block_num, hidden_dims),
            conv2=L.Convolution2D(hidden_dims, hidden_dims, 1),
            conv3=L.Convolution2D(hidden_dims, hidden_dims, 1),
            conv4=L.Convolution2D(hidden_dims, 1, 1)
        )

    def __call__(self, x):
        # TODO: this is for mnist, refactor
        h = F.reshape(F.expand_dims(x, 1), [-1, 1, 28, 28])

        # XXX: activation?
        h = self.conv1(h)
        h = F.relu(self.blocks(h))
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)

        return h


# TODO: rename class
class Classifier(chainer.Chain):
     def __init__(self, predictor):
         super(Classifier, self).__init__(predictor=predictor)

     def __call__(self, x, t):
         y = self.predictor(x)

         # TODO: replace hard-coded reshaping
         nll = F.bernoulli_nll(x, F.reshape(y, [-1, 784]))
         chainer.report({'nll': nll}, self)
         return nll

