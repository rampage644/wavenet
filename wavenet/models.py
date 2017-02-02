'''Models.'''
#%%
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

import wavenet.utils as utils


class MaskedConvolution2D(L.Convolution2D):
    def __init__(self, *args, mask='B', **kwargs):
        super(MaskedConvolution2D, self).__init__(
            *args, **kwargs
        )

        Cout, Cin, kh, kw = self.W.shape
        pre_mask = self.xp.ones_like(self.W.data).astype('f')
        yc, xc = kh // 2, kw // 2

        # context masking - subsequent pixels won't hav access to next pixels (spatial dim)
        pre_mask[:, :, yc+1:, :] = 0.0
        pre_mask[:, :, yc:, xc+1:] = 0.0

        # same pixel masking - pixel won't access next color (conv filter dim)
        # TODO: Implement proper channel masking
        pre_mask[:, :, yc, xc] = 0.0 if mask == 'A' else 1.0

        self.mask = pre_mask

    def __call__(self, x):
        if self.has_uninitialized_params:
            with chainer.cuda.get_device(self._device_id):
                self._initialize_params(x.shape[1])

        return chainer.functions.connection.convolution_2d.convolution_2d(
            x, self.W * self.mask, self.b, self.stride, self.pad, self.use_cudnn,
            deterministic=self.deterministic)

    def to_gpu(self, device=None):
        self._persistent.append('mask')
        res = super().to_gpu(device)
        self._persistent.remove('mask')
        return res


class ResidualBlock(chainer.Chain):
    def __init__(self, in_channels, nobias=False):
        super(ResidualBlock, self).__init__(
            conv1=MaskedConvolution2D(in_channels, in_channels // 2, 1, nobias=nobias),
            conv2=MaskedConvolution2D(in_channels // 2, in_channels // 2, 3, pad=1, nobias=nobias),
            conv3=MaskedConvolution2D(in_channels // 2, in_channels, 1, nobias=nobias)
        )

    def __call__(self, x):
        h = self.conv1(F.relu(x))
        h = self.conv2(F.relu(h))
        h = self.conv3(F.relu(h))

        return F.relu(x + h)


class ResidualBlockList(chainer.ChainList):
    def __init__(self, block_num, hidden_dims, mask='B', nobias=False):
        blocks = [ResidualBlock(hidden_dims, nobias=nobias) for _ in range(block_num)]
        super(ResidualBlockList, self).__init__(*blocks)

    def __call__(self, x):
        h = x
        for block in self:
            h = block(h)
        return h


class PixelCNN(chainer.Chain):
    def __init__(self, in_channels, hidden_dims, block_num, out_hidden_dims, out_dims, nobias=False):
        super(PixelCNN, self).__init__(
            conv1=MaskedConvolution2D(in_channels, hidden_dims, 7, pad=3, mask='A', nobias=nobias),
            blocks=ResidualBlockList(block_num, hidden_dims, nobias=nobias),
            conv2=MaskedConvolution2D(hidden_dims, out_hidden_dims, 1, nobias=nobias),
            conv4=MaskedConvolution2D(out_hidden_dims, out_dims * in_channels, 1, nobias=nobias)
        )
        self.in_channels = in_channels
        self.out_dims = out_dims

    def __call__(self, x):
        h = self.conv1(x)
        h = self.blocks(h)
        h = self.conv2(F.relu(h))
        h = self.conv4(F.relu(h))

        batch_size, _, height, width = h.shape
        h = F.reshape(h, [batch_size, self.in_channels, self.out_dims, height, width])
        h = F.transpose(h, [0, 2, 1, 3, 4])

        return h


# TODO: rename class
class Classifier(chainer.Chain):
     def __init__(self, predictor):
         super(Classifier, self).__init__(predictor=predictor)

     def __call__(self, x, t):
         y = self.predictor(x)

        #  nll = F.sigmoid_cross_entropy(y, t, normalize=False)
         nll = F.softmax_cross_entropy(y, t, normalize=False)
         chainer.report({'nll': nll}, self)
         return nll
