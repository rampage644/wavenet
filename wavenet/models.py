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
        def bmask(i_out, i_in):
            cout_idx = np.expand_dims(np.arange(Cout) % 3 == i_out, 1)
            cin_idx = np.expand_dims(np.arange(Cin) % 3 == i_in, 0)
            a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            return a1 * a2

        for j in range(3):
            pre_mask[bmask(j, j), yc, xc] = 0.0 if mask == 'A' else 1.0

        pre_mask[bmask(0, 1), yc, xc] = 0.0
        pre_mask[bmask(0, 2), yc, xc] = 0.0
        pre_mask[bmask(1, 2), yc, xc] = 0.0

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


class CroppedConvolution(L.Convolution2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def __call__(self, x):
        ret = super().__call__(x)
        kh, kw = self.ksize
        pad_h, pad_w = self.pad
        h_crop = -(kh + 1) if pad_h == kh else None
        w_crop = -(kw + 1) if pad_w == kw else None
        return ret[:, :, :h_crop, :w_crop]


class ResidualBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, filter_size, mask='B', nobias=False):
        super(ResidualBlock, self).__init__(
            vertical_conv=CroppedConvolution(
                in_channels, 2 * out_channels, ksize=[filter_size//2+1, filter_size],
                pad=[filter_size//2+1, filter_size//2]),
            v_to_h_conv=MaskedConvolution2D(2 * out_channels, 2 * out_channels, 1, mask=mask),
            vertical_gate_conv=L.Convolution2D(2*out_channels, 2*out_channels, 1),
            horizontal_conv=CroppedConvolution(
                in_channels, 2 * out_channels, ksize=[1, filter_size//2+1],
                pad=[0, filter_size//2+1]),
            horizontal_gate_conv=L.Convolution2D(2*out_channels, 2*out_channels, 1),
            horizontal_output=MaskedConvolution2D(out_channels, out_channels, 1, mask=mask)
        )

    def _crop(self, x, ksize):
        kh, kw = ksize
        return x[:, :, ]

    def __call__(self, v, h):
        v = self.vertical_conv(v)
        to_vertical = self.v_to_h_conv(v)

        v_t, v_s = F.split_axis(self.vertical_gate_conv(v), 2, axis=1)
        v = F.tanh(v_t) * F.sigmoid(v_s)

        h_ = self.horizontal_conv(h)
        h_t, h_s = F.split_axis(self.horizontal_gate_conv(h_ + to_vertical), 2, axis=1)
        h = self.horizontal_output(F.tanh(h_t) * F.sigmoid(h_s))

        return v, h


class ResidualBlockList(chainer.ChainList):
    def __init__(self, block_num, *args, **kwargs):
        blocks = [ResidualBlock(*args, **kwargs) for _ in range(block_num)]
        super(ResidualBlockList, self).__init__(*blocks)

    def __call__(self, v, h):
        for block in self:
            v_, h_ = block(v, h)
            v, h = v_, h + h_
        return v, h


class PixelCNN(chainer.Chain):
    def __init__(self, in_channels, hidden_dims, block_num, out_hidden_dims, out_dims, nobias=False):
        super(PixelCNN, self).__init__(
            conv1=ResidualBlock(in_channels, hidden_dims, 7, mask='A', nobias=nobias),
            blocks=ResidualBlockList(block_num, hidden_dims, hidden_dims, 3, nobias=nobias),
            conv2=MaskedConvolution2D(hidden_dims, out_hidden_dims, 1, nobias=nobias),
            conv4=MaskedConvolution2D(out_hidden_dims, out_dims * in_channels, 1, nobias=nobias)
        )
        self.in_channels = in_channels
        self.out_dims = out_dims

    def __call__(self, x):
        v, h = self.conv1(x, x)
        # XXX: Consider doing something with vertical stack output as well
        _, h = self.blocks(v, h)
        h = self.conv2(F.relu(h))
        h = self.conv4(F.relu(h))

        batch_size, _, height, width = h.shape
        h = F.reshape(h, [batch_size, self.out_dims, self.in_channels, height, width])

        return h


# TODO: rename class
class Classifier(chainer.Chain):
     def __init__(self, predictor):
         super(Classifier, self).__init__(predictor=predictor)

     def __call__(self, x, t):
         y = self.predictor(x)

         nll = F.softmax_cross_entropy(y, t, normalize=False)
         chainer.report({'nll': nll}, self)
         return nll
