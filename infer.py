'''Train'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import sys
import numpy as np
import scipy.misc

import chainer
import chainer.training
import chainer.training.extensions as extensions
import chainer.functions as F

import wavenet.models as models
import wavenet.utils as utils


def generate_and_save_samples(sample_fn, height, width, channels, count, filename):
    def save_images(images):
        images = images.reshape((count, count, channels, height, width))
        images = images.transpose(1, 3, 0, 4, 2)
        images = images.reshape((height * count, width * count, channels))
        scipy.misc.toimage(images, cmin=0.0, cmax=255.0).save(filename)

    samples = chainer.Variable(
        chainer.cuda.cupy.zeros((count ** 2, channels, height, width), dtype='float32'))

    for i in range(height):
        for j in range(width):
            for k in range(channels):
                probs = F.softmax(sample_fn(samples))[:, :, k, i, j]
                _, level_count = probs.shape
                samples.data[:, k, i, j] = chainer.cuda.to_gpu(utils.sample_from(probs.data.get())) / (level_count - 1)
                print('\r{:.2f}%'.format(100.0 * (i * width + j + 1) / height / width), end='')
    print()

    samples.to_cpu()

    save_images(samples.data * 255.0)

def main():
    parser = argparse.ArgumentParser(description='PixelCNN')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='',
                        help='Path to model for generation')
    parser.add_argument('--hidden_dim', '-d', type=int, default=128,
                        help='Number of hidden dimensions')
    parser.add_argument('--out_hidden_dim', type=int, default=16,
                        help='Number of hidden dimensions')
    parser.add_argument('--blocks_num', '-n', type=int, default=15,
                        help='Number of layers')
    parser.add_argument('--levels', type=int, default=2,
                        help='Level number to quantisize pixel values')
    parser.add_argument('--output', '-o', type=str, default='samples.jpg',
                        help='Output filename')
    parser.add_argument('--label', '-l', type=np.int32, default=0,
                        help='Class label to generate')
    args = parser.parse_args()

    IN_CHANNELS = 3
    # multiply hidden dim by IN_CHANNELS to make sure mask is disible by IN_CHANNELS
    model = models.PixelCNN(IN_CHANNELS, args.hidden_dim, args.blocks_num, args.out_hidden_dim, args.levels)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
    chainer.serializers.load_npz(args.model, model)

    def sample_fn(samples):
        B, C, H, W = samples.shape
        return model(samples, np.ones(B).astype('i') * args.label)

    generate_and_save_samples(sample_fn, 28, 28, IN_CHANNELS, 9, args.output)


if __name__ == '__main__':
    sys.exit(main())
