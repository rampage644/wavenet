'''Train'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import os
import sys

import chainer
import chainer.training
import chainer.training.extensions as extensions

import wavenet.models as models
import wavenet.utils as utils


def main():
    parser = argparse.ArgumentParser(description='PixelCNN')
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--out', '-o', default='',
                        help='Output directory')
    parser.add_argument('--data','-d', default=os.getcwd(),
                        help='Input data directory')
    parser.add_argument('--hidden_dim', type=int, default=16,
                        help='Number of hidden dimensions')
    parser.add_argument('--out_hidden_dim', type=int, default=8,
                        help='Number of hidden dimensions')
    parser.add_argument('--stacks_num', '-s', type=int, default=1,
                        help='Number of stacks')
    parser.add_argument('--layers_num', '-l', type=int, default=10,
                        help='Number of layers per stack')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Bound for gradient hard clipping')
    parser.add_argument('--levels', type=int, default=256,
                        help='Level number to quantisize values')
    parser.add_argument('--rate', type=int, default=16000,
                        help='Downsample rate for input files')
    parser.add_argument('--chunk', type=int, default=1024,
                        help='Chunk length: input files would be splitted into chunks')
    parser.add_argument('--stats', type=bool, default=False,
                        help='Collect layerwise statistics')
    args = parser.parse_args()

    model = models.Classifier(
        models.WaveNet(args.levels, args.hidden_dim, args.out_hidden_dim, args.stacks_num,
                       args.layers_num, 2))

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(args.learning_rate)
    optimizer.setup(model)

    train = utils.VCTK(args.data)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    log_trigger = (10, 'iteration')
    trainer.extend(extensions.LogReport(trigger=log_trigger))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'main/nll', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    sys.exit(main())
