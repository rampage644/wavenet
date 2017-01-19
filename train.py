'''Train'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
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
    parser.add_argument('--hidden_dim', '-d', type=int, default=128,
                        help='Number of hidden dimensions')
    parser.add_argument('--out_hidden_dim', type=int, default=16,
                        help='Number of hidden dimensions')
    parser.add_argument('--blocks_num', '-n', type=int, default=15,
                        help='Number of layers')
    parser.add_argument('--gradclip', type=float, default=1.0,
                        help='Bound for gradient hard clipping')
    args = parser.parse_args()

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    # XXX: 1 is for mnist dataset, 1 dimension - grayscale
    # XXX: double hidden dims as stated in paper
    model = models.Classifier(models.PixelCNN(1, args.hidden_dim, args.blocks_num, args.out_hidden_dim))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientHardClipping(-args.gradclip, args.gradclip))

    train, test = chainer.datasets.get_mnist(ndim=3, withlabel=False)
    train, test = utils.binarize(train), utils.binarize(test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/nll'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/nll', 'validation/main/nll', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    chainer.serializers.save_npz('pixelcnn', model.predictor)


if __name__ == '__main__':
    sys.exit(main())
