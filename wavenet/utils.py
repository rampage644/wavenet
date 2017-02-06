'''Utilities.'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import operator


def binarize(images, xp=np):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (xp.random.uniform(size=images.shape) < images).astype('i')


def quantisize(images, levels):
    return (np.digitize(images, np.arange(levels) / levels) - 1).astype('i')


def convert_to_rgb(images, xp=np):
    return xp.tile(images, [1, 3, 1, 1])


def sample_from(distribution):
    batch_size, bins = distribution.shape
    return np.array([np.random.choice(bins, p=distr) for distr in distribution])


def extract_labels(data):
    return np.fromiter(map(operator.itemgetter(1), data), dtype='i')


def extract_images(data):
    return np.array(list(map(operator.itemgetter(0), data))).astype('f')
