'''Utilities.'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np


def binarize(images, xp=np):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (xp.random.uniform(size=images.shape) < images).astype('float32')
