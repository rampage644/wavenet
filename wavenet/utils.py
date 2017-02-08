'''Utilities.'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import itertools
import operator
import os

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

from chainer.dataset.dataset_mixin import DatasetMixin


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


def mulaw(audio, mu=255):
    return np.sign(audio) * np.log1p(mu * audio) / np.log1p(mu)


def wav_to_float(audio, bits=16):
    '''Squash -2 ** 15; 2 ** 15 into [-1, 1] range'''
    return audio / 2 ** (bits-1)


def wav_files_in(dir):
    for path, _, files in os.walk(dir):
        names = [name for name in files if '.wav' in name]
        for name in names:
            yield os.path.join(path, name)


def _preprocess(ifilename, rate, chunk_length):
    baserate, data = wavfile.read(ifilename)
    audio = signal.resample_poly(data, rate, baserate)
    audio = mulaw(wav_to_float(audio))
    while len(audio) >= chunk_length:
        yield audio[:chunk_length]
        audio = audio[chunk_length:]


def nth(iterable, n, default=None):
    "Returns the nth item or a default value (from itertool recipes)"
    return next(itertools.islice(iterable, n, None), default)


class VCTK(DatasetMixin):
    def __init__(self, root_dir, rate, chunk_length):
        self.indices = {}
        self.dir = root_dir
        self.rate = rate
        self.chunk = chunk_length

        self._populate()

    def _populate(self):
        idx = 0
        for wfile in wav_files_in(self.dir):
            for cidx, _ in enumerate(_preprocess(wfile, self.rate, self.chunk)):
                self.indices[idx] = [wfile, cidx]
                idx += 1

    def __len__(self):
        return len(self.indices)

    def get_example(self, i):
        wfile, idx = self.indices[i]
        sample = np.expand_dims(np.expand_dims(
            nth(_preprocess(wfile, self.rate, self.chunk), idx), 0), 0).astype(np.float32)
        return (sample, quantisize(sample, 256), np.array(0))
