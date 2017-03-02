'''Dataset preprocessing.'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import concurrent.futures
import os
import re

import numpy as np

import wavenet.utils as utils

BATCH = 10240
RATE = 8000
CHUNK = 1600

def guess_label_from(filename):
    numbers = re.findall(r'(\d+)', filename)
    return int(numbers[0]) if len(numbers) else -1


def split_into(data, n):
    res = []
    for i in range(n):
        res.append(data[i::n])
    return res


def process_files(files, id, output, rate, size, chunk_length, batch):
    data = []
    labels = []
    data_filename = os.path.join(output, 'vctk_{}'.format(id))
    label_filename = os.path.join(output, 'vctk_{}_label'.format(id))
    with open(data_filename, 'wb') as dfile, open(label_filename, 'wb') as lfile:
        for filename in files:
            label = guess_label_from(filename)

            def _flush():
                np.save(dfile, np.array(data))
                np.save(lfile, np.array(labels))
                data.clear()
                labels.clear()

            for chunk in utils._preprocess(filename, rate, size, chunk_length):
                data.append(chunk)
                labels.append(label)

            if len(data) >= batch:
                _flush()
            _flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.getcwd())
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--rate', type=int, default=RATE)
    parser.add_argument('--stacks_num', type=int, default=5)
    parser.add_argument('--layers_num', type=int, default=10)
    parser.add_argument('--target_length', type=int, default=CHUNK)
    parser.add_argument('--flush_every', type=int, default=BATCH)
    args = parser.parse_args()

    files = list(utils.wav_files_in(args.data))
    file_groups = split_into(files, args.workers)

    size = utils.receptive_field_size(args.layers_num, args.stacks_num)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i in range(args.workers):
            pool.submit(process_files, file_groups[i], i, args.output, args.rate,
                        size, args.target_length + size, args.flush_every)


if __name__ == '__main__':
    main()
