'''Server for visualization.'''
from __future__ import (absolute_import, print_function, unicode_literals, division)

import json
from random import random
import numpy as np

from bokeh.layouts import column
from bokeh.models import Button
from bokeh.palettes import Set1
from bokeh.plotting import figure, curdoc

FILENAME = 'log'

# create a plot and style its properties
p = figure(plot_width=1200, plot_height=800)
COLORS = Set1[3][:2] * 2
ALPHAS = [.4, .4, 1., 1.]
loss = p.multi_line(xs=[[]] * 4,
                    ys=[[]] * 4,
                    color=COLORS,
                    alpha=ALPHAS)

# # create a callback that will add a number in a random location
TRAIN_KEY, TEST_KEY = 'main/nll', 'validation/main/nll'
WINDOW_SIZE = 10


def callback():
    with open(FILENAME) as ifile:
        data = json.load(ifile)
        train_ts = np.array([(record['iteration'], record[TRAIN_KEY]) for record in data], 'f')
        test_ts = np.array([(record['iteration'], record[TEST_KEY]) for record in data if TEST_KEY in record], 'f')

    window = np.hamming(WINDOW_SIZE)
    window /= window.sum()
    def smooth(data):
        return np.concatenate([data[:WINDOW_SIZE-1], np.convolve(data, window, mode='valid')])

    train_smooth = smooth(train_ts[:, 1])
    test_smooth = smooth(test_ts[:, 1])

    loss.data_source.data.update({
        'xs': [train_ts[:, 0], test_ts[:, 0]] * 2,
        'ys': [train_ts[:, 1], test_ts[:, 1], train_smooth, test_smooth],
    })


# # add a button widget and configure with the call back
button = Button(label="Update")
button.on_click(callback)

# put the button and plot in a layout and add to the document
curdoc().add_root(column(button, p))
