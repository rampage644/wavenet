'''Server for visualization.'''
#%%
from __future__ import (absolute_import, print_function, unicode_literals, division)

import json
import itertools
import numpy as np

from bokeh.layouts import column, gridplot
from bokeh.models import Button
from bokeh.palettes import Set1
from bokeh.plotting import figure, curdoc, output_notebook, show

#%%
def _keys_for(link):
    data = ['W/data', 'W/grad', 'b/data', 'b/grad']
    weights_biases = ['{}/{}/{}'.format(link, entry, key) for entry in data for key in SUFFIX_KEYS]
    return ['{}/W-b/data/zeros'.format(link)] + weights_biases


def plot_stats(p, ds, key):
    def fill_between(x, y_lower, y_upper, alpha):
        return p.patch(np.append(x, x[::-1]), np.append(y_lower, y_upper), alpha=alpha)

    def get(suffix):
        return ds.get(key + '/' + suffix, np.empty((1, 2)))

    xt = get('mean')[:, 0]
    y0 = fill_between(xt, get('min')[:, 1], get('max')[:, 1], 0.1)
    y1 = fill_between(xt, get('percentile/0')[:, 1], get('percentile/6')[:, 1], 0.1)
    y2 = fill_between(xt, get('percentile/1')[:, 1], get('percentile/5')[:, 1], 0.1)
    y3 = fill_between(xt, get('percentile/2')[:, 1], get('percentile/4')[:, 1], 0.2)
    y4 = p.line(xt, get('percentile/3')[:, 1])

    return [y0, y1, y2, y3, y4]


FILENAME = 'log'
COLORS = Set1[3][:2] * 2
ALPHAS = [.4, .4, 1., 1.]
TRAIN_KEY, TEST_KEY = 'main/nll', 'validation/main/nll'
TIME_KEY = 'iteration'
WINDOW_SIZE = 10
PREFIX_KEYS = [
    'predictor'
]
SUFFIX_KEYS = ['max', 'mean', 'min', 'percentile/0', 'percentile/1', 'percentile/2',
               'percentile/3', 'percentile/4', 'percentile/5', 'percentile/6', 'std']
DATA_KEYS = [TRAIN_KEY, TEST_KEY] + list(itertools.chain(*[_keys_for(link) for link in PREFIX_KEYS]))

#%%
loss_plot = figure(plot_width=1000, plot_height=800)
loss = loss_plot.multi_line(xs=[[]] * 4,
                            ys=[[]] * 4,
                            color=COLORS,
                            alpha=ALPHAS)

plots = []
dataseries = {}
source = {}


for prefix in PREFIX_KEYS:
    for key in ['W/data', 'W/grad', 'b/data', 'b/grad']:
        complex_key = prefix+'/'+key
        p = figure(title=complex_key)
        source[complex_key] = plot_stats(p, dataseries, complex_key)
        plots.append(p)
    for key in ['W-b/data/zeros']:
        def get(key):
            return dataseries.get(key, np.empty((1, 2)))
        complex_key = prefix+'/'+key
        p = figure(title=complex_key)
        source[complex_key] = p.line(get(complex_key)[:, 0], get(complex_key)[:, 1])
        plots.append(p)


grid = gridplot(plots, plot_width=250, plot_height=250, ncols=5)


def callback():
    with open(FILENAME) as ifile:
        data = json.load(ifile)
        for key in DATA_KEYS:
            dataseries[key] = np.array([
                (rcrd[TIME_KEY], rcrd[key]) for rcrd in data if key in rcrd], 'f')


    window = np.hamming(WINDOW_SIZE)
    window /= window.sum()

    def smooth(data):
        return np.concatenate([data[:WINDOW_SIZE-1], np.convolve(data, window, mode='valid')])

    train_ts = dataseries['main/nll']
    test_ts = dataseries['validation/main/nll']

    train_smooth = smooth(train_ts[:, 1])
    test_smooth = smooth(test_ts[:, 1])

    loss.data_source.data.update({
        'xs': [train_ts[:, 0], test_ts[:, 0]] * 2,
        'ys': [train_ts[:, 1], test_ts[:, 1], train_smooth, test_smooth],
    })


# # add a button widget and configure with the call back
button = Button(label="Update")
button.on_click(callback)


# output_notebook()
layout = column(button, loss_plot, grid)
curdoc().add_root(layout)

# show(layout)
