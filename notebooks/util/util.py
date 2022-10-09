import pandas as pd
import json
import os
from matplotlib import pyplot as plt
import numpy as np

# Configuration
anomaly_color = 'sandybrown'
prediction_color = 'yellowgreen'
training_color = 'yellowgreen'
validation_color = 'gold'
test_color = 'coral'

def load_series(file_name, data_folder):
    # Load the input data
    data_path = os.path.join(data_folder, 'data', file_name)
    # data_path = f'{data_folder}/data/{file_name}'
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    # Load the labels
    label_path = os.path.join(data_folder, 'labels', 'combined_labels.json')
    # label_path = f'{data_folder}/labels/combined_labels.json'
    with open(label_path) as fp:
        labels = pd.Series(json.load(fp)[file_name])
    labels = pd.to_datetime(labels)
    # Load the windows
    window_path = os.path.join(data_folder, 'labels', 'combined_windows.json')
    # window_path = f'{data_folder}/labels/combined_windows.json'
    window_cols = ['begin', 'end']
    with open(window_path) as fp:
        windows = pd.DataFrame(columns=window_cols,
                data=json.load(fp)[file_name])
    windows['begin'] = pd.to_datetime(windows['begin'])
    windows['end'] = pd.to_datetime(windows['end'])
    # Return data
    return data, labels, windows


def plot_series(data, labels=None,
                    windows=None,
                    predictions=None,
                    highlights=None,
                    val_start=None,
                    test_start=None,
                    figsize=None,
                    show_sampling_points=False,
                    show_markers=False,
                    filled_version=None):
    # Open a new figure
    plt.figure(figsize=figsize)
    # Plot data
    if not show_markers:
        plt.plot(data.index, data.values, zorder=0)
    else:
        plt.plot(data.index, data.values, zorder=0,
                marker='.', markersize=3)
    if filled_version is not None:
        filled = filled_version.copy()
        filled[~data['value'].isnull()] = np.nan
        plt.scatter(filled.index, filled,
                marker='.', c='tab:orange', s=5);
    if show_sampling_points:
        vmin = data.min()
        lvl = np.full(len(data.index), vmin)
        plt.scatter(data.index, lvl, marker='.',
                c='tab:red', s=5)
    # Rotated x ticks
    plt.xticks(rotation=45)
    # Plot labels
    if labels is not None:
        plt.scatter(labels.values, data.loc[labels],
                    color=anomaly_color, zorder=2, s=5)
    # Plot windows
    if windows is not None:
        for _, wdw in windows.iterrows():
            plt.axvspan(wdw['begin'], wdw['end'],
                        color=anomaly_color, alpha=0.3, zorder=1)
    # Plot training data
    if val_start is not None:
        plt.axvspan(data.index[0], val_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is None and test_start is not None:
        plt.axvspan(data.index[0], test_start,
                    color=training_color, alpha=0.1, zorder=-1)
    if val_start is not None:
        plt.axvspan(val_start, test_start,
                    color=validation_color, alpha=0.1, zorder=-1)
    if test_start is not None:
        plt.axvspan(test_start, data.index[-1],
                    color=test_color, alpha=0.3, zorder=0)
    # Predictions
    if predictions is not None:
        plt.scatter(predictions.values, data.loc[predictions],
                    color=prediction_color, alpha=.4, zorder=3,
                    s=5)
    plt.tight_layout()


def plot_gp(target=None, pred=None, std=None, samples=None,
        target_samples=None, figsize=None):
    plt.figure(figsize=figsize)
    if target is not None:
        plt.plot(target.index, target, c='black', label='target')
    if pred is not None:
        plt.plot(pred.index, pred, c='tab:blue',
                label='predictions')
    if std is not None:
        plt.fill_between(pred.index, pred-1.96*std, pred+1.96*std,
                alpha=.3, fc='tab:blue', ec='None',
                label='95% C.I.')
    # Add scatter plots
    if samples is not None:
        try:
            x = samples.index
            y = samples.values
        except AttributeError:
            x = samples[0]
            y = samples[1]
        plt.scatter(x, y, color='tab:orange',
              label='samples', marker='x')
    if target_samples is not None:
        try:
            x = target_samples.index
            y = target_samples.values
        except AttributeError:
            x = target_samples[0]
            y = target_samples[1]
        plt.scatter(x, y,
                color='black', label='target', s=5)
    plt.legend()
    plt.tight_layout()


def plot_distribution_2D(f, xr, yr, figsize=None):
    # Build the input
    nx = len(xr)
    ny = len(yr)
    xc = np.repeat(xr, ny)
    yc = np.tile(yr, nx)
    data = np.vstack((xc, yc)).T
    dvals = np.exp(f.pdf(data))
    dvals = dvals.reshape((nx, ny))
    # Build a new figure
    plt.figure(figsize=figsize)
    plt.pcolor(dvals)
    plt.tight_layout()
    xticks = np.linspace(0, len(xr), 6)
    xlabels = np.linspace(xr[0], xr[-1], 6)
    plt.xticks(xticks, xlabels)
    yticks = np.linspace(0, len(yr), 6)
    ylabels = np.linspace(yr[0], yr[-1], 6)
    plt.yticks(yticks, ylabels)


def plot_bars(data, figsize=None, generate_x=False):
    plt.figure(figsize=figsize)
    if generate_x:
        x = 0.5 + np.arange(len(data))
    else:
        x = data.index
    plt.bar(x, data, width=0.7)
    plt.xticks(x[::10], data.index[::10])
    plt.tight_layout()
