import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def classification_subplot(ax, data, save_path, target_column, title_type='train_'):
    target_value = data.groupby(target_column).size().to_dict()
    target_level = list(target_value.keys())
    ax.bar(target_level, list(target_value.values()), width=0.3, color=[0.012, 0.5, 0.5], edgecolor='k')
    ax.set_title(title_type+'_' + Path(save_path).stem)
    ax.set_xlabel(target_column)
    ax.set_ylabel('count')

def regression_subplot(ax, data, save_path, target_column, title_type='train_'):
    ax.hist(data[target_column], bins=20, color=[0.8, 0.5, 0.5], edgecolor='k')
    ax.set_title(title_type + Path(save_path).stem)
    ax.set_xlabel(target_column)
    ax.set_ylabel('count')


def plot_data(data, target_column, dataset_type, save_path):
    """
    :param data: pd.DataFrame includes two columns['smlies', target_column].
    :param target_column: Name of the columns containing target values.
    :param dataset_type: type of  dataset, in ['regression', 'classification', 'multiclass']
    :param save_path: .jpg file save path
    """
    fig = plt.figure(figsize=(7, 5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.style.use('ggplot')
    if dataset_type == 'regression':
        regression_subplot(fig.add_subplot(1, 1, 1), data, save_path.replace('.jpg', '_all_data.jpg'), target_column, '')
    else:
        classification_subplot(fig.add_subplot(1, 1, 1), data, save_path.replace('.jpg', '_all_data.jpg'), target_column, '')

    if save_path:
        fig.savefig(save_path.replace('.jpg', '_all_data.jpg'))

    fig.show()

