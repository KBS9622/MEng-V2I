import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data(path, which_data, preprocess, resample):
    """Load, Preprocess and Resample the CSV file"""
    data_dir = os.path.join(path, which_data)

    data = pd.read_csv(data_dir,
                       parse_dates=['timeStamp']
                       )

    data['timeStamp'] = pd.to_datetime(data['timeStamp'])

    cols_to_drop = ['Unnamed: 0',
                    'tripID',
                    'deviceID',
                    'accData',
                    'battery',
                    'cTemp',
                    'dtc',
                    'eLoad',
                    'iat',
                    'imap',
                    'kpl',
                    'maf',
                    'rpm',
                    'tAdv',
                    'tPos',
                    'fuel',
                    'gps_speed'
                    ]

    if preprocess:
        outlier_index = data.loc[data['speed'] > 200, :].index

        for x in outlier_index:
            data.loc[x, 'speed'] = data.loc[x - 1, 'speed']

    data.drop(cols_to_drop, axis=1, inplace=True)
    data = data.set_index('timeStamp')

    if resample:
        data = data.resample("1S").first().ffill()

    data['date'] = data.index.copy()

    data.drop(['date'], axis=1, inplace=True)

    print(data)

    return data


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def plot_full(path, data, id, feature):
    """Plot Full Graph of Drive Cycle of Specific Device ID"""
    data.plot(y=feature, figsize=(16, 8))
    plt.xlabel('DateTime', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylabel('Speed', fontsize=10)
    plt.grid()
    plt.title('Drive Cycle of Device ID {}'.format(id))
    plt.savefig(os.path.join(path, 'Drive_Cycle_Device_ID_{}.png'.format(id)))


def plot_split(path, data, id, valid_start, test_start, feature):
    """Plot Splitted Graph of Drive Cycle of Specific Device ID"""
    data[data.index < valid_start][[feature]].rename(columns={feature: 'train'}) \
        .join(data[(data.index >= valid_start) & (data.index < test_start)][[feature]] \
              .rename(columns={feature: 'validation'}), how='outer') \
        .join(data[data.index >= test_start][[feature]].rename(columns={feature: 'test'}), how='outer') \
        .plot(y=['train', 'validation', 'test'], figsize=(16, 8), fontsize=15)

    plt.xlabel('DateTime', fontsize=10)
    plt.xticks(rotation=45)
    plt.ylabel('Speed', fontsize=10)
    plt.grid()
    plt.title('Drive Cycle of Device ID {} Splitted'.format(id))
    plt.savefig(os.path.join(path, 'Drive_Cycle_Device_ID_{}_Splitted.png'.format(id)))


def split_sequence_uni_step(sequence, n_steps):
    """Rolling Window Function for Uni-step"""

    sequence = sequence.values

    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps

        if end_ix > len(sequence)-1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def split_sequence_multi_step(sequence, n_steps_in, n_steps_out):
    """Rolling Window Function for Multi-step"""

    sequence = sequence.values

    X, y = list(), list()

    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        if out_end_ix > len(sequence):
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)[:, :, 0]


def get_data_loader(X, y, train_split, test_split, batch_size):
    """Get Data Loader"""

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_split, shuffle=False)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, train_size=test_split, shuffle=False)

    # Wrap for Data Loader #
    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_set = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    # Prepare Data Loader #
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader


def test_plot(pred, actual, path, feature, id, network, transfer_learning=False):
    """Plot Test set of Drive Cycle of Specific Device ID"""

    plt.figure(figsize=(10, 8))
    plt.plot(pred, label='Pred')
    plt.plot(actual, label='Actual')

    plt.xlabel('DateTime', fontsize=12)
    plt.xticks()
    plt.ylabel(feature, fontsize=12)
    plt.grid()
    plt.legend(fontsize=16)

    if transfer_learning:
        plt.title('Drive Cycle of Device ID {} Prediction using Pre-trained {}'.format(id, network.__class__.__name__), fontsize=18)
        plt.savefig(os.path.join(path, 'Drive_Cycle_Device_ID_{}_Test_{}_Transfer_Detailed.png'.format(id, network.__class__.__name__)))
        plt.show()
    else:
        plt.title('Drive Cycle of Device ID {} Prediction using {}'.format(id, network.__class__.__name__), fontsize=18)
        plt.savefig(os.path.join(path, 'Drive_Cycle_Device_ID_{}_Test_{}.png'.format(id, network.__class__.__name__)))
        plt.show()


def percentage_error(actual, predicted):
    """Percentage Error"""
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_percentage_error(y_true, y_pred):
    """Mean Percentage Error"""
    mpe = np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred))) * 100
    return mpe


def mean_absolute_percentage_error(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mape = np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100
    return mape


def get_lr_scheduler(optimizer, args):
    """Learning Rate Scheduler"""
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    else:
        raise NotImplementedError

    return scheduler