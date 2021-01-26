import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import torch

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

from config import *


# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(data):
    """Data Loader"""
    data_dir = os.path.join(config.combined_path, data)

    data = pd.read_csv(data_dir,
                       # infer_datetime_format=True,
                       parse_dates=['timeStamp']
                       )

    data.index = data['timeStamp']
    data = data.drop('timeStamp', axis=1)

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


def scaling_window(data, seq_length):
    """Scaling Window Function : Wrapping Data Sequentially"""
    x, y = list(), list()

    for i in range(len(data) - seq_length - 1):
        data_x = data[i:(i + seq_length)]
        data_y = data[i + seq_length]
        x.append(data_x)
        y.append(data_y)

    x, y = np.array(x), np.array(y)

    return x, y


def get_time_series_data_(data, valid_start, test_start, feature, T, print_ratio=False):
    """Time Series Data Loader"""

    # Train set
    train = data.copy()[data.index < valid_start][[feature]]
    train_shifted = train.copy()
    train_shifted['{}_t+1'.format(feature)] = train_shifted[feature].shift(-1)

    for t in range(1, T + 1):
        train_shifted['{}_t-'.format(feature) + str(T - t)] = train_shifted[feature].shift(T - t)

    train_shifted = train_shifted.rename(columns={feature: '{}_original'.format(feature)})
    train_shifted = train_shifted.dropna(how='any')

    X_train = train_shifted[['{}_t-'.format(feature)+str(T-t) for t in range(1, T+1)]]
    y_train = train_shifted[['{}_t+1'.format(feature)]]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    X_train = X_train[..., np.newaxis]

    # Valid set
    look_back_dt = datetime.strptime(valid_start, '%Y-%m-%d %H:%M:%S') - timedelta(seconds=T - 1)

    data.index = pd.to_datetime(data.index)

    valid = data.copy()[(data.index >= look_back_dt) & (data.index < test_start)][[feature]]

    valid_shifted = valid.copy()
    valid_shifted['{}+1'.format(feature)] = valid_shifted[feature].shift(-1)
    for t in range(1, T + 1):
        valid_shifted['{}_t-'.format(feature) + str(T - t)] = valid_shifted[feature].shift(T - t)

    valid_shifted = valid_shifted.dropna(how='any')

    X_valid = valid_shifted[['{}_t-'.format(feature) + str(T - t) for t in range(1, T + 1)]]
    X_valid = X_valid.to_numpy()
    X_valid = X_valid[..., np.newaxis]

    y_valid = valid_shifted['{}+1'.format(feature)]
    y_valid = y_valid.to_numpy()

    # Test set
    test = data.copy()[data.index >= test_start][[feature]]

    test_shifted = test.copy()
    test_shifted['{}_t+1'.format(feature)] = test_shifted[feature].shift(-1)

    for t in range(1, T + 1):
        test_shifted['{}_t-'.format(feature) + str(T - t)] = test_shifted[feature].shift(T - t)

    test_shifted = test_shifted.dropna(how='any')

    X_test = test_shifted[['{}_t-'.format(feature) + str(T - t) for t in range(1, T + 1)]].to_numpy()
    X_test = X_test[..., np.newaxis]

    y_test = test_shifted['{}_t+1'.format(feature)].to_numpy()

    # Convert to Torch
    X_train = torch.from_numpy(X_train).to(device, dtype=torch.float32)
    y_train = torch.from_numpy(y_train).to(device, dtype=torch.float32)

    X_valid = torch.from_numpy(X_valid).to(device, dtype=torch.float32)
    y_valid = torch.from_numpy(y_valid).to(device, dtype=torch.float32)

    X_test = torch.from_numpy(X_test).to(device, dtype=torch.float32)
    y_test = torch.from_numpy(y_test).to(device, dtype=torch.float32)

    if print_ratio:
        total = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]
        X_train_ratio = X_train.shape[0] / total
        X_valid_ratio = X_valid.shape[0] / total
        X_test_ratio = X_test.shape[0] / total

        print(X_train_ratio, X_valid_ratio, X_test_ratio)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, test_shifted


def plot_test_set(path, id, network, scaler, predictions, y_test, test_shifted, transfer_learning=False):
    """Plot Test set of Drive Cycle of Specific Device ID"""
    HORIZON = 1
    test = pd.DataFrame(predictions, columns=['t+' + str(t) for t in range(1, HORIZON + 1)])
    test['timeStamp'] = test_shifted.index

    test = pd.melt(test, id_vars='timeStamp', value_name='Prediction', var_name='h')
    test['Actual'] = np.transpose(y_test.cpu())

    test[['Prediction', 'Actual']] = scaler.inverse_transform(test[['Prediction', 'Actual']])
    test[test.timeStamp <= '2017-12-31'].plot(x='timeStamp', y=['Prediction', 'Actual'], style=['r', 'b'], figsize=(16, 8))

    # test[test.timeStamp <= '2017-12-23 06:55:00'].plot(x='timeStamp', y=['Prediction', 'Actual'], style=['r', 'b'], figsize=(16, 8))

    plt.xlabel('DateTime', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylabel('Speed', fontsize=12)
    plt.grid()
    plt.legend(fontsize=16)

    if transfer_learning:
        plt.title('Drive Cycle of Device ID {} Prediction using Pre-trained {}'.format(id, network), fontsize=18)
        plt.savefig(os.path.join(path, 'Drive_Cycle_Device_ID_{}_Test_{}_Transfer_Detailed.png'.format(id, network)))
        plt.show()
    else:
        plt.title('Drive Cycle of Device ID {} Prediction using {}'.format(id, network), fontsize=18)
        plt.savefig(os.path.join(path, 'Drive_Cycle_Device_ID_{}_Test_{}.png'.format(id, network)))

    return test['Prediction'], test['Actual']


def get_time_series_data(x, y):
    """Time Series Data Loader"""
    train_size = int(len(y) * config.train_split)
    val_size = int(len(y) * config.test_split)

    train_X = torch.from_numpy(x[0:train_size]).float().to(device)
    train_Y = torch.from_numpy(y[0:train_size]).float().to(device)

    val_X = torch.from_numpy(x[train_size:train_size + val_size]).float().to(device)
    val_Y = torch.from_numpy(y[train_size:train_size + val_size]).float().to(device)

    test_X = torch.from_numpy(x[train_size + val_size:len(x)]).float().to(device)
    test_Y = torch.from_numpy(y[train_size + val_size:len(y)]).float().to(device)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def percentage_error(actual, predicted):
    """Percentage Error"""
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def RMSE(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(y_true, y_pred))


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


def MPE(y_true, y_pred):
    """Mean Percentage Error"""
    return np.mean(percentage_error(np.asarray(y_true), np.asarray(y_pred))) * 100


def get_lr_scheduler(optimizer):
    """Learning Rate Scheduler"""
    if config.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay_rate)
    elif config.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0)
    else:
        raise NotImplementedError
    return scheduler

