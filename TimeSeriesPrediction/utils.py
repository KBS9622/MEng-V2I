import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from datetime import datetime, timedelta

import torch
from torch.utils.data import TensorDataset, DataLoader

from config import *

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(which_data, resample):
    """Data Loader"""
    data_dir = os.path.join(config.combined_path, which_data)

    data = pd.read_csv(data_dir,
                       # infer_datetime_format=True,
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

    data.drop(cols_to_drop, axis=1, inplace=True)
    data = data.set_index('timeStamp')

    if resample:
        data = data.resample("1S").first().ffill()
        # data.to_csv(os.path.join(config.combined_path, 'resampled_' + which_data))
        data.loc['2017-12-23 12:00:00':'2017-12-23 15:00:00', 'speed'] = 0

    print(data['2017-12-23 12:00:00':'2017-12-23 15:00:00'].sum())

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
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)

    X_valid = torch.from_numpy(X_valid)
    y_valid = torch.from_numpy(y_valid)

    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    if print_ratio:
        total = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]
        X_train_ratio = X_train.shape[0] / total
        X_valid_ratio = X_valid.shape[0] / total
        X_test_ratio = X_test.shape[0] / total

        print(X_train_ratio, X_valid_ratio, X_test_ratio)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, test_shifted


def get_data_loader(X_train, y_train, X_valid, y_valid, batch_size, val_batch_size):
    """Get Data Loader"""

    # Wrap for Data Loader #
    train_set = TensorDataset(X_train, y_train)
    val_set = TensorDataset(X_valid, y_valid)

    # Prepare Data Loader #
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def test(path, id, network, scaler, predictions, y_test, test_shifted, transfer_learning=False):
    """Plot Test set of Drive Cycle of Specific Device ID"""

    HORIZON = 1
    test = pd.DataFrame(predictions, columns=['t+' + str(t) for t in range(1, HORIZON + 1)])
    test['timeStamp'] = test_shifted.index

    test = pd.melt(test, id_vars='timeStamp', value_name='Prediction', var_name='h')
    test['Actual'] = np.transpose(y_test.cpu())

    test[['Prediction', 'Actual']] = scaler.inverse_transform(test[['Prediction', 'Actual']])
    test[test.timeStamp <= '2017-12-23'].plot(x='timeStamp', y=['Prediction', 'Actual'], style=['r', 'b'], figsize=(16, 8))

    pred_test, label = test['Prediction'], test['Actual']

    # Calculate Loss #
    test_mae = mean_absolute_error(label, pred_test)
    test_mse = mean_squared_error(label, pred_test, squared=True)
    test_rmse = mean_squared_error(label, pred_test, squared=False)
    test_mpe = mean_percentage_error(label, pred_test)
    test_mape = mean_absolute_percentage_error(label, pred_test)
    test_r2 = r2_score(label, pred_test)

    # Print Statistics #
    print("Test {}".format(config.network.__class__.__name__))
    print("Test  MAE : {:.4f}".format(test_mae))
    print("Test  MSE : {:.4f}".format(test_mse))
    print("Test RMSE : {:.4f}".format(test_rmse))
    print("Test  MPE : {:.4f}".format(test_mpe))
    print("Test MAPE : {:.4f}".format(test_mape))
    print("Test  R^2 : {:.4f}".format(test_r2))

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


def get_lr_scheduler(lr_scheduler, optimizer, config):
    """Learning Rate Scheduler"""
    if lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=config.lr_decay_rate)
    elif lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=0)
    else:
        raise NotImplementedError

    return scheduler

