import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from models import RNN, LSTM, GRU
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Reproducibility #
cudnn.deterministic = True
cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():

    # Fix Seed for Reproducibility #
    torch.manual_seed(9)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(9)

    # Samples, Weights, and Plots Path #
    paths = [config.weights_path, config.plots_path, config.numpy_path]
    paths = [make_dirs(path) for path in paths]

    # Prepare Data #
    data = load_data(config.which_data)[[config.feature]]
    id = config.which_data.split('_')[0]

    # Plot Time-series Data #
    plot_full(config.plots_path, data, id, config.feature)
    plot_split(config.plots_path, data, id, config.valid_start, config.test_start, config.feature)

    # Min-Max Scaler #
    scaler = MinMaxScaler()
    data[config.feature] = scaler.fit_transform(data)

    # Split the Dataset #
    train_X, train_Y, val_X, val_Y, test_X, test_Y, test_shifted = \
        get_time_series_data_(data, config.valid_start, config.test_start, config.feature, config.window)

    # Lists #
    train_losses, val_losses = list(), list()

    # Prepare Network #
    if config.network == 'rnn':
        model = RNN(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(device)
    elif config.network == 'lstm':
        model = LSTM(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(device)
    elif config.network == 'gru':
        model = GRU(config.input_size, config.hidden_size, config.num_layers, config.num_classes).to(device)

    if config.mode == 'train':

        if config.transfer_learning:
            model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}_Device_ID_12.pkl'.format(config.network))))

            for param in model.parameters():
                param.requires_grad = True

        # Loss Function #
        criterion = torch.nn.MSELoss()

        # Optimizer #
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        optimizer_scheduler = get_lr_scheduler(optimizer)

        # Train #
        print("Training {} started with total epoch of {} using Driver ID of {}.".format(config.network, config.num_epochs, id))
        for epoch in range(config.num_epochs):

            model.train()
            pred = model(train_X)

            train_loss = criterion(pred, train_Y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())
            optimizer_scheduler.step()

            if (epoch + 1) % config.print_every == 0:
                print("Train | Epoch: [{}/{}] | Train Loss {:.4f}".format(epoch + 1, config.num_epochs, np.average(train_losses)))

        if config.transfer_learning:
            torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}_transfer.pkl'.format(config.network, id)))
            np.save(os.path.join(config.numpy_path, '{}_Device_ID_{}_train_loss_transfer.npy'.format(config.network, id)), train_losses)
        else:
            torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(config.network, id)))
            np.save(os.path.join(config.numpy_path, '{}_Device_ID_{}_train_loss.npy'.format(config.network, id)), train_losses)

    elif config.mode == 'test':

        # Prepare Network #
        if config.transfer_learning:
            model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(config.network, id))))
        else:
            model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(config.network, id))))

        with torch.no_grad():
            pred = model(test_X)
            if config.transfer_learning:
                test_Y, pred = plot_test_set(config.plots_path, id, config.network, scaler, pred, test_Y, test_shifted, transfer_learning=True)
            else:
                test_Y, pred = plot_test_set(config.plots_path, id, config.network, scaler, pred, test_Y, test_shifted)

        # Print Statistics #
        print("Metrics of {} follow:".format(config.network))
        print(" MAE : {:.4f}".format(MAE(test_Y, pred)))
        print(" MSE : {:.4f}".format(MSE(test_Y, pred)))
        print("RMSE : {:.4f}".format(RMSE(test_Y, pred)))
        print(" MPE : {:.4f}".format(MPE(test_Y, pred)))
        print("MAPE : {:.4f}".format(MAPE(test_Y, pred)))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()