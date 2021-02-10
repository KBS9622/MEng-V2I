import os
import random
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import *
from models import DNN, CNN, RNN, LSTM, GRU, RecursiveLSTM, AttentionalLSTM
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Reproducibility #
cudnn.deterministic = True
cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():

    # Fix Seed for Reproducibility #
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Samples, Weights, and Plots Path #
    paths = [config.weights_path, config.plots_path, config.numpy_path]
    for path in paths:
        make_dirs(path)

    # Prepare Data #
    data = load_data(config.which_data, config.resample)[[config.feature]]
    id = config.which_data.split('_')[0]
    print("Data of {} Successfully Loaded!".format(config.which_data))

    # Plot Time-series Data #
    if config.plot:
        plot_full(config.plots_path, data, id, config.feature)
        plot_split(config.plots_path, data, id, config.valid_start, config.test_start, config.feature)

    # Min-Max Scaler #
    scaler = MinMaxScaler()
    data[config.feature] = scaler.fit_transform(data)

    # Split the Dataset #
    train_X, train_Y, val_X, val_Y, test_X, test_Y, test_shifted = \
        get_time_series_data_(data, config.valid_start, config.test_start, config.feature, config.window)

    # Get Data Loader #
    train_loader, val_loader = \
        get_data_loader(train_X, train_Y, val_X, val_Y, config.batch_size, config.val_batch_size)

    # Constants #
    best_val_loss = 100
    best_val_improv = 0

    # Lists #
    train_losses, val_losses = list(), list()
    val_maes, val_mses, val_rmses, val_mapes, val_mpes, val_r2s = list(), list(), list(), list(), list(), list()

    # Prepare Network #
    if config.network == 'dnn':
        model = DNN(config.seq_length, config.hidden_size, config.output_size).to(device)
    elif config.network == 'cnn':
        model = CNN(config.seq_length, config.batch_size).to(device)
    elif config.network == 'rnn':
        model = RNN(config.input_size, config.hidden_size, config.num_layers, config.output_size).to(device)
    elif config.network == 'lstm':
        model = LSTM(config.input_size, config.hidden_size, config.num_layers, config.output_size, config.bidirectional).to(device)
    elif config.network == 'gru':
        model = GRU(config.input_size, config.hidden_size, config.num_layers, config.output_size).to(device)
    elif config.network == 'recursive':
        model = RecursiveLSTM(config.input_size, config.hidden_size, config.num_layers, config.output_size).to(device)
    elif config.network == 'attentional':
        model = AttentionalLSTM(config.input_size, config.key, config.query, config.value, config.hidden_size, config.num_layers, config.output_size, config.bidirectional).to(device)
    else:
        raise NotImplementedError

    if config.mode == 'train':

        if config.transfer_learning:
            model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}_Device_ID_12.pkl'.format(config.network))))

            for param in model.parameters():
                param.requires_grad = True

        # Loss Function #
        criterion = torch.nn.MSELoss()

        # Optimizer #
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        optimizer_scheduler = get_lr_scheduler(config.lr_scheduler, optimizer, config)

        # Train and Validation #
        print("Training {} started with total epoch of {} using Driver ID of {}.".format(config.network, config.num_epochs, id))
        for epoch in range(config.num_epochs):

            # Train #
            model.train()

            for i, (data, label) in enumerate(train_loader):

                # Data Preparation #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                pred = model(data)

                # Calculate Loss #
                train_loss = criterion(pred, label)

                # Back Propagation and Update #
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # Add items to Lists #
                train_losses.append(train_loss.item())

                if (i+1) % config.print_every == 0:
                    print("Train | Epoch [{}/{}] | Iter [{}/{}] | Loss {:.4f}"
                          .format(epoch + 1, config.num_epochs, i+1, len(train_loader), np.average(train_losses)))

            optimizer_scheduler.step()

            # Validation #
            with torch.no_grad():

                model.eval()
                for i, (data, label) in enumerate(val_loader):

                    # Data Preparation #
                    data = data.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.float32)

                    # Forward Data #
                    pred_val = model(data)

                    # Calculate Loss #
                    val_loss = criterion(pred_val, label)
                    val_mae = mean_absolute_error(label.cpu(), pred_val.cpu())
                    val_mse = mean_squared_error(label.cpu(), pred_val.cpu(), squared=True)
                    val_rmse = mean_squared_error(label.cpu(), pred_val.cpu(), squared=False)
                    val_mpe = mean_percentage_error(label.cpu(), pred_val.cpu())
                    val_mape = mean_absolute_percentage_error(label.cpu(), pred_val.cpu())
                    val_r2 = r2_score(label.cpu(), pred_val.cpu())

                    # Add item to Lists #
                    val_losses.append(val_loss.item())
                    val_maes.append(val_mae.item())
                    val_mses.append(val_mse.item())
                    val_rmses.append(val_rmse.item())
                    val_mpes.append(val_mpe.item())
                    val_mapes.append(val_mape.item())
                    val_r2s.append(val_r2.item())

                # Print Statistics #
                print("Val Loss : {:.4f}".format(np.average(val_losses)))
                print("Val  MAE : {:.4f}".format(np.average(val_maes)))
                print("Val  MSE : {:.4f}".format(np.average(val_mses)))
                print("Val RMSE : {:.4f}".format(np.average(val_rmses)))
                print("Val  MPE : {:.4f}".format(np.average(val_mpes)))
                print("Val MAPE : {:.4f}".format(np.average(val_mapes)))
                print("Val  R^2 : {:.4f}".format(np.average(val_r2s)))

                # Save the model Only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)

                    if config.transfer_learning:
                        torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}_transfer.pkl'.format(config.network, id)))
                        np.save(os.path.join(config.numpy_path, '{}_Device_ID_{}_train_loss_transfer.npy'.format(config.network, id)), train_losses)
                    else:
                        torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(config.network, id)))
                        np.save(os.path.join(config.numpy_path, '{}_Device_ID_{}_train_loss.npy'.format(config.network, id)), train_losses)

                    print("Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += 1
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

    elif config.mode == 'test':

        # Prepare Network #
        if config.transfer_learning:
            model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(config.network, id))))
        else:
            model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(config.network, id))))

        with torch.no_grad():

            model.eval()

            # Data Preparation #
            test_X = test_X.to(device, dtype=torch.float32)
            test_Y = test_Y.to(device, dtype=torch.float32)

            # Forward Data #
            pred_test = model(test_X)

            # Derive Metric and Plot #
            if config.transfer_learning:
                test(config.plots_path, id, config.network, scaler, pred_test, test_Y, test_shifted, transfer_learning=True)
            else:
                test(config.plots_path, id, config.network, scaler, pred_test, test_Y, test_shifted)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()


    # TODO #
    # 1. Benchmarking again #
    # 2. Increase windwo size #
    # 3. set test start and end date #
    # 4. to_csv #
    # 5. Resample function & column (preprocessing function) #
    # 6. output a dataframe consisted of 'datetime' and 'speed'
    # 7. Lagging Issue (Deepen the model & increase the window size)