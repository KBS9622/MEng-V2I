import os
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch

from config import *
from models import DNN, CNN, RNN, LSTM, GRU, RecursiveLSTM, AttentionalLSTM
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    #print the config args
    print(config.transfer_learning)
    print(config.mode)
    print(config.input_size)

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
    data = load_data(config.combined_path, config.which_data, config.preprocess, config.resample)
    # id = config.which_data.split('_')[0]
    id = 12 #BOON added
    print("Data of {} is successfully Loaded!".format(config.which_data))
    print(type(data))
    print(data.shape)

    # Plot Time-series Data #
    if config.plot:
        plot_full(config.plots_path, data, id, config.feature)
        plot_split(config.plots_path, data, id, config.valid_start, config.test_start, config.feature)

    # Min-Max Scaler #
    scaler = MinMaxScaler()
    data.iloc[:,:] = scaler.fit_transform(data)
    print(type(data))

    # Split the Dataset #
    train_X, train_Y, val_X, val_Y, test_X, test_Y, test_shifted = \
        get_time_series_data_(data, config.valid_start, config.test_start, config.feature, config.label, config.window)

    print(train_X.shape)
    print(train_Y.shape)

    # Get Data Loader #
    train_loader, val_loader, test_loader = \
        get_data_loader(train_X, train_Y, val_X, val_Y, test_X, test_Y, config.batch_size)

    # Constants #
    best_val_loss = 100
    best_val_improv = 0

    # Lists #
    train_losses, val_losses = list(), list()
    val_maes, val_mses, val_rmses, val_mapes, val_mpes, val_r2s = list(), list(), list(), list(), list(), list()

    # Prepare Network #
    if config.network == 'dnn':
        model = DNN(config.window, config.hidden_size, config.output_size).to(device)
    elif config.network == 'cnn':
        model = CNN(config.window, config.hidden_size, config.output_size).to(device)
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

        # If fine-tuning #
        print('config.TL = {}'.format(config.transfer_learning))
        if config.transfer_learning:
            print('config.TL = {}'.format(config.transfer_learning))
            print('TL: True')
            model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}_Device_ID_12.pkl'.format(config.network))))

            for param in model.parameters():
                param.requires_grad = True

        # Loss Function #
        criterion = torch.nn.MSELoss()

        # Optimizer #
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.5, 0.999))
        optimizer_scheduler = get_lr_scheduler(config.lr_scheduler, optimizer, config)

        # Train and Validation #
        print("Training {} started with total epoch of {} using Driver ID of {}.".format(config.network, config.num_epochs, id))
        for epoch in range(config.num_epochs):

            # Train #
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

            print("Epoch [{}/{}]".format(epoch+1, config.num_epochs))
            print("Train")
            print("Loss : {:.4f}".format(np.average(train_losses)))

            optimizer_scheduler.step()

            # Validation #
            with torch.no_grad():
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
                print("Validation")
                print("Loss : {:.4f}".format(np.average(val_losses)))
                print(" MAE : {:.4f}".format(np.average(val_maes)))
                print(" MSE : {:.4f}".format(np.average(val_mses)))
                print("RMSE : {:.4f}".format(np.average(val_rmses)))
                print(" MPE : {:.4f}".format(np.average(val_mpes)))
                print("MAPE : {:.4f}".format(np.average(val_mapes)))
                print(" R^2 : {:.4f}".format(np.average(val_r2s)))

                # Save the model only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)

                    # if config.transfer_learning:
                    #     torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}_transfer.pkl'.format(config.network, id)))
                    # else:
                    #     torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(config.network, id)))

                    if config.transfer_learning:
                        torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}_transfer_BOON_reshaped.pkl'.format(config.network, id)))
                    else:
                        torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}_BOON_reshaped.pkl'.format(config.network, id)))

                    print("Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += 1
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

                    if best_val_improv == 10:
                        break

    elif config.mode == 'test':

        # Prepare Network #
        if config.transfer_learning:
            model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}_transfer_BOON_reshaped.pkl'.format(config.network, id))))
        else:
            model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}_Device_ID_{}_BOON_reshaped.pkl'.format(config.network, id))))

        print("{} for Device ID {} is successfully loaded!".format((config.network).upper(), id))

        with torch.no_grad():

            pred_test, labels = list(), list()

            for i, (data, label) in enumerate(test_loader):

                # Data Preparation #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                pred = model(data)

                # Add items to Lists #
                pred_test += pred
                labels += label

            # Derive Metric and Plot #
            if config.transfer_learning:
                pred, actual = test(config.plots_path, id, config.network, scaler, pred_test, labels, test_shifted, transfer_learning=True)
            else:
                pred, actual = test(config.plots_path, id, config.network, scaler, pred_test, labels, test_shifted)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
    print('hi Boon')

    # 1. be able to set test start and end date #
    # 2. output a dataframe consisted of 'datetime' and 'speed (pred)' and 'speed (actual)' #