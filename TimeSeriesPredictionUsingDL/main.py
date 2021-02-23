import os
import random
import argparse
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from models import DNN, CNN, RNN, LSTM, GRU, RecursiveLSTM, AttentionalLSTM
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):

    # Fix Seed for Reproducibility #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Samples, Weights, and Plots Path #
    paths = [args.weights_path, args.plots_path, args.numpy_path]
    for path in paths:
        make_dirs(path)

    # Prepare Data #
    data = load_data(args.combined_path, args.which_data, args.preprocess, args.resample)[[args.feature]]
    id = args.which_data.split('_')[0]
    print("Data of {} is successfully Loaded!".format(args.which_data))

    # Plot Time-series Data #
    if args.plot:
        plot_full(args.plots_path, data, id, args.feature)
        plot_split(args.plots_path, data, id, args.valid_start, args.test_start, args.feature)

    # Min-Max Scaler #
    scaler = MinMaxScaler()
    data[args.feature] = scaler.fit_transform(data)

    # Split the Dataset #
    copied_data = data.copy()

    if args.multi_step:
        X, y = split_sequence_multi_step(copied_data, args.window, args.output_size)
    else:
        X, y = split_sequence_uni_step(copied_data, args.window)

    # Get Data Loader #
    train_loader, val_loader, test_loader = get_data_loader(X, y, args.train_split, args.test_split, args.batch_size)

    # Constants #
    best_val_loss = 100
    best_val_improv = 0

    # Lists #
    train_losses, val_losses = list(), list()
    val_maes, val_mses, val_rmses, val_mapes, val_mpes, val_r2s = list(), list(), list(), list(), list(), list()
    test_maes, test_mses, test_rmses, test_mapes, test_mpes, test_r2s = list(), list(), list(), list(), list(), list()

    # Prepare Network #
    if args.network == 'dnn':
        model = DNN(args.window,
                    args.hidden_size,
                    args.output_size).to(device)

    elif args.network == 'cnn':
        model = CNN(args.window,
                    args.hidden_size,
                    args.output_size).to(device)

    elif args.network == 'rnn':
        model = RNN(args.input_size,
                    args.hidden_size,
                    args.num_layers,
                    args.output_size).to(device)

    elif args.network == 'lstm':
        model = LSTM(args.input_size,
                     args.hidden_size,
                     args.num_layers,
                     args.output_size,
                     args.bidirectional).to(device)

    elif args.network == 'gru':
        model = GRU(args.input_size,
                    args.hidden_size,
                    args.num_layers,
                    args.output_size).to(device)

    elif args.network == 'recursive':
        model = RecursiveLSTM(args.input_size,
                              args.hidden_size,
                              args.num_layers,
                              args.output_size).to(device)

    elif args.network == 'attentional':
        model = AttentionalLSTM(args.input_size,
                                args.qkv,
                                args.hidden_size,
                                args.num_layers,
                                args.output_size,
                                args.bidirectional).to(device)

    else:
        raise NotImplementedError

    if args.mode == 'train':

        # If fine-tuning #
        if args.transfer_learning:
            model.load_state_dict(torch.load(os.path.join(args.weights_path, 'BEST_{}_Device_ID_12.pkl'.format(model.__class__.__name__))))

            for param in model.parameters():
                param.requires_grad = True

        # Loss Function #
        criterion = torch.nn.MSELoss()

        # Optimizer #
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optimizer_scheduler = get_lr_scheduler(optimizer, args)

        # Train and Validation #
        print("Training {} started with total epoch of {} using Driver ID of {}.".format(model.__class__.__name__, args.num_epochs, id))

        for epoch in range(args.num_epochs):

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

            print("Epoch [{}/{}]".format(epoch + 1, args.num_epochs))
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
                    # val_mpe = mean_percentage_error(label.cpu(), pred_val.cpu())
                    # val_mape = mean_absolute_percentage_error(label.cpu(), pred_val.cpu())
                    val_r2 = r2_score(label.cpu(), pred_val.cpu())

                    # Add item to Lists #
                    val_losses.append(val_loss.item())
                    val_maes.append(val_mae.item())
                    val_mses.append(val_mse.item())
                    val_rmses.append(val_rmse.item())
                    # val_mpes.append(val_mpe.item())
                    # val_mapes.append(val_mape.item())
                    val_r2s.append(val_r2.item())

                # Print Statistics #
                print("Validation")
                print("Loss : {:.4f}".format(np.average(val_losses)))
                print(" MAE : {:.4f}".format(np.average(val_maes)))
                print(" MSE : {:.4f}".format(np.average(val_mses)))
                print("RMSE : {:.4f}".format(np.average(val_rmses)))
                # print(" MPE : {:.4f}".format(np.average(val_mpes)))
                # print("MAPE : {:.4f}".format(np.average(val_mapes)))
                print(" R^2 : {:.4f}".format(np.average(val_r2s)))

                # Save the model only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)

                    if args.transfer_learning:
                        torch.save(model.state_dict(), os.path.join(args.weights_path, 'BEST_{}_Device_ID_{}_transfer.pkl'.format(model.__class__.__name__, id)))
                    else:
                        torch.save(model.state_dict(), os.path.join(args.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(model.__class__.__name__, id)))

                    print("Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += 1
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

                    if best_val_improv == 10:
                        break

    elif args.mode == 'test':

        # Prepare Network #
        if args.transfer_learning:
            model.load_state_dict(torch.load(os.path.join(args.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(model.__class__.__name__, id))))
        else:
            model.load_state_dict(torch.load(os.path.join(args.weights_path, 'BEST_{}_Device_ID_{}.pkl'.format(model.__class__.__name__, id))))

        print("{} for Device ID {} is successfully loaded!".format(model.__class__.__name__, id))

        with torch.no_grad():

            for i, (data, label) in enumerate(test_loader):

                # Data Preparation #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                pred_test = model(data)

                # Convert to Original Value Range #
                pred_test = pred_test.data.cpu().numpy()
                label = label.data.cpu().numpy()

                if not args.multi_step:
                    label = label.reshape(-1, 1)

                pred_test = scaler.inverse_transform(pred_test)
                label = scaler.inverse_transform(label)

                # Calculate Loss #
                test_mae = mean_absolute_error(label, pred_test)
                test_mse = mean_squared_error(label, pred_test, squared=True)
                test_rmse = mean_squared_error(label, pred_test, squared=False)
                # test_mpe = mean_percentage_error(label, pred_test)
                # test_mape = mean_absolute_percentage_error(label, pred_test)
                test_r2 = r2_score(label, pred_test)

                # Add item to Lists #
                test_maes.append(test_mae.item())
                test_mses.append(test_mse.item())
                test_rmses.append(test_rmse.item())
                # test_mpes.append(test_mpe.item())
                # test_mapes.append(test_mape.item())
                test_r2s.append(test_r2.item())

            # Print Statistics #
            print("Test {}".format(model.__class__.__name__))
            print(" MAE : {:.4f}".format(np.average(test_maes)))
            print(" MSE : {:.4f}".format(np.average(test_mses)))
            print("RMSE : {:.4f}".format(np.average(test_rmses)))
            # print(" MPE : {:.4f}".format(np.average(test_mpes)))
            # print("MAPE : {:.4f}".format(np.average(test_mapes)))
            print(" R^2 : {:.4f}".format(np.average(test_r2s)))

            # Derive Metric and Plot #
            if args.transfer_learning:
                test_plot(pred_test, label, args.plots_path, args.feature, id, model, transfer_learning=False)
            else:
                test_plot(pred_test, label, args.plots_path, args.feature, id, model, transfer_learning=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234, help='fix seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='total epoch')
    parser.add_argument('--window', type=int, default=60, help='the number of window, unit : second')

    parser.add_argument('--train_split', type=float, default=0.8, help='train-test split ratio')
    parser.add_argument('--test_split', type=float, default=0.5, help='validation-test split ratio')

    parser.add_argument('--plot', type=bool, default=False, help='plot graph or not')
    parser.add_argument('--preprocess', type=bool, default=False, help='remove outliers')
    parser.add_argument('--resample', type=bool, default=False, help='resample')
    parser.add_argument('--multi_step', type=bool, default=False, help='multi step or not')

    parser.add_argument('--feature', type=str, default='speed', help='extract which feature for prediction')
    parser.add_argument('--network', type=str, default='lstm', choices=['dnn', 'cnn', 'rnn', 'lstm', 'gru', 'recursive', 'attentional'])
    parser.add_argument('--transfer_learning', type=bool, default=False, help='transfer learning')

    parser.add_argument('--which_data', type=str, default='12_sep_oct_nov_nov_dec.csv', help='which data to use')
    parser.add_argument('--combined_path', type=str, default='./combined/', help='combined data path')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--numpy_path', type=str, default='./results/numpy/', help='numpy files path')
    parser.add_argument('--plots_path', type=str, default='./results/plots/', help='plots path')

    parser.add_argument('--valid_start', type=str, default='2017-12-12 00:00:00', help='validation start date')
    parser.add_argument('--test_start', type=str, default='2017-12-23 00:00:00', help='test start date')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
    parser.add_argument('--lr_decay_every', type=int, default=100, help='decay learning rate for every n epoch')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

    parser.add_argument('--input_size', type=int, default=1, help='input_size')
    parser.add_argument('--hidden_size', type=int, default=10, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--output_size', type=int, default=1, help='output_size')
    parser.add_argument('--bidirectional', type=bool, default=False, help='bidirectional or not')
    parser.add_argument('--qkv', type=int, default=10, help='key')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    parser.add_argument('-f')
    args = parser.parse_args()

    torch.cuda.empty_cache()
    main(args)

    # TODO
    # 1. percentage error fix
    # 2. multi step time series plot