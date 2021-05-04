import os
import argparse
import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler

from models import SkipGenerator
from utils import make_dirs, pre_processing, moving_windows, post_processing, plot_sample, make_csv


def generate_timeseries(args):

    # Device Configuration #
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Inference Path #
    make_dirs(args.inference_path)

    # Prepare Generator #
    if args.model == 'skip':
        G = SkipGenerator(args.latent_dim, args.ts_dim, args.conditional_dim).to(device)
        G.load_state_dict(torch.load(os.path.join(args.weights_path, 'TimeSeries_Generator_using{}_Epoch_{}.pkl'.format(args.criterion.upper(), args.num_epochs))))

    else:
        raise NotImplementedError
    
    # Prepare Data #
    data = pd.read_csv(args.data_path)[args.column]
    
    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()

    preprocessed_data = pre_processing(data, scaler_1, scaler_2, args.delta)

    X = moving_windows(preprocessed_data, args.ts_dim)
    label = moving_windows(data.to_numpy(), args.ts_dim)

    # Lists #
    real, fake = list(), list()

    # Inference #
    for idx in range(0, data.shape[0], args.ts_dim):
        
        end_ix = idx + args.ts_dim

        if end_ix > len(data)-1:
            break

        samples = X[idx, :]
        samples = np.expand_dims(samples, axis=0)
        samples = np.expand_dims(samples, axis=1)

        samples = torch.from_numpy(samples).to(device)
        start_dates = label[idx, 0]

        noise = torch.randn(args.val_batch_size, 1, args.latent_dim).to(device)

        with torch.no_grad():
            fake_series = G(noise)
        fake_series = torch.cat((samples[:, :, :args.conditional_dim].float(), fake_series.float()), dim=2)

        samples = np.squeeze(samples.cpu().data.numpy())
        fake_series = np.squeeze(fake_series.cpu().data.numpy())
        
        samples = post_processing(samples, start_dates, scaler_1, scaler_2, args.delta)
        fake_series = post_processing(fake_series, start_dates, scaler_1, scaler_2, args.delta)

        real += samples.tolist()
        fake += fake_series.tolist()
    
    plot_sample(real, fake, args.num_epochs-1, args)
    make_csv(real, fake, args.num_epochs-1, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_num', type=int, default=5, help='gpu number')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--val_batch_size', type=int, default=1, help='mini-batch size for validation')

    parser.add_argument('--data_path', type=str, default='./data/12_sep_oct_nov_nov_dec.csv', help='data path')
    parser.add_argument('--column', type=str, default='speed', help='which column to generate')
    parser.add_argument('--criterion', type=str, default='wgangp', choices=['l2', 'wgangp'], help='criterion')

    parser.add_argument('--model', type=str, default='skip', choices=['skip', 'recur'], help='which network to train')

    parser.add_argument('--delta', type=float, default=0.8, help='delta')
    parser.add_argument('--ts_dim', type=int, default=600, help='time series dimension, how many time steps to synthesize')
    parser.add_argument('--latent_dim', type=int, default=100, help='noise dimension')
    parser.add_argument('--conditional_dim', type=int, default=3, help='conditional dimension')

    parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
    parser.add_argument('--inference_path', type=str, default='./results/inference/', help='inference path')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--num_epochs', type=int, default=5000, help='total epoch for training')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    generate_timeseries(args)