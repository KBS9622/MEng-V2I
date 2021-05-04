import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import argparse
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from models import Discriminator, Generator
from utils import make_dirs, pre_processing, moving_windows, get_lr_scheduler, get_gradient_penalty
from utils import plot_sample, get_samples, generate_fake_samples, make_csv

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(args):

    # Device Configuration #
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Fix Seed for Reproducibility #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Samples, Plots, Weights and CSV Path #
    paths = [args.samples_path, args.plots_path, args.weights_path, args.csv_path]
    for path in paths:
        make_dirs(path)

    # Prepare Data #
    data = pd.read_csv(args.data_path)[args.column]

    # Pre-processing #
    scaler_1 = StandardScaler()
    scaler_2 = StandardScaler()
    preprocessed_data = pre_processing(data, scaler_1, scaler_2, args.delta)

    X = moving_windows(preprocessed_data, args.ts_dim)
    label = moving_windows(data.to_numpy(), args.ts_dim)

    # Prepare Networks #
    D = Discriminator(args.ts_dim).to(device)
    G = Generator(args.latent_dim, args.ts_dim, args.conditional_dim).to(device)

    # Loss Function #
    if args.criterion == 'l2':
        criterion = nn.MSELoss()
    elif args.criterion == 'wgangp':
        pass
    else:
        raise NotImplementedError

    # Optimizers #
    D_optim = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    G_optim = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))

    D_optim_scheduler = get_lr_scheduler(D_optim, args)
    G_optim_scheduler = get_lr_scheduler(G_optim, args)

    # Lists #
    D_losses, G_losses = list(), list()

    # Train #
    print("Training Time Series GAN started with total epoch of {}.".format(args.num_epochs))
    
    for epoch in range(args.num_epochs):

        # Initialize Optimizers #
        G_optim.zero_grad()
        D_optim.zero_grad()

        if args.criterion == 'l2':
            n_critics = 1
        elif args.criterion == 'wgangp':
            n_critics = 5

        #######################
        # Train Discriminator #
        #######################

        for j in range(n_critics):
            series, start_dates = get_samples(X, label, args.batch_size)

            # Data Preparation #
            series = series.to(device)
            noise = torch.randn(args.batch_size, 1, args.latent_dim).to(device)

            # Adversarial Loss using Real Image #
            prob_real = D(series.float())

            if args.criterion == 'l2':
                real_labels = torch.ones(prob_real.size()).to(device)
                D_real_loss = criterion(prob_real, real_labels)

            elif args.criterion == 'wgangp':
                D_real_loss = -torch.mean(prob_real)

            # Adversarial Loss using Fake Image #
            fake_series = G(noise)
            fake_series = torch.cat((series[:, :, :args.conditional_dim].float(), fake_series.float()), dim=2)

            prob_fake = D(fake_series.detach())

            if args.criterion == 'l2':
                fake_labels = torch.zeros(prob_fake.size()).to(device)
                D_fake_loss = criterion(prob_fake, fake_labels)

            elif args.criterion == 'wgangp':
                D_fake_loss = torch.mean(prob_fake)
                D_gp_loss = args.lambda_gp * get_gradient_penalty(D, series.float(), fake_series.float(), device)

            # Calculate Total Discriminator Loss #
            D_loss = D_fake_loss + D_real_loss
                
            if args.criterion == 'wgangp':
                D_loss += args.lambda_gp * D_gp_loss
                        
            # Back Propagation and Update #
            D_loss.backward()
            D_optim.step()

        ###################
        # Train Generator #
        ###################

        # Adversarial Loss #
        fake_series = G(noise)
        fake_series = torch.cat((series[:, :, :args.conditional_dim].float(), fake_series.float()), dim=2)
        prob_fake = D(fake_series)

        # Calculate Total Generator Loss #
        if args.criterion == 'l2':
            real_labels = torch.ones(prob_fake.size()).to(device)
            G_loss = criterion(prob_fake, real_labels)

        elif args.criterion == 'wgangp':
            G_loss = -torch.mean(prob_fake)

        # Back Propagation and Update #
        G_loss.backward()
        G_optim.step()

        # Add items to Lists #
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())

        ####################
        # Print Statistics #
        ####################

        print("Epochs [{}/{}] | D Loss {:.4f} | G Loss {:.4f}".format(epoch+1, args.num_epochs, np.average(D_losses), np.average(G_losses)))
        
        # Adjust Learning Rate #
        D_optim_scheduler.step()
        G_optim_scheduler.step()

        # Save Model Weights and Series #
        if (epoch+1) % args.save_every == 0:
            torch.save(G.state_dict(), os.path.join(args.weights_path, 'TimeSeries_Generator_using{}_Epoch_{}.pkl'.format(args.criterion.upper(), epoch + 1)))
            
            series, fake_series = generate_fake_samples(X, label, G, scaler_1, scaler_2, args, device)
            plot_sample(series, fake_series, epoch, args)
            make_csv(series, fake_series, epoch, args)

    print("Training finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_num', type=int, default=5, help='gpu number')
    parser.add_argument('--seed', type=int, default=1234, help='seed')

    parser.add_argument('--data_path', type=str, default='./data/12_sep_oct_nov_nov_dec.csv', help='data path')
    parser.add_argument('--column', type=str, default='speed', help='which column to generate')

    parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size')
    parser.add_argument('--val_batch_size', type=int, default=1, help='mini-batch size for validation')
    parser.add_argument('--num_epochs', type=int, default=500, help='total epoch for training')
    parser.add_argument('--print_every', type=int, default=100, help='print statistics for every default iteration')
    parser.add_argument('--save_every', type=int, default=10, help='save model weights for every default epoch')

    parser.add_argument('--delta', type=float, default=0.7, help='delta')
    parser.add_argument('--ts_dim', type=int, default=600, help='time series dimension, how many time steps to synthesize')
    parser.add_argument('--latent_dim', type=int, default=100, help='noise dimension')
    parser.add_argument('--conditional_dim', type=int, default=3, help='conditional dimension')

    parser.add_argument('--criterion', type=str, default='wgangp', choices=['l2', 'wgangp'], help='criterion')
    parser.add_argument('--lambda_gp', type=int, default=10, help='constant for gradient penalty')
    parser.add_argument("--n_critics", type=int, default=5, help="number of training iterations for WGAN discriminator")

    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
    parser.add_argument('--lr_decay_every', type=int, default=250, help='decay learning rate for every default epoch')
    parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, options: [Step, Plateau, Cosine]')

    parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
    parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--csv_path', type=str, default='./results/csv/', help='csv path')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    train(args)

    # TODO: Pre-processing & Post-Processing #