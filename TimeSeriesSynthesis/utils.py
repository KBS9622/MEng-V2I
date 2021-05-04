import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import lambertw

import torch
from torch.autograd import grad


def make_dirs(path):
    """Make Directory If not Exists"""
    if not os.path.exists(path):
        os.makedirs(path)


def pre_processing(data, scaler_1, scaler_2, delta):
    """Pre-processing"""
    data += 1e-1
    log_returns = np.log(data/data.shift(1)).fillna(0).to_numpy()
    log_returns = np.reshape(log_returns, (log_returns.shape[0], 1))
    log_returns = scaler_1.fit_transform(log_returns)
    log_returns = np.squeeze(log_returns)
    
    log_returns_w = (np.sign(log_returns) * np.sqrt(lambertw(delta*log_returns**2)/delta)).real
    log_returns_w = log_returns_w.reshape(-1, 1)
    log_returns_w = scaler_2.fit_transform(log_returns_w)
    log_returns_w = np.squeeze(log_returns_w)

    return log_returns_w


def post_processing(data, start_dates, scaler_1, scaler_2, delta):
    """Post-processing"""
    data = scaler_2.inverse_transform(data)
    data = data * np.exp(0.5 * delta * data **2)
    data = scaler_1.inverse_transform(data)
    data = np.exp(data)
    
    post_data = np.empty((data.shape[0], ))
    post_data[0] = start_dates
    for i in range(1, data.shape[0]):
        post_data[i] = post_data[i-1] * data[i]

    return np.array(post_data)


def moving_windows(x, length):
    """Moving Windows"""
    X = list()

    for i in range(0, len(x)+1-length, 4):
        X.append(x[i : i + length])
    X = np.array(X)

    return X


def get_lr_scheduler(optimizer, args):
    """Learning Rate Scheduler"""
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_every, threshold=0.001, patience=1)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=0)
    else:
        raise NotImplementedError

    return scheduler


def get_gradient_penalty(discriminator, real_images, fake_images, device):
    """Gradient Penalty"""
    epsilon = torch.rand(real_images.size(0), 1, 1).to(device)
    epsilon = epsilon

    x_hat = (epsilon * real_images + (1 - epsilon) * fake_images).requires_grad_(True)
    x_hat_prob = discriminator(x_hat)
    x_hat_grad = torch.ones(x_hat_prob.size()).to(device)

    gradients = grad(outputs=x_hat_prob,
                     inputs=x_hat,
                     grad_outputs=x_hat_grad,
                     create_graph=True,
                     retain_graph=True,
                     only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)

    eps = 1e-12
    gradient_penalty = torch.sqrt(torch.sum(gradients ** 2, dim=1) + eps)
    gradient_penalty = torch.mean((gradient_penalty-1)**2)

    return gradient_penalty


def get_samples(data, label, batch_size):
    """Get Samples"""
    idx = np.random.randint(data.shape[0], size=batch_size)

    samples = data[idx, :]
    samples = np.expand_dims(samples, axis=1)
    samples = torch.from_numpy(samples)
    
    start_dates = label[idx, 0]

    return samples, start_dates


def generate_fake_samples(test_set, label, generator, scaler_1, scaler_2, args, device):
    """Generate Fake Samples"""

    series, start_dates = get_samples(test_set, label, args.val_batch_size)
    series = series.to(device)

    noise = torch.randn(args.val_batch_size, 1, args.latent_dim).to(device)

    fake_series = generator(noise.detach())
    fake_series = torch.cat((series[:, :, :args.conditional_dim].float(), fake_series.float()), dim=2)

    series = np.squeeze(series.cpu().data.numpy())
    fake_series = np.squeeze(fake_series.cpu().data.numpy())
        
    series = post_processing(series, start_dates, scaler_1, scaler_2, args.delta)
    fake_series = post_processing(fake_series, start_dates, scaler_1, scaler_2, args.delta)

    return series, fake_series


def plot_sample(series, fake_series, epoch, args):
    """Plot Samples"""
    
    plt.figure(figsize=(10, 5))
    plt.plot(series, label='real')
    plt.plot(fake_series, label='fake')
    plt.grid(True)
    plt.legend()
    plt.title('Generated_TimeSeries_Epoch{}_using{}.png'.format(epoch+1, args.criterion.upper()))

    plt.savefig(os.path.join(args.samples_path, 'Generated_TimeSeries_using{}_Epoch{}.png'.format(args.criterion.upper(), epoch+1)))


def make_csv(series, fake_series, epoch, args):
    """Convert to CSV files"""

    data = pd.DataFrame({'series' : series, 'fake_series' : fake_series})

    data.to_csv(
        os.path.join(args.csv_path, 'Generated_TimeSeries_using{}_Epoch{}.csv'.format(args.criterion.upper(), epoch+1)),
        header=['Real', 'Fake'],
        index=False
        )