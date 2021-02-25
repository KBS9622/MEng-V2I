import os
import math
from sympy import *
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from lmfit import minimize, Minimizer, Parameters, Parameter, report_fit
import matplotlib.pyplot as plt
import lmfit


def load_csv_data(file_name, subdir=''):
    """
    Loads data from .csv file in to DataFrame

    :param file_name: .csv file name in string
    :param subdir: optional parameter to specify the subdirectory of the file
    :return: extracted data in DataFrame
    """

    file_dir = os.path.realpath('../')
    print(file_dir)
    for root, dirs, files in os.walk(file_dir):
        if root.endswith(subdir):
            for name in files:
                if name == file_name:
                    file_path = os.path.join(root, name)

    df = pd.read_csv(file_path)

    return df

if __name__ == '__main__':
    LMparams = Parameters()
    LMparams.add('A_FS', value = 1.)
    LMparams.add('w_FS', value = 1., min = 0, max = 0.01*2*math.pi)
    LMparams.add('phi_FS', value = 1., min = -math.pi, max = math.pi)

    

    # print(os.getcwd())
    # loads the csv file
    subdir = 'caltrans_processed_drive_cycles/data/1035198_1'
    file_name = '2012-05-22.csv'
    data = load_csv_data(file_name, subdir)
    # get a slice of the data with a relatively long cruising period
    data = data.iloc[1002:1096,:]
    data.reindex()
    # print(data)
    plt.plot(data.loc[:,'timestamp'], data.loc[:,'speed_mph'])
    # plt.show()
    # get the slice of ONLY cruising period
    cruising_data = data.iloc[25:86,:]


    x = np.linspace(1,len(cruising_data),len(cruising_data))
    temp = cruising_data.loc[:,'speed_mph'].to_numpy()
    print(x)
    y = temp- temp.mean()
    print(type(y))
    plt.plot(x, y, 'b')
    plt.show()

# x = np.linspace(1, 10, 250)
# np.random.seed(0)
# y = 3.0 * np.exp(-x / 2) - 5.0 * np.exp(-(x - 0.1) / 10.) + 0.1 * np.random.randn(x.size)
# plt.plot(x, y, 'b')
# plt.show()