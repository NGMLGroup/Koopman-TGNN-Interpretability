import numpy as np
import pandas as pd

from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from torch import from_numpy


def load_dataset(name):
    if name == "acea":
        return load_acea()
    elif name == "spain":
        return load_spain()
    else:
        raise ValueError(f"{name} dataset not defined.")


def load_acea():
    seasonality = 24*7 # 1 week
    forecast_horizon = 24 # 1 day

    mat = loadmat('dataset/TS_Acea.mat')  # load mat-file
    ACEA_data = mat['X'] # original resolution (1 = 10 mins)
    ACEA_data = ACEA_data[::6] # hourly forecast

    # remove 11 weeks anomaly in the dataset
    ACEA_data = np.concatenate((ACEA_data[:16000], ACEA_data[16000+168*11:]))

    return ACEA_data.squeeze(), seasonality, forecast_horizon


def load_spain():
    """
    Spanish energy market daily data.
    Source: https://www.kaggle.com/code/manualrg/daily-electricity-demand-forecast-machine-learning/data
    """

    spain_power_data = np.genfromtxt('dataset/spain_energy_market.csv', delimiter=',', dtype=None, encoding=None)
    data = spain_power_data[...,5] # select column with values
    data = data[spain_power_data[...,2] == 'Demanda real'] # select energy demand values
    data = data.astype(float) # convert into floats

    seasonality = 7 # 1 week
    forecast_horizon = 1 # 1 week

    return data, seasonality, forecast_horizon


def generate_datasets(data, seasonality, forecast_horizon, device, test_percent = 0.15, val_percent = 0.15, scaler = StandardScaler):

    L = seasonality
    F = forecast_horizon

    assert F<=L, "The forecast horizon must be smaller or equal to the seasonality."

    s = pd.Series(data)
    
    # Remove seasonality
    sn = s.diff(periods=L)[L:]
    sn = sn.to_numpy(float)

    diff = (data[L:] - sn)

    X = sn[:-F, np.newaxis]
    Y = sn[F:, np.newaxis]
    diffX = diff[:-F, np.newaxis]
    diffY = diff[F:, np.newaxis]

    n_data,_ = X.shape

    n_te = np.ceil(test_percent*n_data).astype(int)
    n_val = np.ceil(val_percent*n_data).astype(int)
    n_tr = n_data - n_te - n_val

    # Split dataset
    Xtr = X[:n_tr, :]
    Ytr = Y[:n_tr, :]

    Xval = X[n_tr:-n_te, :]
    Yval = Y[n_tr:-n_te, :]

    Xte = X[-n_te:, :]
    Yte = Y[-n_te:, :]
    diffXte = diffX[-n_te:, :]
    diffYte = diffY[-n_te:, :]

    # Scale
    Xscaler = scaler()
    Yscaler = scaler()

    # Fit scaler on training set
    Xtr = Xscaler.fit_transform(Xtr)
    Ytr = to_torch(Yscaler.fit_transform(Ytr), device)

    # Transform the rest
    Xval = Xscaler.transform(Xval)
    Yval = to_torch(Yscaler.transform(Yval), device)

    Xte = Xscaler.transform(Xte)
    Yte = to_torch(Yscaler.transform(Yte), device)

    # Transform the difference due to the seasonality
    Xscaler.with_mean = False
    diffXte = to_torch(Xscaler.transform(diffXte), device)
    diffYte = to_torch(Xscaler.transform(diffYte), device)

    # add constant input
    Xtr = to_torch(np.concatenate((Xtr,np.ones((Xtr.shape[0],1))),axis=1), device)
    Xval = to_torch(np.concatenate((Xval,np.ones((Xval.shape[0],1))),axis=1), device)
    Xte = to_torch(np.concatenate((Xte,np.ones((Xte.shape[0],1))),axis=1), device)

    return Xtr, Ytr, Xval, Yval, Xte, Yte, diffXte, diffYte


def to_torch(array, device):
    """
    Transform numpy arrays to torch tensors and move them to `device`
    """
    
    dtype = 'float32'
    array = array.astype(dtype)
    return from_numpy(array).to(device)