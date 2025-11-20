import os
import random
import sys
import numpy as np
import scipy.io as sio
from scipy import sparse
import torch
import torchvision


def load_data(data_name):
    """Load data """
    main_dir = sys.path[0]
    X_list = []
    Y_list = []

    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene-15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse-21.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['Caltech101_7']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Caltech101_7.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['HandWritten']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'HandWritten.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['ALOI_100']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'ALOI_100.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[2].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['YTF10']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'YTF10.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['AWA']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'AWA7V.mat'))
        X = mat['X'][0]
        X_list.append(X[5].astype('float32'))
        X_list.append(X[6].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))


    elif data_name in ['NoisyMNIST']:
        mat = sio.loadmat(os.path.join(main_dir, 'data','NoisyMNIST30000.mat'))
        X_list.append(mat['X1'])
        X_list.append(mat['X2'])
        Y_list.append(np.squeeze(mat['Y']))


    return X_list, Y_list
