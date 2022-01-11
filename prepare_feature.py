"""
generate features and save to npz.
"""
import os
import pickle
from scipy import signal
import scipy
import numpy as np


def bandpower(x, fs, fmin, fmax):
    f, Pxx = signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    res = scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
    return res


def get_vec_bp(x, fs=100):
    vec_bp = np.zeros(shape=(x.shape[0], 4))
    for idx, _x in enumerate(x):
        delta_bp = bandpower(_x, fs=fs, fmin=1, fmax=4)
        theta_bp = bandpower(_x, fs=fs, fmin=4, fmax=8)
        alpha_bp = bandpower(_x, fs=fs, fmin=8, fmax=13)
        bata_bp = bandpower(_x, fs=fs, fmin=13, fmax=30)
        vec_bp[idx, :] = np.array([delta_bp, theta_bp, alpha_bp, bata_bp])
    return vec_bp

def save_feature(data_dir, save_path, feature_func):
    feature_dict = {}
    file_list = os.listdir(data_dir)
    for file in file_list:
        print('reading file', file)
        npz_data = np.load(os.path.join(data_dir, file))
        # x = npz_data['x']
        x = npz_data['x']
        x = x.reshape(x.shape[0], x.shape[1])
        feature_vec = feature_func(x)
        feature_dict[file] = feature_vec
        # with open(save_path, 'wb') as f:
    with open(save_path, 'wb') as f:
        pickle.dump(feature_dict, f)
        # break

def test():
    with open(save_path, 'rb') as f:
        feature_dict = pickle.load(f)
        vec_bp = feature_dict['SC4142E0.npz']
        print()


if __name__ == '__main__':
    data_dir = "./data/sleepedf/sleep-cassette/eeg_fpz_cz"
    save_dir = "./data/sleepedf/sleep-cassette/feature/"
    # data_dir = "/data/ZhangHongjun/codes/sleep/TSTCC/data_preprocessing/sleep-edf/sleepEDF20_fpzcz"
    # save_dir = "/data/ZhangHongjun/codes/sleep/TSTCC/data_preprocessing/sleep-edf/feature"
    filename = "eeg_fpz_cz_powerband.pkl"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, filename)
    save_feature(data_dir, save_path, get_vec_bp)
    print("save file to ", save_path)