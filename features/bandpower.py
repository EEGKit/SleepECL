"""
extractor band power directory using scipy
"""
from scipy import signal
import scipy
import numpy as np
def bandpower(x, fs, fmin, fmax):
    f, Pxx = signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    res = scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
    return res


def get_vec_bp_ratio(x):
    vec_bp = np.zeros(shape=(x.shape[0], 4))
    for idx, _x in enumerate(x):
        delta_bp = bandpower(_x, fs=100, fmin=0.5, fmax=4)
        theta_bp = bandpower(_x, fs=100, fmin=4, fmax=8)
        alpha_bp = bandpower(_x, fs=100, fmin=8, fmax=13)
        bata_bp = bandpower(_x, fs=100, fmin=13, fmax=30)
        vec_bp[idx, :] = np.array([delta_bp, theta_bp, alpha_bp, bata_bp]) / sum([delta_bp, theta_bp, alpha_bp, bata_bp])
    return vec_bp


def get_vec_bp(x, fs=100):
    vec_bp = np.zeros(shape=(x.shape[0], 4))
    for idx, _x in enumerate(x):
        delta_bp = bandpower(_x, fs=fs, fmin=0.5, fmax=4)
        theta_bp = bandpower(_x, fs=fs, fmin=4, fmax=8)
        alpha_bp = bandpower(_x, fs=fs, fmin=8, fmax=13)
        bata_bp = bandpower(_x, fs=fs, fmin=13, fmax=30)
        vec_bp[idx, :] = np.array([delta_bp, theta_bp, alpha_bp, bata_bp])
    return vec_bp