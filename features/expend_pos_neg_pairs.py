from .bandpower import bandpower
import numpy as np
import torch

def get_vec_bp(x):
    vec_bp = np.zeros(shape=(x.shape[0], 4))
    for idx, _x in enumerate(x):
        delta_bp = bandpower(_x, fs=100, fmin=0.5, fmax=4)
        theta_bp = bandpower(_x, fs=100, fmin=4, fmax=8)
        alpha_bp = bandpower(_x, fs=100, fmin=8, fmax=13)
        bata_bp = bandpower(_x, fs=100, fmin=13, fmax=30)
        vec_bp[idx, :] = np.array([delta_bp, theta_bp, alpha_bp, bata_bp])
    return vec_bp


def find_pos_feature(seq_x, seq_len, label_idx, topK=8):
    """
    Find positive feature using prior knowledge.
    Input: seq_x, shape [bsz, seq_len, n_ch, time], torch.Tensor
    return: feature mask, shape of [bsz, bsz*seq_len], mask_{i, j}=1 mean sample j is positive to sample i.
    """
    bsz, _seq_len, n_ch, t = seq_x.shape
    assert n_ch == 1 and seq_len == _seq_len
    x = seq_x.view(bsz*seq_len, t)
    np_x = x.cpu().numpy()
    vec_bp = get_vec_bp(np_x)
    vec_bp = torch.from_numpy(vec_bp)  # [bsz*seq_len, 4]
    anchor_vec = vec_bp[torch.arange(bsz) * seq_len + label_idx]  # [bsz, 4]
    ex_anchor_vec = anchor_vec.repeat(1, bsz * seq_len)
    ex_anchor_vec = ex_anchor_vec.reshape(-1, anchor_vec.shape[-1])
    ex_vec_bp = vec_bp.repeat(bsz, 1)
    dist = torch.sum((ex_anchor_vec - ex_vec_bp) ** 2, dim=1)
    dist = dist.reshape(bsz, bsz * seq_len)  # [bsz, bsz*seq_len], Euclidean distance, the smaller, the more similar.

    max_idx = torch.argsort(dist, dim=1)
    pos_idx = max_idx[:, :topK]  # pick the topK smallest.
    pos_mask = torch.zeros(size=(bsz, bsz*seq_len)).to(seq_x.device)
    pos_mask = torch.scatter(pos_mask, dim=1, index=pos_idx.to(seq_x.device), src=torch.ones_like(pos_idx).float().to(seq_x.device))
    return pos_mask


def feature_distance(feature, seq_len, label_idx):

    bsz, _seq_len, feature_dim = feature.shape
    assert _seq_len == seq_len
    feature = feature.reshape(bsz*seq_len, feature_dim)
    anchor_vec = feature[torch.arange(bsz) * seq_len + label_idx]  # [bsz, feature_dim]
    ex_anchor_vec = anchor_vec.repeat(1, bsz * seq_len)
    ex_anchor_vec = ex_anchor_vec.reshape(-1, anchor_vec.shape[-1])
    ex_vec = feature.repeat(bsz, 1)
    dist = torch.sqrt(torch.sum((ex_anchor_vec - ex_vec) ** 2, dim=1) + 0.01)
    dist = dist.reshape(bsz, bsz * seq_len)  # [bsz, bsz*seq_len], Euclidean distance, the smaller, the more similar.
    return dist

def topK_pos_mask(dist, bsz, seq_len, topK):
    max_idx = torch.argsort(dist, dim=1)
    pos_idx = max_idx[:, :topK]  # pick the topK smallest.
    pos_mask = torch.zeros(size=(bsz, bsz * seq_len)).to(dist.device)
    pos_mask = torch.scatter(pos_mask, dim=1, index=pos_idx.to(dist.device),
                             src=torch.ones_like(pos_idx).float().to(dist.device))
    return pos_mask




