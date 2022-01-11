"""
deal with temperature and positive mask.
"""
from features.expend_pos_neg_pairs import feature_distance, topK_pos_mask
import torch

def set_temp_pos_mask(feature, topK_ratio, old_pos_mask, min_t, max_t, metric='euc'):
    bsz, seq_len = feature.shape[0], feature.shape[1]
    device = feature.device
    topK = int(bsz * seq_len * topK_ratio)

    if metric == 'euc':
        feature_dist = feature_distance(feature, seq_len=seq_len, label_idx=seq_len//2)  # shape [bsz, bsz], the smaller, the more similar.
    elif metric == 'std_euc':
        pass
    elif metric == 'mah':
        pass
    else:
        raise ValueError

    # set positive and negative according feature distance.
    prior_pos_mask = topK_pos_mask(feature_dist, bsz, seq_len=seq_len, topK=topK)  # ttopK is a essential param.
    pos_mask = (old_pos_mask.bool() | prior_pos_mask.bool()).float()

    # set positive and negative according feature distance and positive mask.
    temps = torch.full_like(pos_mask, fill_value=(min_t+max_t)/2, device=device)

    return temps, pos_mask


