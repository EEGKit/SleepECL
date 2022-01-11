from __future__ import print_function

import torch
import torch.nn as nn


class SimCLRMultiPosConLoss(nn.Module):
    """
    adapt SupConLoss to satisfy sequence moco.
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    """
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07, seq_len=3, label_idx=1):
        super(SimCLRMultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.seq_len = seq_len
        self.label_idx = label_idx

    def forward(self, ec_feat, em_feat, pos_mask, temps=None):
        """
        Only sample positive feature from ec_feat and em_feat. All features in qu_feat are negative.

        Args:
            ec_feat: feature from encoder c, shape [bsz, seq_len, feat_dim]
            ec_feat: feature from encoder m, shape [bsz, seq_len, feat_dim]
            pos_mask: shape [bsz, bsz*seq_len], pos_mask_{i,j}=1 if feature j is positive to feature i.
        Returns:
            A loss scalar.
        """
        assert ec_feat.shape[1] == self.seq_len == em_feat.shape[1], 'seq_len must be the same.'
        device = (torch.device('cuda') if ec_feat.is_cuda else torch.device('cpu'))

        batch_size = ec_feat.shape[0]  # N


        anchor_feat = ec_feat[:, self.label_idx]  # [bsz, feat_dim]

        # unfold dim of seq_len
        ec_feat = ec_feat.reshape(-1, ec_feat.shape[-1])  # [bsz* seq_len, feat_dim]
        em_feat = em_feat.reshape(-1, em_feat.shape[-1])  # [bsz* seq_len, feat_dim]
        contrast = torch.cat([ec_feat, em_feat], dim=0)  # [2*bsz* seq_len, feat_dim]
        if temps is not None:
            temps = temps.repeat(1, 2)  # [bsz, 2*bsz*seq_len]
        else:  # if temps is not specified for each sample, use the default temperature.
            temps = self.temperature

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feat, contrast.T), temps)  # [bsz, 2*bsz*seq_len]

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # stable cosine distance. mark F =  2*bsz*seq_len, [bsz, F]

        # tile mask
        # part1. mask and logits_mask of feature c
        # mask-out self-contrast cases
        c_logits_mask = torch.scatter(  # [bsz, bsz*seq_len]
            torch.ones_like(pos_mask),
            1,
            (torch.arange(batch_size) * self.seq_len + self.label_idx).view(-1, 1).to(device),
            0)
        c_mask = pos_mask * c_logits_mask  # [bsz, bsz*seq_len]

        # part2. mask and logits_mask of feature m
        m_logits_mask = torch.ones_like(pos_mask)
        m_mask = pos_mask.clone()

        # combine mask
        logits_mask = torch.cat([c_logits_mask, m_logits_mask], dim=1)  # [bsz, F]
        mask = torch.cat([c_mask, m_mask], dim=1)  # [bsz, F]


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # [bsz, F]
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # [bsz, F]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # [bsz,]

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # [bsz,]
        loss = -mean_log_prob_pos  # [bsz,]
        loss = loss.mean()

        return loss
