# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from losses.simclr_multi_pos_nce import SimCLRMultiPosConLoss
from losses.simclr_multi_seperate_pos_nce import SimCLRMultiSepPosConLoss
from features.tempAndPosMask import set_temp_pos_mask

class SimCLRContextNonOverlapSkip(nn.Module):
    def __init__(self, cnn_encoder, context_encoder, args, mlp=True):
        super(SimCLRContextNonOverlapSkip, self).__init__()
        self.seq_len = args.window_size
        self.label_idx = args.window_size // 2

        self.sub_seq_len = args.sub_window_size
        self.sub_label_idx = args.sub_window_size // 2

        self.step = args.step
        self.num_sub_seq = (self.seq_len - self.sub_seq_len) // self.step + 1
        assert self.num_sub_seq % 2 == 1
        self.context_idx = self.num_sub_seq // 2

        self.T = args.T

        NceLoss = SimCLRMultiSepPosConLoss if args.sep_pos else SimCLRMultiPosConLoss
        self.cnn_nce = NceLoss(temperature=self.T, seq_len=self.seq_len, label_idx=self.label_idx)
        self.sa_nce = NceLoss(temperature=self.T, seq_len=self.num_sub_seq, label_idx=self.context_idx)
        self.sa_ce = nn.CrossEntropyLoss()

        self.cnn_encoder = cnn_encoder(feature_dim=args.feature_dim, fs=args.fs, down_linear=args.down_linear)
        self.context_encoder = context_encoder(seq_len=self.sub_seq_len, feature_dim=args.feature_dim,  n_head=args.n_head, n_layer=args.n_layer, pretrain=True)


    def forward(self, no_aug_seq_q, seq_q, seq_k, args, feature=None, y=None):
        """
        Input:
            seq_q: a batch of query seq, shape: batch_size* seq_len* ch* time
            seq_k: a batch of key seq, shape: batch_size* seq_len* ch* time
        Output:
            contrastive loss.
        """
        bsz = seq_q.shape[0]
        device = seq_q.device
        input = torch.cat([seq_q, seq_k], dim=0)
        # qk, sa_embedding = self.encoder_q(input)  # optional
        # qk = nn.functional.normalize(qk, dim=2)  #
        # q, k = qk[:bsz], qk[bsz:]

        #####################################
        # cnn part loss
        #####################################
        # calculate nce loss for cnn encoder
        cnn_qk, sa_in_origin = self.cnn_encoder(input)
        loss = 0
        if len(args.cnn_level) != 0:
            cnn_qk = nn.functional.normalize(cnn_qk, dim=2)
            cnn_q, cnn_k = cnn_qk[:bsz], cnn_qk[bsz:]

            cnn_bsz, cnn_seq_len, cnn_feature_dim = cnn_k.shape
            cnn_label_idx = self.label_idx

            cnn_pos_index = (torch.arange(cnn_bsz) * cnn_seq_len + cnn_label_idx).reshape(-1, 1).to(device)
            cnn_pos_mask = torch.scatter(  # [bsz, bsz*seq_len], mask all positive features.  # no DCC
                torch.zeros(size=(cnn_bsz, cnn_bsz * cnn_seq_len)).to(device),
                1,
                cnn_pos_index,
                1)
            cnn_temps = torch.full_like(cnn_pos_mask, fill_value=self.T, device=device)
            if 'unbiased' in args.cnn_level:
                cnn_y = y.reshape(-1, 1)
                cnn_sup_mask = torch.eq(cnn_y, cnn_y.T).float().to(device)
                cnn_sup_mask = cnn_sup_mask[torch.arange(cnn_bsz) * cnn_seq_len + cnn_label_idx]
                cnn_pos_mask = cnn_sup_mask
            else:
                if 'prior' in args.cnn_level:
                    cnn_feature = feature
                    cnn_temps, cnn_pos_mask = set_temp_pos_mask(cnn_feature, args.topK_ratio, cnn_pos_mask,
                                                                min_t=self.T, max_t=self.T, metric=args.metric)
                if 'dcc' in args.cnn_level:
                    cnn_index = torch.arange(cnn_bsz).reshape(-1, 1)  # [bsz, 1]
                    cnn_index_repeat = cnn_index.repeat(1, cnn_seq_len).reshape(1, -1)  # [bsz, 1]
                    cnn_dcc_pos_mask = torch.scatter(  # [bsz, bsz*seq_len], mask all positive features.
                        torch.zeros(size=(cnn_bsz, cnn_bsz * cnn_seq_len)).to(cnn_q.device),
                        0,
                        cnn_index_repeat.to(cnn_q.device),
                        1)
                    if 'prior' in args.cnn_level:
                        cnn_pos_mask = (cnn_pos_mask.bool() & cnn_dcc_pos_mask.bool()).float()
                    else:
                        cnn_pos_mask = (cnn_pos_mask.bool() | cnn_dcc_pos_mask.bool()).float()

            cnn_nce_loss = self.cnn_nce(cnn_q, cnn_k, cnn_pos_mask, cnn_temps)
            loss += cnn_nce_loss

        if len(args.sa_level) != 0:
            #####################################
            # sa part loss
            #####################################
            # calculate nce loss for self-attention encoder
            # sa_in_origin, shape [2*cnn_bsz, cnn_seq_len, cnn_feature_dim]
            # slide on sa_in_origin to generate sub sequence for self-attention module.
            # shape [2*cnn_bsz, num_sub_seq, cnn_feature_dim, sub_seq_len] -> [2*cnn_bsz, num_sub_seq, sub_seq_len, cnn_feature_dim]
            sa_input = sa_in_origin.unfold(dimension=1, size=self.sub_seq_len, step=self.step).permute(0, 1, 3, 2)

            double_cnn_bsz, num_sub_seq, sub_seq_len, cnn_feature_dim = sa_input.shape
            assert num_sub_seq ==  self.num_sub_seq
            sa_input = sa_input.reshape(shape=(double_cnn_bsz * num_sub_seq, sub_seq_len, cnn_feature_dim))
            sa_qk, sa_binary = self.context_encoder(sa_input)
            # sa_qk, shape [2bsz*num_sub_seq, sa_feature_dim]
            # sa_binary, shape [2bsz*num_sub_seq, 2]
            sa_qk = sa_qk.reshape(2*bsz, num_sub_seq, -1)
            sa_qk = nn.functional.normalize(sa_qk, dim=2)
            sa_q, sa_k = sa_qk[:bsz], sa_qk[bsz:]
            sa_bsz, num_sub_seq, sa_feature_dim = sa_q.shape
            context_idx = self.context_idx

            sa_pos_index = (torch.arange(sa_bsz) * num_sub_seq + context_idx).reshape(-1, 1).to(device)
            sa_pos_mask = torch.scatter(  # [bsz, bsz*seq_len], mask all positive features.  # no DCC
                torch.zeros(size=(sa_bsz, sa_bsz * num_sub_seq)).to(device),
                1,
                sa_pos_index,
                1)
            sa_temps = torch.full_like(sa_pos_mask, fill_value=self.T, device=device)
            if 'unbiased' in args.sa_level:
                # y, shape [bsz, seq_len]
                sa_y = y.unfold(dimension=1, size=self.sub_seq_len, step=self.step)  # [bsz, num_sub_seq, sub_seq_len]
                sa_y = sa_y[:, :, self.sub_label_idx]  # [bsz, num_sub_seq]
                sa_y = sa_y.reshape(-1, 1)
                sa_sup_mask = torch.eq(sa_y, sa_y.T).float().to(device)
                sa_sup_mask = sa_sup_mask[torch.arange(sa_bsz) * num_sub_seq + context_idx]
                sa_pos_mask = sa_sup_mask
            else:
                if 'prior' in args.sa_level:
                    sa_feature = feature.unfold(dimension=1, size=self.sub_seq_len, step=self.step).permute(0, 1, 3,
                                                                                                    2)  # [bsz, num_sub_seq, sub_seq_len, 4]
                    sa_feature = sa_feature[:, :, self.sub_label_idx, :]  # [bsz, num_sub_seq, 4]
                    sa_temps, sa_pos_mask = set_temp_pos_mask(sa_feature, args.topK_ratio, sa_pos_mask, min_t=self.T,
                                                              max_t=self.T, metric=args.metric)
                if 'dcc' in args.sa_level:
                    sa_index = torch.arange(sa_bsz).reshape(-1, 1)  # [bsz, 1]
                    sa_index_repeat = sa_index.repeat(1, num_sub_seq).reshape(1, -1)  # [bsz, 1]
                    sa_dcc_pos_mask = torch.scatter(  # [bsz, bsz*seq_len], mask all positive features.
                        torch.zeros(size=(sa_bsz, sa_bsz * num_sub_seq)).to(device),
                        0,
                        sa_index_repeat.to(device),
                        1)
                    if 'prior' in args.sa_level:
                        sa_pos_mask = (sa_pos_mask.bool() & sa_dcc_pos_mask.bool()).float()
                    else:
                        sa_pos_mask = (sa_pos_mask.bool() | sa_dcc_pos_mask.bool()).float()

            sa_nce_loss = self.sa_nce(sa_q, sa_k, sa_pos_mask, sa_temps)
            loss += sa_nce_loss
        return loss








