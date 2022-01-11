import argparse
from utils.coding_utils import print_args, AverageMeter
from datatools.subject_split import split_train_test_files, mass_ss3_ids
from datatools.dataset import SleepDataset
from datatools.augmentation import rand_aug
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time
from backbone.TSTCC_CNN import TSTCCCNN

from backbone.simclr_context_nonoverlap_skip import SimCLRContextNonOverlapSkip
from backbone.self_attention import TransformerEncoder


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='/data/ZhangHongjun/codes/sleep/openpai/tstcc_sleepedf/sleepEDF20_fpzcz')
    parser.add_argument('--feature_path', type=str, default='/data/ZhangHongjun/codes/sleep/openpai/tstcc_sleepedf/feature/eeg_fpz_cz_powerband_5.pkl')
    parser.add_argument('--dataset', type=str, default='sleepedf')
    parser.add_argument('--fs', type=int, default=100)

    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fold_idx", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=9)
    parser.add_argument("--sub_window_size", type=int, default=3)
    parser.add_argument("--step", type=int, default=3)

    parser.add_argument("--cuda", type=int, default=0)

    parser.add_argument('--norm', type=str, default="none")
    parser.add_argument("--aug", action='store_false')  # 默认为True
    parser.add_argument("--debug", action='store_true')  # 默认为False
    parser.add_argument("--scaling_std", type=float, default=0.01)

    # simclr params
    parser.add_argument('--n_epoch', type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)

    # transformer params
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=2)


    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='Only valid for SGD optimizer')
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument('--T', type=float, default=0.07)  # temperature
    parser.add_argument("--sep_pos", action='store_true', help="separate positive in SupContrast")  # 默认为False

    parser.add_argument('--cnn_level',  nargs='+', type=str, default=['dcc', 'prior'], help="['unbiased', 'prior', 'dcc']") #
    parser.add_argument('--sa_level',  nargs='+', type=str, default=['dcc', 'prior'], help="['unbiased', 'prior', 'dcc']")
    parser.add_argument('--down_linear',  action='store_true')

    # params in prior.
    parser.add_argument("--topK_ratio", type=float, default=0.4, help='prior knowledge pick topK most similar sample within a mini-batch')
    parser.add_argument('--metric', type=str, default='euc', choices=['euc', 'std_euc', 'mah'])

    # downstream task
    parser.add_argument("--n_ft_epochs", type=int, default=40)
    parser.add_argument("--ft_batch_size", type=int, default=128)
    parser.add_argument("--not_freeze", action='store_true')

    parser.add_argument('--ntimes', type=int, default=5)  # default 100


    args_parsed = parser.parse_args()
    print_args(parser, args_parsed)

    return args_parsed



class CnnEncoder(nn.Module):
    """treat all input as sequence. specially, one sample sequence length is 1.
    input: batch_size, seq_len, ch, time
    output: batch_size, seq_len, num_classes
    """
    def __init__(self, feature_dim=128, fs=100, pretrain=True, down_linear=False):
        super(CnnEncoder, self).__init__()
        self.pretrain = pretrain
        self.down_linear = down_linear

        cnn_dim = 16256 if fs == 100 else 15616
        kernel_size = 25 if fs == 100 else 64
        stride = 3 if fs == 100 else 8
        self.cnn = TSTCCCNN(kernel_size=kernel_size, stride=stride)

        if down_linear:
            self.down = nn.Linear(cnn_dim, feature_dim)
            # self.cnn_mlp = nn.Sequential(nn.Linear(cnn_dim, cnn_dim), nn.ReLU(), nn.Linear(cnn_dim, feature_dim))
            self.cnn_mlp = nn.Linear(feature_dim, feature_dim)
            self.cnn_to_context_linear = nn.Linear(feature_dim, feature_dim)
        else:
            self.cnn_mlp = nn.Linear(cnn_dim, feature_dim)
            self.cnn_to_context_linear = nn.Linear(cnn_dim, feature_dim)

    def forward(self, x):
        batch_size, seq_len, *_shape = x.shape
        x = x.view(batch_size*seq_len, *_shape)  # shape [bsz*seq_len, 1, time]
        x = self.cnn(x)  # shape [bsz*seq_len, n_channel, feature_len]
        x = x.view(batch_size*seq_len, -1) # shape [bsz*seq_len, n_channel*feature_len]
        if self.down_linear:
            x = self.down(x)

        cnn_out = self.cnn_mlp(x)  # shape [bsz*seq_len, feature_dim]
        cnn_out = cnn_out.view(batch_size, seq_len,
                               -1)  # shape [bsz, seq_len, feature_dim], for instance discrimination
        sa_embedding = self.cnn_to_context_linear(x)  # embedding for self-attention,shape [bsz*seq_len, feature_dim]
        sa_embedding = sa_embedding.reshape(shape=(batch_size, seq_len, -1))  # [bsz, seq_len, feature_dim]

        if self.pretrain:
            return cnn_out, sa_embedding
        else:

            x = x.reshape(batch_size, seq_len, -1)  # [bsz, seq_len, n_channel*feature_len]
            x = x[:, seq_len//2, :]  # [bsz, n_channel*feature_len]
            return x, sa_embedding


class ContextEncoder(nn.Module):
    def __init__(self, seq_len=3, feature_dim=128, n_head=1, n_layer=1, num_classes=5, pretrain=True):
        super(ContextEncoder, self).__init__()
        self.pretrain = pretrain
        encoder_ffn_embed_dim = feature_dim // 2 * 3
        self.context = TransformerEncoder(encoder_embed_dim=feature_dim, encoder_ffn_embed_dim=encoder_ffn_embed_dim,
                                          encoder_attention_heads=n_head, layer_norm_first=False, encoder_layers=n_layer,
                                          seq_len=seq_len)
        self.nce_fc = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim))
        self.seq_fc = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 2))

    def forward(self, sa_embedding):
        batch_size, seq_len, feature_dim = sa_embedding.shape
        int_embedding = self.context(sa_embedding)  # integrated embedding. [bsz, seq_len, feature_dim]
        int_embedding = int_embedding[:, seq_len//2, :]  # [bsz, feature_dim]
        if self.pretrain:
            nce_out = self.nce_fc(int_embedding)   # [bsz, feature_dim]
            seq_out = self.seq_fc(int_embedding)   # [bsz, 2]
            return nce_out, seq_out
        else:
            return int_embedding



CE_loss = nn.CrossEntropyLoss()


def main(args):
    if args.dataset == 'sleepedf':
        fix_train_sids, fix_valid_sids, fix_test_sids = [14, 5, 4, 17, 8, 7, 19, 12, 0, 15, 16, 9], [11, 10, 3, 1], [6,
                                                                                                                     18,
                                                                                                                     2,
                                                                                                                     13]
    elif args.dataset == 'mass_ss3':
        fix_train_sids, fix_valid_sids, fix_test_sids = mass_ss3_ids[:int(len(mass_ss3_ids) * 0.6)], mass_ss3_ids[int(len(mass_ss3_ids) * 0.8):], mass_ss3_ids[int(
            len(mass_ss3_ids) * 0.6):int(len(mass_ss3_ids) * 0.8)]
    train_files, valid_files, train_sids, valid_sids = split_train_test_files(args.data_dir, args.dataset, args.n_folds,
                                                                              args.fold_idx, return_id=True,
                                                                              fix_train_sids=fix_train_sids,
                                                                              fix_test_sids=fix_valid_sids)
    train_files, test_files, train_sids, test_sids = split_train_test_files(args.data_dir, args.dataset, args.n_folds,
                                                                            args.fold_idx, return_id=True,
                                                                            fix_train_sids=fix_train_sids,
                                                                            fix_test_sids=fix_test_sids)


    train_set = SleepDataset(train_files, window_size=args.window_size, label_idx=args.window_size//2, dataset=args.dataset, norm=args.norm, use_feature=True, feature_path=args.feature_path)

    valid_set = SleepDataset(valid_files, window_size=args.window_size, label_idx=args.window_size//2, dataset=args.dataset, norm=args.norm, use_feature=True, feature_path=args.feature_path)
    test_set = SleepDataset(test_files, window_size=args.window_size, label_idx=args.window_size//2, dataset=args.dataset, norm=args.norm, use_feature=True, feature_path=args.feature_path)
    n_sample = len(train_set) + +len(valid_set) + len(test_set)
    print('n_sample', n_sample)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SimCLRContextNonOverlapSkip(cnn_encoder=CnnEncoder, context_encoder=ContextEncoder, mlp=True, args=args).cuda()

    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.sgd_momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError

    for epoch in range(args.n_epoch):
        if args.aug:
            train_set.rolling()
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        train_losses = AverageMeter('trainLoss', ':.5f')
        model.train()
        tic = time.time()
        # for train_x, train_y, feature in train_loader:
        for _ret in train_loader:
            train_x, train_y, feature = _ret[0].cuda(), _ret[1].cuda(), _ret[2].cuda()

            x1, x2 = train_x.clone(), train_x.clone()
            rand_aug(x1, n_augs=1)
            rand_aug(x2, n_augs=1)
            loss = model(train_x, x1, x2, args, feature, train_y)  # 64,3, 1, 3000
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.update(loss.item(), train_x.size(0))
        print(f"[e{epoch}/{args.n_epoch}] TR l={train_losses.avg:.4f}  ({time.time() - tic:.1f}s)")
    test_acc, test_f1 = finetune(model, train_set, valid_loader, test_loader, args)



def finetune(simclr_net, train_set, valid_loader, test_loader, args):
    cnn_encoder = CnnEncoder(feature_dim=args.feature_dim, fs=args.fs, pretrain=False, down_linear=args.down_linear).cuda()
    context_encoder = ContextEncoder(seq_len=args.sub_window_size, feature_dim=args.feature_dim, n_head=args.n_head, n_layer=args.n_layer, num_classes=5, pretrain=False).cuda()
    if args.fs == 100:
        rep_dim = 256 if args.down_linear else 16384
    else:
        rep_dim = 256 if args.down_linear else 15744
    linear_clr = nn.Linear(rep_dim, 5).cuda()
    if not args.not_freeze:
        print("freeze the pretrain model")
        for name, param in cnn_encoder.named_parameters():
            param.requires_grad = False

    state_dict = simclr_net.state_dict()
    for k in list(state_dict.keys()):
        if k.startswith('cnn_encoder'):
            # remove prefix
            state_dict[k[len("cnn_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = cnn_encoder.load_state_dict(state_dict, strict=False)
    print(msg.missing_keys)
    print(set(msg.missing_keys))
    assert set(msg.missing_keys) == set()
    print("=> loaded cnn pre-trained cnn model ")
    if not args.not_freeze:
        print("freeze the pretrain sa model")
        for name, param in context_encoder.named_parameters():
                param.requires_grad = False
    # init the fc layer
    linear_clr.weight.data.normal_(mean=0.0, std=0.01)
    linear_clr.bias.data.zero_()
    # load cnn_encoder from pre-trained
    state_dict = simclr_net.state_dict()
    for k in list(state_dict.keys()):
        if k.startswith('context_encoder') and not k.startswith('context_encoder.class_fc'):
            # remove prefix
            state_dict[k[len("context_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = context_encoder.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == set()
    print("=> loaded pre-trained context model ")

    if args.optimizer == 'adam':
        optimizer = Adam([{'params': cnn_encoder.parameters()}, {'params': context_encoder.parameters()}, {'params': linear_clr.parameters()}],
                         lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    else:
        raise ValueError
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)



    for epoch in range(args.n_ft_epochs):
        if args.aug:
            train_set.rolling()
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_losses = AverageMeter('trainLoss', ':.5f')

        cnn_encoder.train()
        context_encoder.train()
        linear_clr.train()

        tic = time.time()
        preds, trues = [], []
        for train_x, train_y, *_ in train_loader:
            train_x, train_y = train_x.cuda(), train_y.cuda()[:, args.window_size//2]
            x1 = train_x.clone()
            rand_aug(x1, n_augs=1)
            cnn_linear_in, sa_in = cnn_encoder(x1)  # sa_in [bsz, seq_len, feature_dim]
            # train_y_hat = context_encoder(sa_in)  # shape[bsz, 5], coresponding the label at label_idx


            sa_input = sa_in.unfold(dimension=1, size=args.sub_window_size, step=args.step).permute(0, 1, 3, 2)
            bsz, num_sub_seq, sub_seq_len, cnn_feature_dim = sa_input.shape
            sa_input = sa_input[:, num_sub_seq//2, :, :]  # bsz, sub_seq_len, cnn_feature_dim
            sa_linear_in = context_encoder(sa_input)
            linear_in = torch.cat([cnn_linear_in, sa_linear_in], dim=1)
            train_y_hat = linear_clr(linear_in)



            ce_loss = CE_loss(train_y_hat, train_y)
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            train_losses.update(ce_loss.item(), train_x.size(0))
            preds.extend(np.argmax(train_y_hat.cpu().detach().numpy(), axis=1))
            trues.extend(train_y.cpu().numpy())
        valid_losses, valid_acc, valid_f1, valid_time = test(cnn_encoder, context_encoder,linear_clr, valid_loader, CE_loss)
        test_losses, test_acc, test_f1, test_time = test(cnn_encoder, context_encoder, linear_clr, test_loader, CE_loss)
        scheduler.step(valid_losses.avg)

        print(
            f"[e{epoch}/{args.n_ft_epochs}] TR l={train_losses.avg:.4f} a={accuracy_score(trues, preds):.4f} f1={f1_score(trues, preds, average='macro'):.4f} ({time.time() - tic:.1f}s)"
            f"| VA  l={valid_losses.avg:.4f} a={valid_acc:.4f} f1={valid_f1:.4f} ({valid_time:.1f}s)"
            f"| TE  l={test_losses.avg:.4f} a={test_acc:.4f} f1={test_f1:.4f} ({test_time:.1f}s)")
    return test_acc, test_f1


def test(cnn_encoder, context_encoder, linear_clr, test_loader, loss_func):
    cnn_encoder.eval()
    context_encoder.eval()
    linear_clr.eval()
    tic = time.time()
    test_losses = AverageMeter('trainLoss', ':.5f')
    preds, trues = [], []
    with torch.no_grad():
        for test_x, test_y, *_ in test_loader:
            test_x, test_y = test_x.cuda(), test_y.cuda()[:, args.window_size//2]
            cnn_linear_in, sa_in = cnn_encoder(test_x)

            sa_input = sa_in.unfold(dimension=1, size=args.sub_window_size, step=args.step).permute(0, 1, 3, 2)
            bsz, num_sub_seq, sub_seq_len, cnn_feature_dim = sa_input.shape
            sa_input = sa_input[:, num_sub_seq // 2, :, :]  # bsz, sub_seq_len, cnn_feature_dim
            sa_linear_in = context_encoder(sa_input)  # [bsz, 5]
            linear_in = torch.cat([cnn_linear_in, sa_linear_in], dim=1)
            test_y_hat = linear_clr(linear_in)


            # test_y_hat = context_encoder(sa_in)  # shape[bsz, 5], coresponding the label at label_idx
            loss = loss_func(test_y_hat, test_y)
            test_losses.update(loss.item(), test_x.size(0))
            preds.extend(np.argmax(test_y_hat.cpu().detach().numpy(), axis=1))
            trues.extend(test_y.cpu().numpy())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    return test_losses, acc, f1, time.time()-tic




if __name__ == '__main__':
    args = parse_args()
    main(args)
