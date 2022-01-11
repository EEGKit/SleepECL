import numpy as np
import torch
import random



def normal_scaling(x, std=0.01):
    """
    make sure to pass the clone of x, or the original x with be modified.
    apply normal distribution scaling using given std.
    input: x, shape: batch_size* seq_len* num_ch* time
    """
    x.copy_(x * (1 + torch.normal(mean=0, std=std, size=x.size()).to(x.device)))

def hard_scaling(x, scale_range=0.05):
    """
    make sure to pass the clone of x, or the original x with be modified.
        apply normal distribution scaling using given std.
        input: x, shape: batch_size* seq_len* num_ch* time
    """
    low_ratio = 1 - scale_range
    up_ratio = 1 + scale_range
    ratio = low_ratio + (up_ratio - low_ratio) * float(torch.rand(1))
    x.copy_(x * ratio)

def time_mask(x, up_mask_percent=0.15):
    """
    make sure to pass the clone of x, or the original x with be modified.
    input: x, shape: batch_size* seq_len* num_ch* time
    """
    st = int(torch.randint(0, x.shape[-1], size=(1,)))
    mask_len = int(torch.rand(1) * x.shape[-1] * up_mask_percent)  # 3000 * 0.1 = 300
    x[:, :, :, st: st+mask_len] = 0



def rand_aug(x, n_augs=1):
    aug_list = ['normal_scaling', 'hard_scaling', 'time_mask']
    augs = random.sample(aug_list, n_augs)
    for aug in augs:
        f = eval(aug)
        f(x)
        # print(aug)
        # print(x)







if __name__ == '__main__':
    x = torch.ones(size=(2, 1, 1, 20))
    rand_aug(x.clone(), n_augs=3)


