from torch.utils.data import Dataset
import numpy as np
import torch
import pickle
import os

class SleepDataset(Dataset):
    def __init__(self, file_list, window_size=0, label_idx=0, dataset='sleepedf', sequence=True, norm='zscore', verbose=True, use_feature=False, feature_path=None):
        super(SleepDataset, self).__init__()
        assert norm in ['zscore', 'none']
        self.window_size = window_size
        self.label_idx = label_idx
        self.use_feature = use_feature
        self.ni_data_list, self.ni_label_list = [], []  # ni_data_list 中的元素为一晚睡眠数据 N * C * 3000, torch.float
        self.ni_feature_list = []

        for file in file_list:
            npz_data = np.load(file)
            _x = npz_data['x']
            _x = _x.reshape(_x.shape[0], _x.shape[1])
            ni_data = torch.from_numpy(np.expand_dims(_x, 1)).float()
            if norm == 'zscore':  # normalize each night signal
                # print(f"apply zscore, mean={torch.mean(ni_data):.2f} std={torch.std(ni_data):.2f}")
                ni_data = (ni_data - torch.mean(ni_data)) / torch.std(ni_data)
            self.ni_data_list.append(ni_data)
            self.ni_label_list.append(torch.from_numpy(npz_data['y']).long())
            if verbose:
                print(f'reading file{file} {len(ni_data)} samples')

        if use_feature:
            with open(feature_path, 'rb') as f:
                feature_dict = pickle.load(f)
                for file in file_list:
                    ni_feature = torch.from_numpy(feature_dict[os.path.basename(file)])  # shape [N, feature_dim]
                    self.ni_feature_list.append(ni_feature)

        # keep ni_data_list and ni_data_list stay, and will not slide from it while doing augmentation.
        if sequence:
            self.data, self.label, self.feature = self.__slide_window(self.ni_data_list, self.ni_label_list, self.ni_feature_list)
        else:
            self.data, self.label, self.feature = torch.cat(self.ni_data_list, dim=0), torch.cat(self.ni_label_list, dim=0), torch.cat(self.ni_feature_list, dim=0)


    def __slide_window(self, ni_data_list, ni_label_list, ni_feature_list):
        data, label, feature = [], [], []
        for ni_data, ni_label in zip(ni_data_list, ni_label_list):
            data.append(ni_data.unfold(dimension=0, size=self.window_size, step=1))
            label.append(ni_label.unfold(dimension=0, size=self.window_size, step=1))

        if not self.use_feature:
            return torch.cat(data, dim=0).permute(0, 3, 1, 2), torch.cat(label, dim=0), []
        else:
            for ni_feature in ni_feature_list:
                feature.append(ni_feature.unfold(dimension=0, size=self.window_size, step=1))
            return torch.cat(data, dim=0).permute(0, 3, 1, 2), torch.cat(label, dim=0), torch.cat(feature, dim=0).permute(0, 2, 1)



    def rolling(self):
        # 1. time rolling
        aug_9data_list = rolling(self.ni_data_list, percent=0.1)
        self.data, self.label, self.feature = self.__slide_window(aug_9data_list, self.ni_label_list, self.ni_feature_list)



    def __getitem__(self, index):
        if self.use_feature:
            return self.data[index], self.label[index], self.feature[index]
        else:
            return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]


def rolling(ni_data_list, percent=0.1):
    """
    rolling one night sleep signal, return a copy of origin data list and label list
    """
    # print("apply rolling")
    aug_9data_list = []
    for ni_data in ni_data_list:
        aug_ni_data = ni_data.clone()
        offset = np.random.uniform(-percent, percent) * aug_ni_data.shape[-1]
        _n, _c, _t = aug_ni_data.shape
        assert _c == 1, "only one channel is compatible"
        # t_aug_ni_data = aug_ni_data.reshape(-1)
        roll_x = torch.roll(aug_ni_data.reshape(-1), int(offset), dims=0)
        roll_x = roll_x.reshape(_n, _c, _t)
        aug_9data_list.append(roll_x)
    return aug_9data_list