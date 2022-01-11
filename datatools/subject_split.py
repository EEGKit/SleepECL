import os
import glob
import numpy as np
import re

edf_sub_ids = np.array([i for i in range(20)])
edfx_sub_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 70, 71, 72, 73, 74, 75, 76, 77, 80, 81, 82])
mass_ss3_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                         27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52,
                         53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64])


def split_train_test_files(data_dir, dataset, n_folds, fold_idx, return_id=False, fix_train_sids=None, fix_test_sids=None):
    subject_files = glob.glob(os.path.join(data_dir, "*.npz"))
    # Split training and test sets
    if dataset == 'sleepedf':
        sids = edf_sub_ids
    elif dataset == 'sleepedfx':
        sids = edfx_sub_ids
    elif dataset == 'mass_ss3':
        sids = mass_ss3_ids

    fold_pids = np.array_split(sids, n_folds)
    test_sids = fold_pids[fold_idx]
    train_sids = np.setdiff1d(sids, test_sids)
    if fix_train_sids is not None and fix_test_sids is not None:
        train_sids = fix_train_sids
        test_sids = fix_test_sids

    print("Train SIDs: ({}) {}".format(len(train_sids), train_sids))
    print("Test SIDs: ({}) {}".format(len(test_sids), test_sids))
    # Get corresponding files
    train_files = np.hstack([get_subject_files(dataset=dataset, files=subject_files, sid=sid) for sid in train_sids])
    test_files = np.hstack([get_subject_files(dataset=dataset, files=subject_files, sid=sid)for sid in test_sids])
    if return_id:
        return train_files, test_files, train_sids, test_sids
    return train_files, test_files




def get_subject_files(dataset, files, sid):
    if "mass" in dataset:
        reg_exp = f"{str(sid).zfill(2)}.npz"
    elif "sleepedf" in dataset:
        reg_exp = f"S[C|T][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
    elif "isruc" in dataset:
        reg_exp = f"subject{sid+1}.npz"
    else:
        raise Exception("Invalid datasets.")
    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
    return subject_files


def get_sids(dataset):
    if dataset == 'sleepedf':
        return edf_sub_ids
