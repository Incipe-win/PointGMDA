"""
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
"""

import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import random

warnings.filterwarnings("ignore")


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ModelNetFewShot(Dataset):
    def __init__(self, data_path, n_points, use_normals, num_category, subset, way, shot, fold):
        self.root = data_path
        self.npoints = n_points
        self.use_normals = use_normals
        self.num_category = num_category
        self.process_data = True
        self.uniform = True
        split = subset
        self.subset = subset

        self.way = way
        self.shot = shot
        self.fold = fold
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f"{self.way}way_{self.shot}shot", f"{self.fold}.pkl")

        print("Load processed data from %s..." % self.pickle_path)

        with open(self.pickle_path, "rb") as f:
            self.dataset = pickle.load(f)[self.subset]

        print("The size of %s data is %d" % (split, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, _ = self.dataset[index]

        points[:, 0:3] = pc_normalize(points[:, 0:3])
        if not self.use_normals:
            points = points[:, 0:3]

        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == "train":
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return current_points, label
