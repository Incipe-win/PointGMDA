import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import os

from tqdm import tqdm
from torch.utils.data import DataLoader

np.random.seed(0)

class_names = [
    "airplane",
    "bathtub",
    "bed",
    "bench",
    "bookshelf",
    "bottle",
    "bowl",
    "car",
    "chair",
    "cone",
    "cup",
    "curtain",
    "desk",
    "door",
    "dresser",
    "flower_pot",
    "glass_box",
    "guitar",
    "keyboard",
    "lamp",
    "laptop",
    "mantel",
    "monitor",
    "night_stand",
    "person",
    "piano",
    "plant",
    "radio",
    "range_hood",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "table",
    "tent",
    "toilet",
    "tv_stand",
    "vase",
    "wardrobe",
    "xbox",
]


template = [
    "a point cloud model of {}.",
    "There is a {} in the scene.",
    "There is the {} in the scene.",
    "a photo of a {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of a {}.",
    "itap of my {}.",
    "itap of the {}.",
    "a photo of a {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of a {}.",
    "a good photo of the {}.",
    "a bad photo of a {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of a {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of a {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of a {}.",
    "a cropped photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of a {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of a {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of a {}",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]


class ModelNet40_fs(Dataset):
    def __init__(self, root, split="train", fold=0, num_point=1024, data_aug=True):
        super().__init__()
        self.root = root
        self.fold = fold
        self.split = split
        self.num_point = num_point
        self.data_aug = data_aug

        self.point_path, self.point_label = self.get_point()

    def get_point(self):
        # == will be returned later ==
        point_path_list = []
        label_list = []
        # ============================

        picked_index = np.zeros(40)
        picked_index[self.fold * 10 : (self.fold + 1) * 10] = 1

        class_list = np.arange(40)
        if self.split == "train":
            picked_index = (1 - picked_index).astype(bool)
        else:
            picked_index = picked_index.astype(bool)

        class_list = class_list[picked_index]
        for c in class_list:
            class_fold = os.path.join(self.root, str(c))
            for i in os.listdir(class_fold):
                point_path_list.append(os.path.join(class_fold, i))
                label_list.append(c)

        return point_path_list, label_list

    def __len__(self):
        return len(self.point_path)

    def translate_pointcloud(self, pointcloud):
        xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype("float32")
        return translated_pointcloud

    def __getitem__(self, index):
        point = np.loadtxt(self.point_path[index], delimiter=",").astype(np.float32)[: self.num_point]
        point = point[:, :3]
        point[:, [1, 2]] = point[:, [2, 1]]
        label = self.point_label[index]

        if self.split == "train" and self.data_aug:
            point = self.translate_pointcloud(point)
            np.random.shuffle(point)

        pointcloud = torch.FloatTensor(point)
        label = torch.LongTensor([label])
        # pointcloud = pointcloud.permute(1, 0)
        return pointcloud, label


"""
In the WACV paper
- Totoal 80 epochs used for training
- 400 training episodes and 600 validating episodes for each epoch
- For testing, episodes=700
- n_way=5. k_shot=1. query=15 for each classes

"""


class NShotTaskSampler(Sampler):
    def __init__(self, dataset, episode_num, k_way, n_shot, query_num):
        super().__init__(dataset)
        self.dataset = dataset
        self.episode_num = episode_num
        self.k_way = k_way
        self.n_shot = n_shot
        self.query_num = query_num
        self.label_set = self.get_label_set()
        self.data, self.label = self.dataset.point_path, self.dataset.point_label

    def get_label_set(self):
        point_label_set = np.unique(self.dataset.point_label)
        return point_label_set

    def __iter__(self):
        for _ in range(self.episode_num):
            support_list = []
            query_list = []
            picked_cls_set = np.random.choice(self.label_set, self.k_way, replace=False)

            # picked_classnames = [class_names[cls] for cls in picked_cls_set]
            # print("Current 5-way classes:", picked_classnames)

            for picked_cls in picked_cls_set:
                target_index = np.where(self.label == picked_cls)[0]
                picked_target_index = np.random.choice(target_index, self.n_shot + self.query_num, replace=False)

                support_list.append(picked_target_index[: self.n_shot])
                query_list.append(picked_target_index[self.n_shot :])

            s = np.concatenate(support_list)
            q = np.concatenate(query_list)

            """
            For epi_index
            - it's the index used for each batch
            - the first k_way*n_shot images is the support set
            - the last k_way*query images is for the query set 
            """
            epi_index = np.concatenate((s, q))
            # np.random.shuffle(epi_index[self.n_way:])
            yield epi_index

    def __len__(self):
        return self.episode_num


def get_sets(data_path, fold=0, k_way=5, n_shot=1, query_num=15, data_aug=True):
    # train_dataset = ModelNet40_fs(root=data_path, split="train", fold=fold, data_aug=data_aug)
    # train_sampler = NShotTaskSampler(
    #     dataset=train_dataset, episode_num=400, k_way=k_way, n_shot=n_shot, query_num=query_num
    # )
    # train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

    val_dataset = ModelNet40_fs(root=data_path, split="test", fold=fold, data_aug=data_aug)
    val_sampler = NShotTaskSampler(
        dataset=val_dataset, episode_num=700, k_way=k_way, n_shot=n_shot, query_num=query_num
    )
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    return val_loader


if __name__ == "__main__":
    root = "../data/dataset/modelnet40_normal_resampled"
    dataset = ModelNet40_fs(root)

    test_dataset, test_loader = get_sets(data_path=root)
    k_way, n_shot, query_num = 5, 1, 15
    print(len(test_loader))
    for x, y in test_loader:
        """
        x' shape is (80,3,1024)
        y's shpae is (80,1)
        """
        support_size = k_way * n_shot
        query_size = k_way * query_num

        support_x = x[:support_size]  # 支持集样本
        support_y = y[:support_size]  # 支持集标签

        query_x = x[support_size:]  # 查询集样本
        query_y = y[support_size:]  # 查询集标签

        print(f"support_x shape: {support_x.shape}, support_y shape: {support_y.shape}")
        print(f"query_x shape: {query_x.shape}, query_y shape: {query_y.shape}")
        break
