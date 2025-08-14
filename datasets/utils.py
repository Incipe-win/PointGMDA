import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image
import numpy as np


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError("No file exists at {}".format(path))

    while True:
        try:
            img = Image.open(path).convert("RGB")
            return img
        except IOError:
            print("Cannot read image from {}, " "probably due to heavy IO. Will re-try".format(path))


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".") and "sh" not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=-1, classname="", rgb=""):
        # assert isinstance(impath, str)
        # assert isinstance(label, int)
        # assert isinstance(domain, int)
        # assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname
        self._rgb = rgb

    @property
    def rgb(self):
        return self._rgb

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """

    dataset_dir = ""  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError("Input domain must belong to {}, " "but got [{}]".format(self.domains, domain))

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print("Extracting file ...")

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=True):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False, return_img0=False, k_tfm=1):
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        if self.k_tfm > 1 and transform is None:
            raise ValueError("Cannot augment the image {} times " "because transform is None".format(self.k_tfm))

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {"label": item.label, "domain": item.domain, "impath": item.impath, "rgb": item.rgb}
        return output["impath"], output["label"], output["rgb"]

    def rotate_point_cloud(batch_data):
        """Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
        """batch_pc: BxNx3"""
        for b in range(batch_pc.shape[0]):
            dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
            drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
            if len(drop_idx) > 0:
                batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
        return batch_pc

    def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
        """Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
        """
        B, N, C = batch_data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            batch_data[batch_index, :, :] *= scales[batch_index]
        return batch_data

    def shift_point_cloud(batch_data, shift_range=0.1):
        """Randomly shift point cloud. Shift is per point cloud.
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, shifted batch of point clouds
        """
        B, N, C = batch_data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
        for batch_index in range(B):
            batch_data[batch_index, :, :] += shifts[batch_index, :]
        return batch_data

    def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
        """Randomly jitter points. jittering is per point.
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, jittered batch of point clouds
        """
        B, N, C = batch_data.shape
        assert clip > 0
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        jittered_data += batch_data
        return jittered_data

    def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
        """Randomly perturb the point clouds by small rotations
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
            Rx = np.array(
                [[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]]
            )
            Ry = np.array(
                [[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]]
            )
            Rz = np.array(
                [[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]]
            )
            R = np.dot(Rz, np.dot(Ry, Rx))
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
        return rotated_data


def build_data_loader(
    data_source=None, batch_size=64, input_size=224, tfm=None, is_train=True, shuffle=False, dataset_wrapper=None
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()),
        # multiprocessing_context=multiprocessing.get_context("spawn"),
    )
    assert len(data_loader) > 0

    return data_loader
