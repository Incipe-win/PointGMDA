import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from models.uni3d import create_uni3d
from utils_10way import *
import datasets.data_utils as d_utils
import open_clip
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict
import itertools
import models
from collections import OrderedDict
import re
from datasets.ModelNetDatasetFewShot import ModelNetFewShot
from torch.utils.data import DataLoader
from pointnet2_ops import pointnet2_utils
import MinkowskiEngine as ME
import math


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="settings of Tip-Adapter in yaml format")
    parser.add_argument("--ckpt_path", default="", help="the ckpt to test 3d zero shot")
    parser.add_argument("--shot", dest="shot", type=int, default=1, help="shots number")
    parser.add_argument("--seed", dest="seed", type=int, default=1, help="seed")
    parser.add_argument("--dbg", dest="dbg", type=float, default=0, help="debug mode")
    parser.add_argument("--model", default="create_uni3d", type=str)
    parser.add_argument("--npoints", default=2048, type=int, help="number of points used for pre-train and test.")
    parser.add_argument("--group-size", type=int, default=64, help="Pointcloud Transformer group size.")
    parser.add_argument("--num-group", type=int, default=512, help="Pointcloud Transformer number of groups.")
    parser.add_argument("--pc-encoder-dim", type=int, default=512, help="Pointcloud Transformer encoder dimension.")
    parser.add_argument(
        "--clip-model",
        type=str,
        default="RN50",
        help="Name of the vision and text backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pc-model",
        type=str,
        default="RN50",
        help="Name of pointcloud backbone to use.",
    )
    parser.add_argument("--pc-feat-dim", type=int, default=768, help="Pointcloud feature dimension.")
    parser.add_argument("--embed-dim", type=int, default=1024, help="teacher embedding dimension.")
    parser.add_argument("--evaluate_3d", action="store_true", help="eval ulip only")
    args = parser.parse_args()
    return args


def gmm(vecs, labels, n_components=2, num=5):
    mus = []
    for i in range(num):
        class_mask = labels == i
        class_data = vecs[class_mask]
        if len(class_data) < n_components:
            mu = torch.mean(class_data, dim=0)
        else:
            gmm = GMM(n_components=n_components, reg_covar=1e-3).fit(class_data)
            mu = (gmm.weights_[:, None] * gmm.means_).sum(dim=0)
        mus.append(mu)
    mus = torch.stack(mus)

    center_vecs = torch.cat([vecs[labels == i] - mus[i] for i in range(num)])
    d = center_vecs.shape[1]
    cov = (center_vecs.T @ center_vecs) / (center_vecs.shape[0] - 1)
    cov_reg = cov + torch.trace(cov) / d * torch.eye(d).cuda()  # 正则化
    cov_inv = d * torch.linalg.pinv(cov_reg)

    ps = torch.ones(num).cuda() * 1.0 / num
    W = torch.einsum("nd, dc -> cn", mus, cov_inv)
    b = ps.log() - 0.5 * torch.einsum("nd,dc,nc->n", mus, cov_inv, mus)
    return W, b


def run(
    cfg,
    train_loader_cache,
    pc_model,
    test_features,
    test_labels,
    n_components=2,
    num=5,
):
    device = test_features.device

    with torch.no_grad():
        vecs = []
        labels = []
        for i in range(cfg["augment_epoch"]):
            for pc, target in tqdm(train_loader_cache):
                pc, target = pc.cuda(), target.cuda()
                fps_idx = pointnet2_utils.furthest_point_sample(pc, 1200)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(1200, 1024, False)]
                pc = (
                    pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx)
                    .transpose(1, 2)
                    .contiguous()
                )  # (B, N, 3)
                # print(f"pc1 shape: {pc.shape}")
                pc[:, [1, 2]] = pc[:, [2, 1]]
                rgb = torch.ones_like(pc) * 0.4
                rgb = rgb.cuda()
                feat = torch.cat((pc, rgb), dim=-1)
                pc_features = get_model(pc_model).encode_pc(feat)
                pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
                vecs.append(pc_features)
                labels.append(target)
        vecs = torch.cat(vecs)
        labels = torch.cat(labels)

    pca = PCA(n_components=0.95, random_state=42)
    pca.fit(vecs.detach().cpu().numpy())

    vecs = torch.tensor(pca.transform(vecs.detach().cpu().numpy()), device=device)
    test_features = torch.tensor(pca.transform(test_features.detach().cpu().numpy()), device=device)

    vecs = vecs / vecs.norm(dim=-1, keepdim=True)
    test_features = test_features / test_features.norm(dim=-1, keepdim=True)

    W, b = gmm(vecs, labels, n_components, num)

    test_logits = test_features.float() @ W + b
    notune_acc = cls_acc(test_logits, test_labels)
    print("Nonetune acc:", notune_acc)
    return notune_acc


def evaluate_class_set(cfg, pc_model, shots, way, fold):
    train_dataset = ModelNetFewShot("./data/ModelNetFewshot", 8192, False, 40, "train", way, shots, fold)
    test_dataset = ModelNetFewShot("./data/ModelNetFewshot", 8192, False, 40, "test", way, shots, fold)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    test_features, test_labels = pre_load_features(cfg, "test", pc_model, test_dataloader)

    acc = run(cfg, train_loader, pc_model, test_features, test_labels, num=way)
    return acc


def main():
    # Load config file
    args = get_arguments()
    assert os.path.exists(args.config)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # Load cfg for conditional prompt.
    print("\nRunning configs.")
    print(cfg, "\n")

    model = create_uni3d(args).cuda()
    checkpoint = torch.load(args.ckpt_path, weights_only=False)
    sd = checkpoint["module"]
    if next(iter(sd.items()))[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results = {10: [], 20: []}

    way = 10
    shot_list = [10, 20]
    for shots in shot_list:
        cfg["shots"] = shots
        for trial in range(10):
            acc = evaluate_class_set(cfg, model, shots, way, trial)
            results[shots].append(acc)

    for shots in shot_list:
        print(f"\n{way}-way {shots}-shot:")
        accs = torch.tensor(results[shots])
        mean = accs.mean().item()
        std = accs.std().item()

        std = accs.std().item()
        n = accs.size(0)

        margin_error = 1.96 * (std / math.sqrt(n))
        lower = mean - margin_error
        upper = mean + margin_error

        print(f"{shots}-shot: {mean:.2f}±{margin_error:.2f}")


if __name__ == "__main__":
    main()
