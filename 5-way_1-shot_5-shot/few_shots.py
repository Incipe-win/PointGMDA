import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from utils_5way import *
import open_clip
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict
from scipy import stats
import itertools
import models
import re
from collections import OrderedDict
import math


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="settings of Tip-Adapter in yaml format")
    parser.add_argument("--ckpt_path", default="", help="the ckpt to test 3d zero shot")
    args = parser.parse_args()
    return args


def gmm(vecs, labels, clip_weights, n_components=2):
    mus = []
    for i in range(clip_weights.shape[1]):
        class_mask = labels == i
        class_data = vecs[class_mask]
        if len(class_data) < n_components:
            mu = torch.mean(class_data, dim=0)
        else:
            gmm = GMM(n_components=n_components, reg_covar=1e-3).fit(class_data)
            mu = (gmm.weights_[:, None] * gmm.means_).sum(dim=0)
        mus.append(mu)
    mus = torch.stack(mus)

    center_vecs = torch.cat([vecs[labels == i] - mus[i] for i in range(clip_weights.shape[1])])
    d = center_vecs.shape[1]
    cov = (center_vecs.T @ center_vecs) / (center_vecs.shape[0] - 1)
    cov_reg = cov + torch.trace(cov) / d * torch.eye(d).cuda()  # 正则化
    cov_inv = d * torch.linalg.pinv(cov_reg)

    ps = torch.ones(clip_weights.shape[1]).cuda() * 1.0 / clip_weights.shape[1]
    W = torch.einsum("nd, dc -> cn", mus, cov_inv)
    b = ps.log() - 0.5 * torch.einsum("nd,dc,nc->n", mus, cov_inv, mus)
    return W, b


def run(
    cfg, pc_model, val_loader, results, way, shots, fold, class_names, template, clip_model, prompt_path, n_components=2
):
    with torch.no_grad():
        # Precompute CLIP weights for all possible classes
        all_clip_weights = clip_classifier(class_names, template, clip_model.float(), [prompt_path])

        for i, (x, y) in enumerate(tqdm(val_loader)):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            num = x.shape[0]
            y = y.view(-1)

            support_size = way * shots
            query_size = num - support_size

            support_y = torch.arange(way, device="cuda").repeat_interleave(shots)
            query_y = torch.arange(way, device="cuda").repeat_interleave(15)

            rgb = torch.full_like(x, 0.4)
            feat = torch.cat((x, rgb), dim=-1)
            all_features = pc_model(x, feat, device="cuda", quantization_size=0.02)
            all_features = all_features / all_features.norm(dim=-1, keepdim=True)

            support_features = all_features[:support_size]
            query_features = all_features[support_size:]

            vecs = support_features.repeat(cfg["augment_epoch"], 1)
            labels = support_y.repeat(cfg["augment_epoch"])

            pca = PCA(n_components=0.95, random_state=42)
            pca.fit(vecs.detach().cpu().numpy())

            vecs = torch.tensor(pca.transform(vecs.detach().cpu().numpy()), device="cuda")
            query_features = torch.tensor(pca.transform(query_features.detach().cpu().numpy()), device="cuda")

            class_indices = y[:support_size].unique().long()
            clip_weights = all_clip_weights[:, class_indices].cuda()
            clip_weights_np = clip_weights.detach().cpu().numpy().T
            clip_weights_reduced = pca.transform(clip_weights_np)
            clip_weights_pca = torch.tensor(clip_weights_reduced.T, device="cuda")
            clip_weights_pca = clip_weights_pca / clip_weights_pca.norm(dim=0, keepdim=True)
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)

            W, b = gmm(vecs, labels, clip_weights_pca, n_components)

            test_logits = query_features.float() @ clip_weights_pca.float() + (query_features.float() @ W + b)
            acc = cls_acc(test_logits, query_y)
            results[shots][fold].append(acc)


def evaluate_class_set(cfg, clip_model, pc_model, shots, way, fold, results):
    if cfg["dataset"] == "modelnet40":
        from Dataloader.model_net_cross_val import get_sets
        from Dataloader.model_net_cross_val import template
        from Dataloader.model_net_cross_val import class_names

        data_path = "./data/dataset/modelnet40_normal_resampled"
    else:
        from Dataloader.scanobjectnn_cross_val import get_sets
        from Dataloader.scanobjectnn_cross_val import template
        from Dataloader.scanobjectnn_cross_val import class_names

        data_path = "./data/dataset/ScanObjectNN/PB_T50_RS_nobg_txt"

    val_loader = get_sets(
        data_path=data_path,
        fold=fold,
        k_way=way,
        n_shot=shots,
        query_num=15,
        data_aug=True,
    )

    # Load prompts
    if cfg["dataset"] == "modelnet40":
        prompt_path = "./prompt/modelnet40.json"
    elif cfg["dataset"] == "scanobjectnn":
        prompt_path = "./prompt/scanobjectnn.json"

    # print(f"classnames: {val_dataset.classnames}")
    # clip_weights = clip_classifier(val_dataset.classnames, template, clip_model.float(), [prompt_path])

    run(cfg, pc_model, val_loader, results, way, shots, fold, class_names, template, clip_model, prompt_path)


def main():
    # Load config file
    args = get_arguments()
    assert os.path.exists(args.config)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # Load cfg for conditional prompt.
    print("\nRunning configs.")
    print(cfg, "\n")

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k", cache_dir="./ckpt"
    )
    clip_model.cuda().eval()

    model = models.make(1).cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    checkpoint = torch.load(args.ckpt_path, weights_only=False)
    model_dict = OrderedDict()
    pattern = re.compile("module.")
    for k, v in checkpoint["state_dict"].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
    model.load_state_dict(model_dict)
    model.eval()

    if cfg["dataset"] == "modelnet40":
        k_fold = 4
    else:
        k_fold = 3
    results = {1: {i: [] for i in range(k_fold)}, 5: {i: [] for i in range(k_fold)}}
    way = 5
    shot_list = [1, 5]
    for shots in shot_list:
        cfg["shots"] = shots
        for fold in range(k_fold):
            print(f"Evaluating fold {fold + 1} for {cfg['dataset']} with {shots}-shot setting.")
            evaluate_class_set(cfg, clip_model, model, shots, way, fold, results)

    print(f"\nResults for {cfg['dataset']}:")
    for shots in [1, 5]:
        print(f"\n5-way {shots}-shot:")
        for group_idx in range(k_fold):
            accs = torch.tensor(results[shots][group_idx])
            mean = accs.mean().item()
            std = accs.std().item()
            n = accs.size(0)

            margin_error = 1.96 * (std / math.sqrt(n))
            lower = mean - margin_error
            upper = mean + margin_error

            print(f"Group {group_idx+1}: {mean:.2f}±{margin_error:.2f}")


if __name__ == "__main__":
    main()
