import os
import random
import argparse
import yaml
import torch
from utils import *
from collections import OrderedDict
from PointGMDA import PointGMDA
from torch.utils.data import DataLoader
from datasets import build_dataset
from datasets.utils import build_data_loader
import numpy as np
import models
import re
import MinkowskiEngine as ME


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="settings of Tip-Adapter in yaml format")
    parser.add_argument("--shot", dest="shot", type=int, default=1, help="shots number")
    parser.add_argument("--seed", dest="seed", type=int, default=1, help="seed")
    parser.add_argument("--ckpt_path", default="", help="the ckpt to test 3d zero shot")
    args = parser.parse_args()
    return args


def main():
    # Load config file
    args = get_arguments()
    assert os.path.exists(args.config)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg["shots"] = args.shot
    cfg["seed"] = args.seed
    print("shots", cfg["shots"])
    print("seed", cfg["seed"])

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    cache_dir = os.path.join(f'./caches/{cfg["seed"]}/{cfg["dataset"]}')
    os.makedirs(cache_dir, exist_ok=True)
    cfg["cache_dir"] = cache_dir
    print(cfg)

    # Prepare dataset
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Textual features
    print("\nGetting textual features as CLIP's classifier.")
    clip_weights_cupl_all = torch.load(cfg["cache_dir"] + "/text_weights_cupl_t_all.pt", weights_only=False)
    cate_num, prompt_cupl_num, dim = clip_weights_cupl_all.shape
    print(f"cate_num is {cate_num}, prompt_cupl_num is {prompt_cupl_num}, dim is {dim}")
    clip_weights_cupl = clip_weights_cupl_all.mean(dim=1).t()
    clip_weights_cupl = clip_weights_cupl / clip_weights_cupl.norm(dim=0, keepdim=True)

    # Construct the cache model by few-shot training set
    print("\nConstructing cache model by few-shot visual features and labels.")
    cache_keys, cache_values = load_few_shot_feature(cfg)

    # Pre-load val features
    print("\nLoading visual features and labels from val set.")
    val_features, val_labels = loda_val_test_feature(cfg, "val")

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    if cfg["dataset"] == "objaverse":
        test_features, test_labels = loda_val_test_feature(cfg, "val")
    else:
        test_features, test_labels = loda_val_test_feature(cfg, "test")

    metric = {}
    model = models.make(4).cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    checkpoint = torch.load(args.ckpt_path, weights_only=False)
    model_dict = OrderedDict()
    pattern = re.compile("module.")
    for k, v in checkpoint["state_dict"].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, "", k)] = v
    model.load_state_dict(model_dict)
    model.eval()

    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])
    train_loader_F = build_data_loader(
        data_source=dataset.train_x,
        batch_size=128,
        is_train=True,
        shuffle=False,
    )

    acc_free = PointGMDA(
        cfg,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights_cupl,
        model,
        train_loader_F,
    )
    metric["PointGMDA"] = acc_free
    print(f"\nPointGMDA: {metric['PointGMDA']:.4f}")


if __name__ == "__main__":
    main()
