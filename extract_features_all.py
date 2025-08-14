import os
import random
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import open_clip
from utils import *
import json
from collections import OrderedDict
from tokenizer import SimpleTokenizer
from torch.utils.data import DataLoader
import models
import re
import MinkowskiEngine as ME


def extract_few_shot_feature(cfg, model, train_loader_cache, norm=True):
    cache_keys = []
    cache_values = []
    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg["augment_epoch"]):
            train_features = []
            print("Augment Epoch: {:} / {:}".format(augment_idx, cfg["augment_epoch"]))
            for i, (pc, target, rgb) in enumerate(tqdm(train_loader_cache)):
                pc, rgb = pc.cuda(), rgb.cuda()
                feat = torch.cat((pc, rgb), dim=-1)
                pc = ME.utils.batched_coordinates([x for x in pc], dtype=torch.float32)
                feat = torch.cat([x for x in feat], dim=0)
                pc_features = model(pc, feat, device="cuda", quantization_size=0.02)
                train_features.append(pc_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    if norm:
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

    if norm:
        torch.save(cache_keys, cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots.pt")
        torch.save(cache_values, cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots.pt")
    else:
        torch.save(
            cache_keys,
            cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots_unnormed.pt",
        )
        torch.save(
            cache_values,
            cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots_unnormed.pt",
        )
    return


def extract_few_shot_feature_all(cfg, model, train_loader_cache, norm=True):
    with torch.no_grad():
        # Ours
        vecs = []
        labels = []
        for i in range(cfg["augment_epoch"]):
            for pc, target, rgb in tqdm(train_loader_cache):
                pc, target, rgb = pc.cuda(), target.cuda(), rgb.cuda()
                feat = torch.cat((pc, rgb), dim=-1)
                pc = ME.utils.batched_coordinates([x for x in pc], dtype=torch.float32)
                feat = torch.cat([x for x in feat], dim=0)
                pc_features = model(pc, feat, device="cuda", quantization_size=0.02)
                if norm:
                    pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)
                vecs.append(pc_features)
                labels.append(target)
        vecs = torch.cat(vecs)
        labels = torch.cat(labels)

        if norm:
            torch.save(vecs, cfg["cache_dir"] + "/" + f"{k}_vecs_f.pt")
            torch.save(labels, cfg["cache_dir"] + "/" + f"{k}_labels_f.pt")
        else:
            torch.save(vecs, cfg["cache_dir"] + "/" + f"{k}_vecs_f_unnormed.pt")
            torch.save(labels, cfg["cache_dir"] + "/" + f"{k}_labels_f_unnormed.pt")


def extract_val_test_feature(cfg, split, model, loader, norm=True):
    features, labels = [], []
    with torch.no_grad():
        for i, (pc, target, rgb) in enumerate(tqdm(loader)):
            pc, target, rgb = pc.cuda(), target.cuda(), rgb.cuda()
            feat = torch.cat((pc, rgb), dim=-1)
            pc = ME.utils.batched_coordinates([x for x in pc], dtype=torch.float32)
            feat = torch.cat([x for x in feat], dim=0)
            pc_features = model(pc, feat, device="cuda", quantization_size=0.02)
            if norm:
                pc_features /= pc_features.norm(dim=-1, keepdim=True)
            features.append(pc_features)
            labels.append(target)
    features, labels = torch.cat(features), torch.cat(labels)
    if norm:
        torch.save(features, cfg["cache_dir"] + "/" + split + "_f.pt")
        torch.save(labels, cfg["cache_dir"] + "/" + split + "_l.pt")
    else:
        torch.save(features, cfg["cache_dir"] + "/" + split + "_f_unnormed.pt")
        torch.save(labels, cfg["cache_dir"] + "/" + split + "_l_unnormed.pt")
    return


def extract_text_feature_all(cfg, classnames, prompt_paths, clip_model, template, norm=True):
    tokenizer = SimpleTokenizer()
    prompts = []
    for prompt_path in prompt_paths:
        f = open(prompt_path)
        prompts.append(json.load(f))
    with torch.no_grad():
        clip_weights = []
        min_len = 1000
        for classname in classnames:
            # Tokenize the prompts
            template_texts = [t.format(classname) for t in template]

            texts = template_texts
            for prompt in prompts:
                texts += prompt[classname]

            texts_token = tokenizer(texts).cuda()
            # texts_token = tokenizer(texts)
            # texts_token = clip.tokenize(texts, truncate=True).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts_token)
            if norm:
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            min_len = min(min_len, len(class_embeddings))
            clip_weights.append(class_embeddings)

        for i in range(len(clip_weights)):
            clip_weights[i] = clip_weights[i][:min_len]

        clip_weights = torch.stack(clip_weights, dim=0).cuda()
        print(clip_weights.shape)

    if norm:
        torch.save(clip_weights, cfg["cache_dir"] + "/text_weights_cupl_t_all.pt")
    else:
        torch.save(clip_weights, cfg["cache_dir"] + "/text_weights_cupl_t_all_unnormed.pt")
    return


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default="", help="the ckpt to test 3d zero shot")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    modelnet40 = "./prompt/modelnet40.json"
    scanobjectnn = "./prompt/scanobjectnn.json"
    args = get_arguments()
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained="laion2b_s39b_b160k", cache_dir="./ckpt"
        )
        clip_model.cuda().eval()

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

        all_dataset = [
            "modelnet40",
            "scanobjectnn",
        ]
        k_shot = [1, 2, 4, 8, 16]
        norm = True

        data_path = "data"
        for set in all_dataset:
            cfg = yaml.load(open("configs/{}.yaml".format(set), "r"), Loader=yaml.Loader)

            cache_dir = os.path.join(f"./caches/{seed}", cfg["dataset"])
            os.makedirs(cache_dir, exist_ok=True)
            cfg["cache_dir"] = cache_dir
            cfg["seed"] = seed
            cfg["dataset"] = set

            for k in k_shot:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                cfg["shots"] = k
                print(cfg)
                dataset = build_dataset(set, data_path, k)
                val_loader = build_data_loader(
                    data_source=dataset.val,
                    batch_size=128,
                    is_train=False,
                    shuffle=False,
                )
                test_loader = build_data_loader(
                    data_source=dataset.test,
                    batch_size=128,
                    is_train=False,
                    shuffle=False,
                )
                train_loader_cache = build_data_loader(
                    data_source=dataset.train_x,
                    batch_size=128,
                    is_train=True,
                    shuffle=False,
                )

                # Construct the cache model by few-shot training set
                print("\nConstructing cache model by few-shot visual features and labels.")
                extract_few_shot_feature(cfg, model, train_loader_cache)
                extract_few_shot_feature_all(cfg, model, train_loader_cache, norm=norm)

            # Extract val/test features
            print("\nLoading visual features and labels from val and test set.")
            extract_val_test_feature(cfg, "val", model, val_loader, norm=norm)
            extract_val_test_feature(cfg, "test", model, test_loader, norm=norm)

            # [dataset.cupl_path, dataset.waffle_path, dataset.DCLIP_path]
            if set == "modelnet40":
                extract_text_feature_all(
                    cfg,
                    dataset.classnames,
                    [modelnet40],
                    clip_model,
                    dataset.template,
                    norm=norm,
                )
            elif set == "scanobjectnn":
                extract_text_feature_all(
                    cfg,
                    dataset.classnames,
                    [scanobjectnn],
                    clip_model,
                    dataset.template,
                    norm=norm,
                )
