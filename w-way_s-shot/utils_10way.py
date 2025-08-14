from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
import clip
import numpy as np
from torch.distributions import MultivariateNormal
import os
import MinkowskiEngine as ME


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def clip_classifier(classnames, template, clip_model, prompt_paths=list()):
    prompts = []
    for prompt_path in prompt_paths:
        f = open(prompt_path)
        prompts.append(json.load(f))
    with torch.no_grad():
        clip_weights = []
        min_len = 1000

        for classname in classnames:
            # Tokenize the prompts
            texts = [t.format(classname) for t in template]

            for prompt in prompts:
                texts += prompt[classname]

            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


def pre_load_features(cfg, split, pc_model, loader, norm=True):
    features, labels = [], []

    with torch.no_grad():
        for i, (pc, target) in enumerate(tqdm(loader)):
            pc, target = pc.cuda(), target.cuda()
            pc[:, [1, 2]] = pc[:, [2, 1]]
            rgb = torch.ones_like(pc) * 0.4
            rgb = rgb.cuda()
            feat = torch.cat((pc, rgb), dim=-1)
            pc_features = get_model(pc_model).encode_pc(feat)
            if norm:
                pc_features /= pc_features.norm(dim=-1, keepdim=True)
            features.append(pc_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)
    return features, labels


def build_cache_model(cfg, pc_model, train_loader_cache):
    cache_keys = []
    cache_values = []

    with torch.no_grad():
        # Data augmentation for the cache model
        for augment_idx in range(cfg["augment_epoch"]):
            train_features = []

            print("Augment Epoch: {:} / {:}".format(augment_idx, cfg["augment_epoch"]))
            for i, (pc, target, rgb) in enumerate(tqdm(train_loader_cache)):
                pc, rgb = pc.cuda(), rgb.cuda()
                feature = torch.cat((pc, rgb), dim=-1)
                pc_features = get_model(pc_model).encode_pc(feature)
                train_features.append(pc_features)
                if augment_idx == 0:
                    target = target.cuda()
                    cache_values.append(target)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

    return cache_keys, cache_values


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar  # 正则化系数

    def fit(self, X):
        n_samples, n_features = X.shape
        device = X.device

        # 初始化参数（添加随机扰动）
        self.weights_ = torch.ones(self.n_components, device=device) / self.n_components
        self.means_ = X[torch.randperm(n_samples)[: self.n_components]] + 1e-6 * torch.randn(
            self.n_components, n_features, device=device
        )
        self.covariances_ = torch.stack(
            [
                torch.eye(n_features, device=device) + self.reg_covar * torch.randn(n_features, device=device)
                for _ in range(self.n_components)
            ]
        )

        prev_lower_bound = -np.inf

        for _ in range(self.max_iter):
            # E-step: 计算对数概率（数值稳定版本）
            log_prob = []
            for k in range(self.n_components):
                try:
                    # 添加正则化确保协方差正定
                    cov = self.covariances_[k] + self.reg_covar * torch.eye(n_features, device=device)
                    mvn = MultivariateNormal(self.means_[k], cov)
                    log_prob_k = mvn.log_prob(X)
                except ValueError:
                    # 如果仍然失败，使用对角线近似
                    log_prob_k = -0.5 * torch.sum(
                        (X - self.means_[k]) ** 2 / (torch.diag(self.covariances_[k]) + self.reg_covar), dim=1
                    )
                log_prob.append(log_prob_k + torch.log(self.weights_[k]))

            weighted_logprob = torch.stack(log_prob, dim=1)
            log_resp = weighted_logprob - torch.logsumexp(weighted_logprob, dim=1, keepdim=True)
            resp = torch.exp(log_resp)

            # M-step: 更新参数
            Nk = resp.sum(dim=0) + 1e-10  # 防止除零
            self.weights_ = Nk / n_samples
            self.means_ = (resp.T @ X) / Nk[:, None]

            # 更新协方差矩阵（添加正则化）
            diff = X[:, None, :] - self.means_
            covs = torch.einsum("nk,nki,nkj->kij", resp, diff, diff) / Nk[:, None, None]
            self.covariances_ = covs + self.reg_covar * torch.eye(n_features, device=device)[None, :, :]

            # 检查收敛
            current_lower_bound = (resp * weighted_logprob).sum()
            if current_lower_bound - prev_lower_bound < self.tol:
                break
            prev_lower_bound = current_lower_bound

        return self
