import torch
import torch.nn.functional as F
import torch.nn as nn
import os


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def load_text_feature(cfg):
    save_path = cfg["cache_dir"] + "/text_weights_gpt_t.pt"
    clip_weights = torch.load(save_path, weights_only=False)
    return clip_weights


def load_few_shot_feature(cfg, norm=True):
    if norm:
        cache_keys = torch.load(cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots.pt", weights_only=False)
        cache_values = torch.load(cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots.pt", weights_only=False)
    else:
        cache_keys = torch.load(
            cfg["cache_dir"] + "/keys_" + str(cfg["shots"]) + "shots_unnormed.pt", weights_only=False
        )
        cache_values = torch.load(
            cfg["cache_dir"] + "/values_" + str(cfg["shots"]) + "shots_unnormed.pt", weights_only=False
        )
    return cache_keys, cache_values


def loda_val_test_feature(cfg, split, norm=True):
    if norm:
        features = torch.load(cfg["cache_dir"] + "/" + split + "_f.pt", weights_only=False)
        labels = torch.load(cfg["cache_dir"] + "/" + split + "_l.pt", weights_only=False)
    else:
        features = torch.load(cfg["cache_dir"] + "/" + split + "_f_unnormed.pt", weights_only=False)
        labels = torch.load(cfg["cache_dir"] + "/" + split + "_l_unnormed.pt", weights_only=False)
    return features, labels


# t_features [c,p,d]
# s_features [c,n,d] or [c,d]
def image_guide_text(cfg, t_features, s_features, gamma=-1, return_weights=False, return_matching=False):
    t_features = t_features / t_features.norm(dim=-1, keepdim=True)

    if gamma == -1:
        if cfg["dataset"] == "modelnet40":
            gamma = 1
        elif cfg["dataset"] == "objaverse":
            gamma = 100
        else:
            gamma = 50

    if len(s_features.shape) == 3:
        s_features = s_features.mean(dim=1)  # c,d
    s_features = s_features / s_features.norm(dim=-1, keepdim=True)
    s_features = s_features.to(t_features.dtype)

    s_features_unsqueezed = s_features.unsqueeze(1)

    raw_weights = torch.bmm(s_features_unsqueezed, t_features.transpose(1, 2))
    raw_weights = raw_weights.squeeze(1)

    matching_score = raw_weights.clone()

    normed_weights = F.softmax(raw_weights * gamma, dim=-1)
    normed_clip_weights = torch.einsum("cp, cpd -> cd", normed_weights, t_features)
    normed_clip_weights = normed_clip_weights / normed_clip_weights.norm(dim=-1, keepdim=True)

    if return_matching:
        return normed_clip_weights, matching_score
    elif return_weights:
        return normed_clip_weights, normed_weights
    else:
        return normed_clip_weights


def vec_sort(vecs_t, matching_score):
    cate_num, prompt_num, dim = vecs_t.shape  # N,P,D

    weights, sorted_idx = torch.topk(matching_score, k=prompt_num, dim=-1)
    sort_vecs_t = []
    for c in range(cate_num):
        sort_vecs_t.append(vecs_t[c][sorted_idx[c]].clone())
    sort_vecs_t = torch.stack(sort_vecs_t, dim=0)

    if len(sort_vecs_t.shape) == 2:
        sort_vecs_t = sort_vecs_t.unsqueeze(1)

    return sort_vecs_t, weights


def image_guide_text_search(cfg, clip_weights_cupl_all, val_features, val_labels, image_weights):
    best_acc = 0
    best_gamma = 0
    for gamma in range(5, 101, 5):
        clip_weights_cupl_IGT, matching_score = image_guide_text(
            cfg, clip_weights_cupl_all, image_weights, return_matching=True, gamma=gamma
        )
        clip_weights_cupl_IGT = clip_weights_cupl_IGT.t()  # D, C

        val_logits = val_features @ clip_weights_cupl_IGT  # N, C
        acc = (val_logits.argmax(-1) == val_labels).sum() / len(val_labels)

        if acc > best_acc:
            best_acc = acc
            best_gamma = gamma
    print("best_gamma:", best_gamma)
    clip_weights_cupl_IGT, matching_score = image_guide_text(
        cfg, clip_weights_cupl_all, image_weights, return_matching=True, gamma=best_gamma
    )
    clip_weights_cupl_IGT = clip_weights_cupl_IGT.t()
    return clip_weights_cupl_IGT, matching_score


class AttentionCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 128),  # 新增最大值和最小值统计量
            nn.GELU(),  # 使用GELU激活函数
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )
        nn.init.kaiming_normal_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)

    def forward(self, clip_weights, cache_keys):
        print(clip_weights.shape, cache_keys.shape)
        # 新增最大值和最小值统计量
        clip_max = clip_weights.max(dim=1)[0]
        clip_min = clip_weights.min(dim=1)[0]
        cache_max = cache_keys.max(dim=0)[0]
        cache_min = cache_keys.min(dim=0)[0]
        # 批量计算统计量
        clip_means = clip_weights.mean(dim=1)
        clip_stds = clip_weights.std(dim=1)
        cache_means = cache_keys.mean(dim=0)
        cache_stds = cache_keys.std(dim=0)

        print(
            "clip_means, clip_stds, cache_means, cache_stds:",
            clip_means.shape,
            clip_stds.shape,
            cache_means.shape,
            cache_stds.shape,
        )

        # 拼接并计算得分
        stats = torch.stack(
            [clip_means, clip_stds, clip_max, clip_min, cache_means, cache_stds, cache_max, cache_min], dim=1
        )
        scores = self.mlp(stats).squeeze(-1)

        # 增强数值稳定性
        scores = scores / torch.sqrt(torch.tensor(64.0))
        return F.softmax(scores, dim=0)


class GDA_Training(nn.Module):
    def __init__(self, cfg, clip_weights, model, cache_keys):
        super().__init__()
        self.shots = cfg["shots"] * cfg["augment_epoch"]
        self.feat_dim, self.cate_num = clip_weights.shape

        self.res = nn.Parameter(torch.zeros([self.cate_num, self.feat_dim]))
        print("res shape:", self.res.shape)
        self.value_weights = nn.Parameter(torch.ones([self.cate_num * self.shots, 1]))

        nn.init.xavier_normal_(self.res)
        nn.init.constant_(self.value_weights, 1.0)

        self.attention_criterion = AttentionCriterion()

    def forward(self, cache_keys, clip_weights, cache_values):
        attention_weights = self.attention_criterion(clip_weights, cache_keys)

        # 优化张量扩展
        res_keys = self.res[:, None, :].expand(-1, self.shots, -1).reshape(-1, self.feat_dim)
        res_keys = res_keys * attention_weights.unsqueeze(0)
        new_cache_keys = cache_keys + res_keys  # 假设允许直接修改

        # 调整clip_weights
        res_text = self.res.t() * attention_weights.unsqueeze(1)
        new_clip_weights = clip_weights + res_text

        # 调整values
        new_cache_values = cache_values * self.value_weights
        print(
            "new_cache_keys shape, new_clip_weights shape, new_cache_values shape:",
            new_cache_keys.shape,
            new_clip_weights.shape,
            new_cache_values.shape,
        )

        return new_cache_keys, new_clip_weights, new_cache_values


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * (1.0 - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()
