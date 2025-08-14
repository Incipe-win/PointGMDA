from utils import *
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from sklearn.decomposition import PCA
from scipy.stats import normaltest
from sklearn.cluster import KMeans
from tqdm import tqdm
import MinkowskiEngine as ME


class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar

    def fit(self, X):
        n_samples, n_features = X.shape
        device = X.device

        self.weights_ = torch.ones(self.n_components, device=device) / self.n_components
        X_np = X.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_components, init="k-means++", random_state=42).fit(X_np)
        self.means_ = torch.tensor(kmeans.cluster_centers_, device=device).float()
        self.covariances_ = torch.stack(
            [
                torch.eye(n_features, device=device) + self.reg_covar * torch.randn(n_features, device=device)
                for _ in range(self.n_components)
            ]
        )

        prev_lower_bound = -np.inf

        for _ in range(self.max_iter):
            log_prob = []
            for k in range(self.n_components):
                try:
                    cov = self.covariances_[k] + self.reg_covar * torch.eye(n_features, device=device)
                    mvn = MultivariateNormal(self.means_[k], cov)
                    log_prob_k = mvn.log_prob(X)
                except ValueError:
                    log_prob_k = -0.5 * torch.sum(
                        (X - self.means_[k]) ** 2 / (torch.diag(self.covariances_[k]) + self.reg_covar),
                        dim=1,
                    )
                log_prob.append(log_prob_k + torch.log(self.weights_[k]))

            weighted_logprob = torch.stack(log_prob, dim=1)
            log_resp = weighted_logprob - torch.logsumexp(weighted_logprob, dim=1, keepdim=True)
            resp = torch.exp(log_resp)

            Nk = resp.sum(dim=0) + 1e-10
            self.weights_ = Nk / n_samples
            self.means_ = (resp.T @ X) / Nk[:, None]

            diff = X[:, None, :] - self.means_
            covs = torch.einsum("nk,nki,nkj->kij", resp, diff, diff) / Nk[:, None, None]
            self.covariances_ = covs + self.reg_covar * torch.eye(n_features, device=device)[None, :, :]

            current_lower_bound = (resp * weighted_logprob).sum()
            if current_lower_bound - prev_lower_bound < self.tol:
                break
            prev_lower_bound = current_lower_bound

        return self


def gmda(
    vecs,
    labels,
    clip_weights,
    val_features,
    val_labels,
    alpha_shift=False,
    n_components=2,
    use_learned_features=True,
):
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
    cov_reg = cov + torch.trace(cov) / d * torch.eye(d).cuda()
    cov_inv = d * torch.linalg.pinv(cov_reg)

    ps = torch.ones(clip_weights.shape[1]).cuda() / clip_weights.shape[1]
    W = torch.einsum("nd,dc->cn", mus, cov_inv)
    b = ps.log() - 0.5 * torch.einsum("nd,dc,nc->n", mus, cov_inv, mus)

    best_val_acc, best_alpha = 0, 0.1
    for alpha in [10**i for i in range(-4, 5)]:
        if alpha_shift:
            val_logits = alpha * val_features @ clip_weights + val_features @ W + b
        else:
            val_logits = 100.0 * val_features @ clip_weights + alpha * (val_features @ W + b)
        acc = cls_acc(val_logits, val_labels)
        if acc > best_val_acc:
            best_val_acc, best_alpha = acc, alpha

    print(f"best_val_alpha: {best_alpha}\tbest_val_acc: {best_val_acc}")
    return best_alpha, W, b, best_val_acc


def helper(
    cfg,
    cache_keys,
    cache_values,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
    model,
    train_loader_F,
):
    feat_dim, cate_num = clip_weights.shape
    cache_values = cache_values.reshape(cate_num, -1, cate_num)
    cache_keys = (
        cache_keys.t().reshape(cate_num, cfg["shots"] * cfg["augment_epoch"], feat_dim).reshape(cate_num, -1, feat_dim)
    )

    cfg["w"] = cfg["w_training"]
    cache_keys, cache_values = (
        cache_keys.reshape(-1, feat_dim),
        cache_values.reshape(-1, cate_num),
    )
    adapter = GDA_Training(cfg, clip_weights, model, cache_keys).cuda()

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["lr"], eps=cfg["eps"], weight_decay=1e-1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["train_epoch"] * len(train_loader_F))
    Loss = SmoothCrossEntropy(alpha=0.1)

    beta, alpha = cfg["init_beta"], cfg["init_alpha"]
    best_acc, best_epoch = 0.0, 0
    # feat_num = cfg["feat_num"]

    for train_idx in range(cfg["train_epoch"]):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print("Train Epoch: {:} / {:}".format(train_idx, cfg["train_epoch"]))

        for i, (pc, target, rgb) in enumerate(tqdm(train_loader_F)):
            pc, target, rgb = pc.cuda(), target.cuda(), rgb.cuda()
            feat = torch.cat((pc, rgb), dim=-1)
            with torch.no_grad():
                pc_features = model(pc, feat, device="cuda", quantization_size=0.02)
                pc_features /= pc_features.norm(dim=-1, keepdim=True)

            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)
            R_fF = pc_features @ new_cache_keys.t()
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100.0 * pc_features @ new_clip_weights
            refine_logits = R_fW + cache_logits * alpha

            loss = Loss(refine_logits, target)

            keys_mse = F.mse_loss(new_cache_keys, cache_keys)
            clip_mse = F.mse_loss(new_clip_weights, clip_weights)

            res_l2 = torch.norm(adapter.res)
            value_weights_l2 = torch.norm(adapter.value_weights)

            current_decay = cfg["keys_mse_decay"] ** train_idx
            keys_mse_weight = cfg["keys_mse_weight"] * current_decay
            clip_mse_weight = cfg["clip_mse_weight"] * current_decay
            loss = (
                loss
                + keys_mse_weight * keys_mse
                + clip_mse_weight * clip_mse
                + cfg["res_l2_weight"] * res_l2
                + cfg["value_weights_l2_weight"] * value_weights_l2
            )

            acc = cls_acc(refine_logits, target)
            correct_samples += acc / 100 * len(refine_logits)
            all_samples += len(refine_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(
            "LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}".format(
                current_lr,
                correct_samples / all_samples,
                correct_samples,
                all_samples,
                sum(loss_list) / len(loss_list),
            )
        )

        # Eval
        adapter.eval()
        with torch.no_grad():
            new_cache_keys, new_clip_weights, R_FW = adapter(cache_keys, clip_weights, cache_values)

            R_fF = val_features @ new_cache_keys.t()
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ R_FW
            R_fW = 100.0 * val_features @ new_clip_weights
            refine_logits = R_fW + cache_logits * alpha
        acc = cls_acc(refine_logits, val_labels)

        print("**** PointGMDA-T's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter, cfg["cache_dir"] + "/PointGMDA-T_" + str(cfg["shots"]) + "shots.pt")

    adapter = torch.load(
        cfg["cache_dir"] + "/PointGMDA-T_" + str(cfg["shots"]) + "shots.pt",
        weights_only=False,
    )
    vecs_v, clip_weights, labels_v = adapter(cache_keys, clip_weights, cache_values)

    pca = PCA(n_components=0.95, random_state=42)
    pca.fit(vecs_v.detach().cpu().numpy())

    vecs_v = torch.tensor(pca.transform(vecs_v.detach().cpu().numpy()), device=val_features.device)
    val_features = torch.tensor(pca.transform(val_features.detach().cpu().numpy()), device=val_features.device)
    test_features = torch.tensor(pca.transform(test_features.detach().cpu().numpy()), device=val_features.device)

    clip_weights_np = clip_weights.detach().cpu().numpy().T
    clip_weights_reduced = pca.transform(clip_weights_np)
    clip_weights = torch.tensor(clip_weights_reduced.T, device=val_features.device)
    return vecs_v, clip_weights, labels_v, val_features, test_features


def PointGMDA(
    cfg,
    val_features,
    val_labels,
    test_features,
    test_labels,
    clip_weights,
    model,
    train_loader_F,
    is_print=True,
    pca_dim=256,
):
    device = val_features.device

    vecs_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_vecs_f.pt", weights_only=False).float()
    labels_v = torch.load(cfg["cache_dir"] + "/" + f"{cfg['shots']}_labels_f.pt", weights_only=False).float()

    print("vecs_v shape, labels_v shape, clip_weights shape:", vecs_v.shape, labels_v.shape, clip_weights.shape)

    vecs_v = vecs_v.T
    labels_v = F.one_hot(labels_v.long(), num_classes=clip_weights.shape[1]).float()
    # print(vecs_v.shape, labels_v.shape)
    vecs_v, clip_weights, labels_v, val_features, test_features = helper(
        cfg,
        vecs_v,
        labels_v,
        val_features,
        val_labels,
        test_features,
        test_labels,
        clip_weights,
        model,
        train_loader_F,
    )
    labels_v = labels_v.argmax(dim=1)
    print(
        "vecs_v shape, clip_weights shape, labels_v shape, val_features shape, test_features shape:",
        vecs_v.shape,
        clip_weights.shape,
        labels_v.shape,
        val_features.shape,
        test_features.shape,
    )

    clip_weights = clip_weights / clip_weights.norm(dim=0, keepdim=True)
    vecs_v = vecs_v / vecs_v.norm(dim=-1, keepdim=True)
    val_features = val_features / val_features.norm(dim=-1, keepdim=True)
    test_features = test_features / test_features.norm(dim=-1, keepdim=True)

    ensemble_predictions = []
    ensemble_weights = []

    for n_components_i in [2, 3, 4]:
        alpha, W, b, val_acc = gmda(
            vecs_v,
            labels_v,
            clip_weights,
            val_features,
            val_labels,
            n_components=n_components_i,
        )

        test_logits = alpha * test_features.float() @ clip_weights.float() + (test_features.float() @ W + b)
        ensemble_predictions.append(test_logits)
        ensemble_weights.append(val_acc)

    ensemble_weights = torch.tensor(ensemble_weights) / sum(ensemble_weights)

    final_logits = torch.zeros_like(ensemble_predictions[0])
    for i, pred in enumerate(ensemble_predictions):
        final_logits += pred * ensemble_weights[i]
    acc = cls_acc(final_logits, test_labels)

    if is_print:
        print("best_val_alpha: %s \t best_val_acc: %s" % (alpha, val_acc))
        print("training-free acc:", acc)
        print()

    return acc
