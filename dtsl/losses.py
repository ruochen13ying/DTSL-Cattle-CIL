import numpy as np
import torch
import torch.nn.functional as F


def classification_loss(logits, rotation_labels):
    return F.cross_entropy(logits, rotation_labels)


def gdkd_loss(current_features, old_features, weight=30.0):
    cur = F.normalize(current_features, p=2, dim=1)
    old = F.normalize(old_features, p=2, dim=1)
    return weight * (1.0 - (cur * old).sum(dim=1)).mean()


def ckd_loss(current_features, old_features, weight=1.0, tau=0.07):
    cur = F.normalize(current_features, p=2, dim=1)
    old = F.normalize(old_features, p=2, dim=1)
    similarity = torch.matmul(cur, old.t()) / tau
    targets = torch.arange(cur.size(0), device=cur.device)
    return weight * F.cross_entropy(similarity, targets)


def logit_kd_loss(current_logits, old_logits, known_identities,
                  weight=1.0, temperature=10.0):
    '''
    Distillation is restricted to the historical rotation-aware label space.
    '''
    old_dim = known_identities * 4
    if old_dim == 0:
        return current_logits.new_zeros(())
    t = temperature
    log_p = F.log_softmax(current_logits[:, :old_dim] / t, dim=1)
    q = F.softmax(old_logits[:, :old_dim] / t, dim=1)
    return weight * (t * t) * F.kl_div(log_p, q, reduction="batchmean")


def _batch_centers(features, raw_labels):
    centers = {}
    for cls in raw_labels.unique():
        mask = raw_labels == cls
        centers[int(cls.item())] = features[mask].mean(dim=0)
    return centers


def prototype_restoration_loss(current_features, old_features, raw_labels, weight=1.0):
    '''
    Paper-aligned residual preservation:
      (f_t - mu_{y,t}) ~= (f_{t-1} - mu_{y,t-1})
    Both centers are evaluated on the same current-task batch.
    '''
    current_centers = _batch_centers(current_features, raw_labels)
    old_centers = _batch_centers(old_features, raw_labels)

    cur_residuals, old_residuals = [], []
    for i, y in enumerate(raw_labels.tolist()):
        cur_residuals.append(current_features[i] - current_centers[y])
        old_residuals.append(old_features[i] - old_centers[y])

    cur_residuals = torch.stack(cur_residuals)
    old_residuals = torch.stack(old_residuals)
    return weight * F.mse_loss(cur_residuals, old_residuals)


def adaptive_replay_loss(model, current_features, raw_labels, prototypes,
                         known_identities, new_identities, weight=14.0):
    '''
    Residual-based prototype virtual replay corresponding to L_adapt.
    Historical raw images are not stored.
    '''
    if known_identities == 0 or len(prototypes) == 0:
        return current_features.new_zeros(())

    current_centers = _batch_centers(current_features, raw_labels)
    residuals = torch.stack([
        current_features[i] - current_centers[int(raw_labels[i].item())]
        for i in range(current_features.size(0))
    ]).detach()

    n_old = residuals.size(0)
    old_ids = np.random.randint(0, known_identities, size=n_old)
    proto = torch.stack([
        torch.as_tensor(prototypes[i], device=current_features.device, dtype=current_features.dtype)
        for i in old_ids
    ])

    restored = proto + residuals
    logits = model.logits_from_features(restored)

    # Rotation-aware classifier: historical virtual samples use the canonical r=0 target.
    targets = torch.as_tensor(old_ids * 4, device=current_features.device, dtype=torch.long)
    class_ratio = known_identities / max(1, new_identities)
    return weight * class_ratio * F.cross_entropy(logits, targets)


def transfer_regularization(transfer_layer, weight=0.5):
    eye = torch.eye(
        transfer_layer.weight.size(0),
        device=transfer_layer.weight.device,
        dtype=transfer_layer.weight.dtype,
    )
    return weight * torch.norm(transfer_layer.weight - eye)
