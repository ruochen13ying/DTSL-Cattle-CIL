import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiasLayer(nn.Module):
    '''
    Task-specific affine calibration:
        z_hat^(k) = alpha_k * z^(k) + beta_k
    '''
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return self.alpha * logits + self.beta


def apply_bias_layers(logits, bias_layers, task_sizes):
    '''
    `task_sizes` are numbers of identities in Task 0, Task 1, ...
    Each identity occupies four rotation-aware classifier outputs.
    '''
    pieces = []
    start = 0
    for layer, identities in zip(bias_layers, task_sizes):
        end = min(start + identities * 4, logits.size(1))
        if end > start:
            pieces.append(layer(logits[:, start:end]))
        start = end
    if start < logits.size(1):
        pieces.append(logits[:, start:])
    return torch.cat(pieces, dim=1)


def gaussian_old_features(prototypes, radii, count, sigma_clip=(0.05, 2.0),
                          device="cpu"):
    old_ids = np.random.randint(0, len(prototypes), size=count)
    mu = torch.stack([
        torch.as_tensor(prototypes[i], dtype=torch.float32, device=device)
        for i in old_ids
    ])
    sigma_np = np.asarray([radii[i] for i in old_ids], dtype=np.float32)
    sigma_np = np.clip(sigma_np, sigma_clip[0], sigma_clip[1])
    sigma = torch.as_tensor(sigma_np, device=device).unsqueeze(1)
    features = mu + torch.randn_like(mu) * sigma
    return features, old_ids


def calibration_step(model, bias_layers, task_sizes, new_images, new_labels,
                     prototypes, radii, known_identities, current_task_size,
                     old_new_balance=5.0, sigma_clip=(0.05, 2.0)):
    '''
    One Stage-2 optimization step.
    The representation network and base classifier should be frozen externally.
    '''
    device = new_images.device

    with torch.no_grad():
        rotated = torch.stack(
            [torch.rot90(new_images, k, (2, 3)) for k in range(4)], dim=1
        ).view(-1, *new_images.shape[1:])
        new_targets = torch.stack(
            [new_labels * 4 + k for k in range(4)], dim=1
        ).view(-1)
        logits_new = model(rotated, incremental=True)["logits"]

    ratio = known_identities / max(1, current_task_size)
    old_count = max(1, int(min(logits_new.size(0) * old_new_balance * ratio, 4096)))

    old_features, old_ids = gaussian_old_features(
        prototypes[:known_identities],
        radii[:known_identities],
        old_count,
        sigma_clip=sigma_clip,
        device=device,
    )
    rotations = torch.randint(0, 4, size=(old_count,), device=device)
    old_targets = torch.as_tensor(old_ids * 4, device=device, dtype=torch.long) + rotations

    with torch.no_grad():
        logits_old = model.logits_from_features(old_features)

    logits = torch.cat([logits_new, logits_old], dim=0)
    targets = torch.cat([new_targets, old_targets], dim=0)
    logits = apply_bias_layers(logits, bias_layers, task_sizes)
    return F.cross_entropy(logits, targets)


def select_calibration_indices(dataset_size, ratio=0.30, seed=42):
    """Select the current-task calibration subset used by Stage 2."""
    import random
    n = max(1, int(dataset_size * ratio))
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:n]


def build_stage2_optimizer(bias_layers, lr=5e-4, weight_decay=0.01):
    """Adam optimizer used for 15 Stage-2 epochs in the manuscript."""
    return torch.optim.Adam(bias_layers.parameters(), lr=lr, weight_decay=weight_decay)
