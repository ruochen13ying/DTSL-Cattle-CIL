import torch

from .calibration import apply_bias_layers


@torch.no_grad()
def calibrated_four_view_logits(model, images, bias_layers, task_sizes, incremental=True):
    '''
    Paper-aligned inference path:
    four rotated views -> task-wise affine calibration -> identity-level averaging.
    '''
    view_scores = []
    for k in range(4):
        rotated = torch.rot90(images, k, (2, 3))
        logits = model(rotated, incremental=incremental)["logits"]
        logits = apply_bias_layers(logits, bias_layers, task_sizes)

        # Average the four rotation-aware outputs belonging to each identity.
        identity_logits = logits.view(logits.size(0), -1, 4).mean(dim=-1)
        view_scores.append(identity_logits)

    return torch.stack(view_scores, dim=0).mean(dim=0)
