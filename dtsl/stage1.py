import torch

from .core import rotation_expand
from .losses import (
    adaptive_replay_loss,
    ckd_loss,
    classification_loss,
    gdkd_loss,
    logit_kd_loss,
    prototype_restoration_loss,
    transfer_regularization,
)


def stage1_objective(model, old_model, images, identity_labels,
                     prototypes, known_identities, new_identities, cfg,
                     current_task_index: int, incremental=True):
    '''
    Compact reference implementation of the manuscript Stage-1 objective.
    '''
    x_rot, y_rot = rotation_expand(images, identity_labels)
    out = model(x_rot, incremental=incremental)

    loss = classification_loss(out["logits"], y_rot)
    parts = {"cls": loss.detach()}

    if known_identities > 0 and old_model is not None:
        with torch.no_grad():
            old_out = old_model(x_rot, incremental=(current_task_index > 1))

        l_gdkd = gdkd_loss(out["features"], old_out["features"], cfg.lambda_gdkd)
        l_ckd = ckd_loss(out["features"], old_out["features"], cfg.lambda_ckd, cfg.tau)
        l_kd = logit_kd_loss(
            out["logits"], old_out["logits"], known_identities,
            cfg.lambda_kd, cfg.kd_temperature,
        )

        # Raw identity labels must be expanded to match the rotated feature batch.
        raw_rot = identity_labels.repeat_interleave(4)
        l_pr = prototype_restoration_loss(
            out["features"], old_out["features"], raw_rot, cfg.lambda_pr
        )
        l_adapt = adaptive_replay_loss(
            model, out["features"], raw_rot, prototypes,
            known_identities, new_identities, cfg.lambda_proto,
        )
        l_tr = transfer_regularization(model.transfer, cfg.lambda_transfer)

        loss = loss + l_gdkd + l_ckd + l_kd + l_pr + l_adapt + l_tr
        parts.update({
            "gdkd": l_gdkd.detach(),
            "ckd": l_ckd.detach(),
            "kd": l_kd.detach(),
            "pr": l_pr.detach(),
            "adapt": l_adapt.detach(),
            "transfer": l_tr.detach(),
        })

    parts["total"] = loss.detach()
    return loss, parts
