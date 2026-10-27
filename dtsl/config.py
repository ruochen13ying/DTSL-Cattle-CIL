from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DTSLConfig:
    # backbone / optimization
    backbone: str = "dinov2_vits14"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 50
    lr: float = 6.5e-5
    weight_decay: float = 5e-4
    step_size: int = 30
    lr_decay: float = 0.1
    backbone_lr_scale: float = 0.05

    # Stage-1 losses
    lambda_pr: float = 1.0
    lambda_proto: float = 14.0
    lambda_gdkd: float = 30.0
    lambda_ckd: float = 1.0
    tau: float = 0.07
    lambda_kd: float = 1.0
    kd_temperature: float = 10.0
    lambda_transfer: float = 0.5

    # Stage-2 calibration
    calibration_ratio: float = 0.30
    stage2_epochs: int = 15
    stage2_lr: float = 5e-4
    stage2_weight_decay: float = 0.01
    old_new_balance: float = 5.0
    sigma_clip: Tuple[float, float] = (0.05, 2.0)

    # reproducibility
    class_arrival_seeds: Tuple[int, ...] = (42, 3407, 129503)
