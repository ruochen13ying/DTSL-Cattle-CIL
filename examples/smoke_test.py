import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dtsl.calibration import BiasLayer, apply_bias_layers
from dtsl.config import DTSLConfig
from dtsl.losses import ckd_loss, gdkd_loss, logit_kd_loss, prototype_restoration_loss
from dtsl.metrics import average_incremental_accuracy, backward_transfer, last_accuracy


cfg = DTSLConfig()
batch = 8
dim = 384
known = 10

current = torch.randn(batch, dim)
old = torch.randn(batch, dim)
labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
current_logits = torch.randn(batch, (known + 2) * 4)
old_logits = torch.randn(batch, known * 4)

print("GDKD:", float(gdkd_loss(current, old, cfg.lambda_gdkd)))
print("CKD:", float(ckd_loss(current, old, cfg.lambda_ckd, cfg.tau)))
print("KD:", float(logit_kd_loss(current_logits, old_logits, known,
                                  cfg.lambda_kd, cfg.kd_temperature)))
print("PR:", float(prototype_restoration_loss(current, old, labels, cfg.lambda_pr)))

bias_layers = torch.nn.ModuleList([BiasLayer(), BiasLayer()])
logits = torch.randn(batch, 12 * 4)
print("Calibrated logits:", tuple(apply_bias_layers(logits, bias_layers, [10, 2]).shape))

history = [88.0, 86.0, 84.0]
matrix = [[90.0], [85.0, 88.0], [82.0, 84.0, 87.0]]
print("Last:", last_accuracy(history))
print("A_avg:", average_incremental_accuracy(history))
print("BWT:", backward_transfer(matrix))
print("Smoke test passed.")
