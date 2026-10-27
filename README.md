# DTSL: Decoupled Two-Stage Learning for Exemplar-Free Class-Incremental Cattle Face Recognition

This repository provides the official core reference implementation of DTSL for exemplar-free class-incremental cattle face recognition. It accompanies the corresponding manuscript and contains the main method-defining modules and experimental settings required to reproduce the DTSL framework.

The repository includes Stage-1 representation learning, Global Directional Knowledge Distillation (GDKD), Contrastive Knowledge Distillation (CKD), logit distillation, prototype restoration and adaptive virtual replay, Stage-2 task-wise bias calibration, four-view inference, preprocessing, evaluation metrics, random-seed control, and the hyperparameter configuration reported in the manuscript.

The private Small-CowDataset and Large-CowDataset are not distributed because the associated cattle identity data were collected through an industry collaboration and are subject to confidentiality and data-use restrictions. Public-data evaluation is performed on Holstein2025 using the official training/validation split described in the manuscript.

## Experimental configuration

The released default configuration matches the manuscript:

| Item | Value |
|---|---|
| Backbone | DINOv2 ViT-S/14 |
| Input size | 224 × 224 |
| Batch size | 16 |
| Stage-1 epochs | 50 |
| Initial learning rate | 6.5e-5 |
| Weight decay | 5e-4 |
| StepLR step size | 30 |
| StepLR decay factor | 0.1 |
| Backbone LR scale at base stage | 0.05 |
| lambda_pr | 1.0 |
| lambda_proto | 14.0 |
| lambda_gdkd | 30.0 |
| lambda_ckd | 1.0 |
| tau | 0.07 |
| lambda_kd | 1.0 |
| T_kd | 10.0 |
| Stage-2 calibration ratio | 0.30 |
| Stage-2 epochs | 15 |
| Stage-2 learning rate | 5e-4 |
| Stage-2 weight decay | 0.01 |
| old/new balancing coefficient | 5.0 |
| sigma clamp | [0.05, 2.0] |
| class-arrival seeds | 42, 3407, 129503 |

`T` in the paper denotes the **number of incremental tasks after the initial base task**.
Therefore, `T = 5, 10, 20` correspond to 6, 11, and 21 total learning stages.

## Preprocessing

Training:
1. resize to 224 × 224;
2. random horizontal flip;
3. brightness jitter with magnitude 63/255;
4. convert to tensor;
5. normalize with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
   `[0.229, 0.224, 0.225]`.

Evaluation:
resize, tensor conversion, and normalization only.

For the private CowDatasets, samples of each identity are sorted, shuffled using seed 42,
and split 80/20 into train/test partitions. For Holstein2025, the official training/validation
partition is retained.

## Code-to-paper mapping

| Manuscript component | Reference implementation |
|---|---|
| Rotation-aware labels | `dtsl/core.py::rotation_expand` |
| `L_cls` | `dtsl/losses.py::classification_loss` |
| `L_gdkd` | `dtsl/losses.py::gdkd_loss` |
| `L_ckd` | `dtsl/losses.py::ckd_loss` |
| `L_kd` | `dtsl/losses.py::logit_kd_loss` |
| `L_pr` | `dtsl/losses.py::prototype_restoration_loss` |
| `L_adapt` | `dtsl/losses.py::adaptive_replay_loss` |
| Stage-2 affine calibration | `dtsl/calibration.py` |
| Stage-2 30% subset + Adam setup | `select_calibration_indices`, `build_stage2_optimizer` |
| four-view calibrated inference | `dtsl/inference.py` |

## Minimal check

```bash
pip install -r requirements.txt
python examples/smoke_test.py
```

The smoke test validates the tensor interfaces of the released method modules. It does not
download any private data.

## Data interface

The reference dataset loader accepts a CSV manifest with:

```text
path,label,task,split
relative/path/to/image.jpg,0,0,train
relative/path/to/image2.jpg,0,0,val
```

For Holstein2025, construct the manifest from the official 82-identity training/validation
partition. The open-set gallery/probe identities are not used in the class-incremental
experiments reported in the manuscript.

## Reproducibility note

The released modules correspond to the main DTSL components and experimental settings described in the manuscript.
