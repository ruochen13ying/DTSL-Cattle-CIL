import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, scale: float = 30.0):
        super().__init__()
        self.out_features = out_dim
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        f = F.normalize(features, dim=1)
        w = F.normalize(self.weight, dim=1)
        return torch.mm(f, w.t()) * self.scale


class DTSLNet(nn.Module):
    '''
    Paper-aligned network core:
    - DINOv2 ViT-S/14
    - learnable transfer layer for incremental stages
    - cosine classifier over four rotation-aware labels per identity
    '''
    def __init__(self, backbone_name="dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        self.feature_dim = self.backbone.embed_dim
        self.transfer = nn.Linear(self.feature_dim, self.feature_dim)
        nn.init.eye_(self.transfer.weight)
        nn.init.zeros_(self.transfer.bias)
        self.classifier = None

    def expand_classifier(self, num_identities: int):
        out_dim = num_identities * 4
        new_head = CosineClassifier(self.feature_dim, out_dim)
        if self.classifier is not None:
            old = self.classifier.out_features
            with torch.no_grad():
                new_head.weight[:old].copy_(self.classifier.weight)
        self.classifier = new_head.to(next(self.parameters()).device)

    def set_backbone_trainability(self, base_stage: bool):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if base_stage:
            for p in self.backbone.blocks[-1].parameters():
                p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

    def encode(self, x, incremental: bool):
        f = self.backbone(x)
        if incremental:
            f = self.transfer(f)
        return f

    def logits_from_features(self, features):
        return self.classifier(features)

    def forward(self, x, incremental=False):
        features = self.encode(x, incremental=incremental)
        logits = self.logits_from_features(features)
        return {"features": features, "logits": logits}

    def frozen_copy(self):
        m = copy.deepcopy(self)
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
        return m


def rotation_expand(images, identity_labels):
    '''
    Four rotated views and rotation-aware labels:
        y_tilde = 4*y + r, r in {0,1,2,3}
    '''
    x = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], dim=1)
    x = x.view(-1, images.shape[1], images.shape[2], images.shape[3])
    y = torch.stack([identity_labels * 4 + k for k in range(4)], dim=1).view(-1)
    return x, y
