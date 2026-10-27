from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(train: bool, image_size: int = 224):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def split_identity_paths(paths, seed: int = 42, train_ratio: float = 0.8):
    """
    Private CowDataset protocol used in the manuscript:
    sort -> fixed-seed shuffle -> first 80% train / remaining 20% test.
    """
    import random
    paths = sorted(list(paths))
    rng = random.Random(seed)
    rng.shuffle(paths)
    split = int(len(paths) * train_ratio)
    return paths[:split], paths[split:]


class ManifestDataset(Dataset):
    '''
    CSV columns:
        path,label,task,split

    `split` should be "train", "val", or "test".
    Holstein2025 should retain the official train/validation partition.
    '''
    def __init__(self, manifest, root=".", split="train", task_ids=None, image_size=224):
        self.root = Path(root)
        df = pd.read_csv(manifest)
        df = df[df["split"] == split].copy()
        if task_ids is not None:
            df = df[df["task"].isin(task_ids)].copy()

        self.paths = df["path"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = build_transform(train=(split == "train"), image_size=image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.root / self.paths[index]).convert("RGB")
        image = self.transform(image)
        label = self.labels[index]
        return index, image, label
