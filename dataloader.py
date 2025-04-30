import glob
import os
import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
import torch
from torch.utils.data import random_split
import numpy as np

# Sample usage:
# get_data_loader(data_path, opts)

STYLE_CLASSES = sorted(set( [
    "가르마",
    "기타남자스타일",
    "기타레이어드",
    "기타여자스타일",
    "남자일반숏",
    "댄디",
    "루프",
    "리젠트",
    "리프",
    "미스티",
    "바디",
    "베이비",
    "보니",
    "보브",
    "빌드",
    "소프트투블럭댄디",
    "숏단발",
    "쉐도우",
    "쉼표",
    "스핀스왈로",
    "시스루댄디",
    "애즈",
    "에어",
    "여자일반숏",
    "원랭스",
    "원블럭댄디",
    "테슬",
    "포마드",
    "플리츠",
    "허쉬",
    "히피",
]))
STYLE_TO_IDX = {style: i for i, style in enumerate(STYLE_CLASSES)}


class StyleImageDataset(Dataset):

    def __init__(self, root_dir, ext='*.jpg', transform=None):
        self.label_dir = os.path.join(root_dir, "labels")
        self.image_dir = os.path.join(root_dir, "hair_images")
        self.transform = transform
        self.samples = []

        self._load_dataset(ext)
        self.targets = [label for _, label in self.samples] 
        print(f"Loaded {len(self.samples)} samples from {self.label_dir}")

    def _load_dataset(self, ext):
        # i = 0 
        json_files = glob.glob(os.path.join(self.label_dir, "*", "*", "*.json")) 
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue

            style = data.get("basestyle")
            if style not in STYLE_TO_IDX:
                continue

            json_rel_path = os.path.relpath(json_path, self.label_dir)
            folder1, folder2, filename = json_rel_path.split(os.sep)
            image_filename = os.path.splitext(filename)[0].replace("_", "-") + ".jpg"
            image_path = os.path.join(self.image_dir, folder1, folder2, image_filename)

            if not os.path.exists(image_path):
                continue

            self.samples.append((image_path, STYLE_TO_IDX[style]))
            # i+=1
            # if i>15: break 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, style_idx = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.zeros(len(STYLE_CLASSES))
        label[style_idx] = 1.0

        return image, label

# from stargan
def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def get_data_loader(data_path, opts):
    if opts.data_preprocess == 'resize_only':
        transform = transforms.Compose([
            transforms.Resize((opts.image_size, opts.image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
    elif opts.data_preprocess == 'vanilla':
        load_size = int(1.1 * opts.image_size)
        transform = transforms.Compose([
            transforms.Resize((load_size, load_size), Image.BICUBIC),
            transforms.RandomCrop(opts.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
    elif opts.data_preprocess == 'clip':
        # based on clip github's preprocessing
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ]) 
    else:
        raise ValueError(f"Unknown data_preprocess type: {opts.data_preprocess}")

    full_dataset = StyleImageDataset(data_path, opts.ext, transform)

    val_split = int(len(full_dataset) * 0.002)
    train_dataset, val_dataset = random_split(full_dataset, [len(full_dataset) - val_split, val_split])
    
    train_targets = [full_dataset.targets[i] for i in train_dataset.indices]
    train_sampler = _make_balanced_sampler(train_targets) 
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, sampler = train_sampler,
        shuffle=False, num_workers=opts.num_workers
    ) 
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, 
        shuffle=False, num_workers=opts.num_workers
    )

    return train_loader, val_loader

def get_all_data_loader(data_path, opts):
    if opts.data_preprocess == 'resize_only':
        transform = transforms.Compose([
            transforms.Resize((opts.image_size, opts.image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
    elif opts.data_preprocess == 'vanilla':
        load_size = int(1.1 * opts.image_size)
        transform = transforms.Compose([
            transforms.Resize((load_size, load_size), Image.BICUBIC),
            transforms.RandomCrop(opts.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
    elif opts.data_preprocess == 'clip':
        # based on clip github's preprocessing
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            ),
        ]) 
    else:
        raise ValueError(f"Unknown data_preprocess type: {opts.data_preprocess}")

    full_dataset = StyleImageDataset(data_path, opts.ext, transform)
    sampler = _make_balanced_sampler(full_dataset.targets)
    
    full_loader = DataLoader(
        dataset=full_dataset, batch_size=opts.batch_size, sampler = sampler,
        shuffle=False, num_workers=opts.num_workers
    )

    return full_loader
