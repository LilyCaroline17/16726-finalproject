import glob
import os
import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# Sample usage:
# get_data_loader(data_path, opts)

# Define and freeze the style classes
STYLE_CLASSES = sorted(set([ 
    "댄디", "포마드", "리젠트", "가르마", "쉼표", "크롭", "투블럭", "허쉬컷", "삭발", "베이비펌",
    "스핀스왈로", "아이비리그", "쉐도우펌", "내추럴펌", "볼륨펌", "애즈펌", "스왈로펌", "리젠트펌",
    "댄디펌", "투블럭펌", "포마드펌", "리프컷", "댄디컷", "볼륨매직", "다운펌", "아이롱펌", "호일펌",
    "애플컷", "기타남자스타일", "커트", "올백"
]))
STYLE_TO_IDX = {style: i for i, style in enumerate(STYLE_CLASSES)}


class StyleImageDataset(Dataset):
    """Loads images and 1-hot encoded 'basestyle' vectors from JSONs"""

    def __init__(self, main_dir, ext='*.png', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.samples = []

        self._load_dataset(ext) 

        print(f"Loaded {len(self.samples)} samples from {main_dir}")

    def _load_dataset(self, ext):
        json_files = glob.glob(os.path.join(self.main_dir, "*.json"))

        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue

            image_filename = data.get("path")
            if not image_filename:
                continue
            print("HEY",image_filename)

            # Assume image is in the same folder
            image_path = os.path.join(self.main_dir, os.path.basename(image_filename))
            if not os.path.exists(image_path) or not image_path.endswith(ext.split(".")[-1]):
                continue

            style = data.get("basestyle")
            if style not in STYLE_TO_IDX:
                continue

            self.samples.append((image_path, STYLE_TO_IDX[style]))

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


def get_data_loader(data_path, opts):
    """Creates DataLoader with image + style vector"""
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
    else:
        raise ValueError(f"Unknown data_preprocess type: {opts.data_preprocess}")

    dataset = StyleImageDataset(
        data_path, opts.ext, transform
    )

    dloader = DataLoader(
        dataset=dataset, batch_size=opts.batch_size,
        shuffle=True, num_workers=opts.num_workers
    )

    return dloader