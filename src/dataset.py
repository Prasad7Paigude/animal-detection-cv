import json
import logging
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from config.settings import RESIZE_DIM

logger = logging.getLogger(__name__)


def load_coco_annotations(
    folder_path: str,
) -> tuple[dict[int, str], list[dict], dict[int, str]]:
    json_path = os.path.join(folder_path, "_annotations.coco.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    images = {img["id"]: img["file_name"] for img in data["images"]}
    annotations = data["annotations"]
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    return images, annotations, categories


train_transform = T.Compose([
    T.Resize(RESIZE_DIM),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
])

eval_transform = T.Compose([
    T.Resize(RESIZE_DIM),
    T.ToTensor(),
])


class AnimalDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        transforms: T.Compose | None = None,
    ) -> None:
        self.folder_path = folder_path
        self.transforms = transforms
        self.images, self.annotations, self.categories = load_coco_annotations(
            folder_path
        )

        self.image_annotations: dict[int, list[dict]] = {
            img_id: [] for img_id in self.images
        }
        for ann in self.annotations:
            self.image_annotations[ann["image_id"]].append(ann)

        self.image_ids = [
            img_id
            for img_id in self.images
            if len(self.image_annotations[img_id]) > 0
        ]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.folder_path, self.images[img_id])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for ann in self.image_annotations[img_id]:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target: dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
