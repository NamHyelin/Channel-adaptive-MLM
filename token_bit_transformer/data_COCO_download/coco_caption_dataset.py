import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset


class CocoCaptionPairs(Dataset):
    """
    Returns (image_path_or_None, caption) pairs.

    - If image_dir is provided, image_path is a Path to the image file.
    - Otherwise, image_path is None (caption-only use).
    - One image has multiple captions; you can:
        * return a random caption per image (default), or
        * return all captions by "flattening" (set flatten=True).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",  # "train" or "val"
        image_dir: Optional[str] = None,  # e.g. "train2017" or "val2017"
        flatten: bool = False,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.split = split
        self.flatten = flatten
        random.seed(seed)

        ann_path = self.root / "annotations" / f"captions_{split}2017.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Missing {ann_path}. Run coco_caption_setup.py first."
            )

        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # image_id -> file_name
        self.id2file: Dict[int, str] = {}
        for img in data["images"]:
            self.id2file[img["id"]] = img["file_name"]

        # image_id -> captions list
        self.id2caps: Dict[int, List[str]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            self.id2caps.setdefault(img_id, []).append(ann["caption"])

        self.image_ids = list(self.id2caps.keys())

        self.image_root: Optional[Path] = None
        if image_dir is not None:
            self.image_root = self.root / image_dir
            if not self.image_root.exists():
                raise FileNotFoundError(f"Missing image dir: {self.image_root}")

        if self.flatten:
            # Build flat list of (image_id, caption_index)
            self.flat_index: List[Tuple[int, int]] = []
            for img_id in self.image_ids:
                caps = self.id2caps[img_id]
                for j in range(len(caps)):
                    self.flat_index.append((img_id, j))
        else:
            self.flat_index = []

    def __len__(self) -> int:
        return len(self.flat_index) if self.flatten else len(self.image_ids)

    def __getitem__(self, idx: int):
        if self.flatten:
            img_id, j = self.flat_index[idx]
            caption = self.id2caps[img_id][j]
        else:
            img_id = self.image_ids[idx]
            caption = random.choice(self.id2caps[img_id])

        img_path = None
        if self.image_root is not None:
            img_path = self.image_root / self.id2file[img_id]

        return img_path, caption
