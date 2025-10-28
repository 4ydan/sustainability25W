"""Dataset loading utilities for image captioning."""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
import requests
from tqdm import tqdm


class COCOCaptionDataset(Dataset):
    """COCO Caption dataset loader."""
    
    def __init__(self, data_dir: str, split: str = "val2017", max_samples: int = None, transform=None):
        """
        Initialize COCO Caption dataset.
        
        Args:
            data_dir: Root directory for COCO data
            split: Dataset split (val2017, train2017)
            max_samples: Maximum number of samples to load (for subset experiments)
            transform: Optional transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Paths
        self.image_dir = self.data_dir / split
        self.ann_file = self.data_dir / "annotations" / f"captions_{split}.json"
        
        # Download annotations if not present
        self._download_annotations()
        
        # Load COCO annotations
        self.coco = COCO(str(self.ann_file))
        
        # Get image IDs
        self.img_ids = list(self.coco.imgs.keys())
        if max_samples is not None:
            self.img_ids = self.img_ids[:max_samples]
        
        print(f"Loaded {len(self.img_ids)} images from COCO {split}")
    
    def _download_annotations(self):
        """Download COCO annotations if not present."""
        os.makedirs(self.data_dir / "annotations", exist_ok=True)
        
        if not self.ann_file.exists():
            print(f"Downloading COCO annotations for {self.split}...")
            url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            # Note: In practice, user should download this manually
            print(f"Please download annotations from: {url}")
            print(f"Extract to: {self.data_dir / 'annotations'}")
            raise FileNotFoundError(f"Annotations not found at {self.ann_file}")
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, List[str], int]:
        """
        Get image and its captions.
        
        Returns:
            image: PIL Image
            captions: List of reference captions
            image_id: COCO image ID
        """
        img_id = self.img_ids[idx]
        
        # Load image
        img_info = self.coco.imgs[img_id]
        img_path = self.image_dir / img_info['file_name']
        
        if not img_path.exists():
            # Return a dummy image if file not found (for testing without full dataset)
            image = Image.new('RGB', (224, 224), color='gray')
        else:
            image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get all captions for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]
        
        return image, captions, img_id
    
    def get_annotations_dict(self) -> Dict:
        """Get annotations in evaluation format."""
        annotations = {}
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            annotations[img_id] = [ann['caption'] for ann in anns]
        return annotations


def create_coco_dataloader(
    data_dir: str,
    split: str = "val2017",
    batch_size: int = 1,
    num_workers: int = 4,
    max_samples: int = None
) -> DataLoader:
    """
    Create COCO dataloader.
    
    Args:
        data_dir: Root directory for COCO data
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of worker processes
        max_samples: Maximum number of samples
        
    Returns:
        DataLoader instance
    """
    dataset = COCOCaptionDataset(
        data_dir=data_dir,
        split=split,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x  # Return list of tuples
    )
    
    return dataloader


def create_dummy_dataset(num_samples: int = 100) -> List[Tuple[Image.Image, List[str], int]]:
    """
    Create a dummy dataset for testing without downloading full COCO.
    
    Args:
        num_samples: Number of dummy samples to create
        
    Returns:
        List of (image, captions, image_id) tuples
    """
    dummy_data = []
    for i in range(num_samples):
        # Create a simple colored image
        image = Image.new('RGB', (224, 224), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
        captions = [f"A sample image number {i}", f"This is test image {i}"]
        image_id = i
        dummy_data.append((image, captions, image_id))
    
    return dummy_data
