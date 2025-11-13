"""
Fed-PRISM Client Implementation
Implements Algorithm 3 (ClientUpdate) for YOLO models
"""

from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import copy


class YOLODataset(Dataset):
    """Dataset wrapper for YOLO training data"""
    def __init__(self, images: List[torch.Tensor], targets: List[torch.Tensor]):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


class FedPRISMClient:
    """
    Federated Personalized Relevance-based Intelligent Soft-assignment Models (Fed-PRISM)
    client logic implementing Algorithm 3
    """

    def __init__(
        self,
        client_id: int,
        model,
        dataset: Dict[str, Tuple[List[torch.Tensor], List[torch.Tensor]]],
        local_epochs: int,
        learning_rate: float,
        batch_size: int,
        device: str = "cpu",
    ):
        """Initialize a Fed-PRISM client"""
        self.client_id = client_id
        self.device = torch.device(device)
        self.model = copy.deepcopy(model)
        self.model.to(self.device)
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        train_images, train_targets = dataset.get("train", ([], []))
        val_images, val_targets = dataset.get("val", ([], []))

        self.train_dataset = YOLODataset(train_images, train_targets)
        self.val_dataset = YOLODataset(val_images, val_targets) if val_images else None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        self.val_loader = (
            DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
            )
            if self.val_dataset
            else None
        )

    def _collate_fn(self, batch):
        """Custom collate function for YOLO data"""
        if len(batch) == 0:
            img_size = getattr(self.model, "img_size", 320)
            return (
                torch.empty(0, 3, img_size, img_size, device=self.device),
                torch.empty(0, 6, device=self.device),
            )

        images = []
        targets_list = []

        for img_idx, (img, tgt) in enumerate(batch):
            images.append(img)
            if tgt.numel() > 0:
                tgt_copy = tgt.clone()
                tgt_copy[:, 0] = img_idx
                targets_list.append(tgt_copy)

        images_tensor = torch.stack(images).to(self.device)
        if targets_list:
            targets_tensor = torch.cat(targets_list, dim=0).to(self.device)
        else:
            targets_tensor = torch.empty(0, 6, device=self.device)

        return images_tensor, targets_tensor

    def set_model(self, model):
        """Set the model for this client (receives personalized model from server)"""
        self.model = copy.deepcopy(model)
        self.model.to(self.device)

    def get_model_state(self):
        """Get the current model state dictionary (for clustering)"""
        return self.model.model.state_dict()

    def train(self):
        """Algorithm 3: ClientUpdate"""
        if len(self.train_dataset) == 0:
            zero_delta = {
                key: torch.zeros_like(param) for key, param in self.model.model.state_dict().items()
            }
            return zero_delta

        theta_start = copy.deepcopy(self.model.model.state_dict())
        optimizer = optim.SGD(self.model.model.parameters(), lr=self.learning_rate)

        for _ in range(self.local_epochs):
            for images, targets in self.train_loader:
                if images.numel() == 0:
                    continue
                self.model.training_step(images, targets, optimizer)

        theta_end = self.model.model.state_dict()
        delta = {}
        for key in theta_start.keys():
            delta[key] = theta_end[key] - theta_start[key]

        return delta

    def evaluate(self):
        """Evaluate the model on validation data using mAP@0.5"""
        loader = self.val_loader if self.val_loader is not None else self.train_loader
        if loader is None or len(loader.dataset) == 0:
            return 0.0, 0

        self.model.model.eval()
        total_map = 0.0
        total_images = 0

        with torch.no_grad():
            for images, targets in loader:
                if images.numel() == 0:
                    continue
                map50 = self.model.compute_map50(
                    images, targets, conf_threshold=0.01, iou_threshold=0.5
                )
                total_map += map50 * images.shape[0]
                total_images += images.shape[0]

        avg_map = total_map / total_images if total_images > 0 else 0.0
        return avg_map, total_images

