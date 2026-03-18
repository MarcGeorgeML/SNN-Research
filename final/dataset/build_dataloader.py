from torch.utils.data import DataLoader

from .multimodal_dataset import MultimodalDataset
from .collate import multimodal_collate


def build_dataloaders(feature_root, batch_size=32, num_workers=4):

    train_dataset = MultimodalDataset(f"{feature_root}/train")
    val_dataset = MultimodalDataset(f"{feature_root}/validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multimodal_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multimodal_collate,
    )

    return train_loader, val_loader
