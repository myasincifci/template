from typing import List

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import pytorch_lightning as pl
from lightly.transforms.byol_transform import BYOLView1Transform
from data_modules.dr_dataset import get_loo_dr

class DomainMapper():
    def __init__(self):
        self.unique_domains = [0,1,2,3]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

class DRDM(pl.LightningDataModule):
    def __init__(self, cfg, leave_out: str) -> None:
        super().__init__()
        self.data_dir = cfg.data.path + '/DR'
        self.batch_size = cfg.param.batch_size

        if cfg.data.color_aug:
            self.train_transform = T.Compose([
                T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.3, 0.3),
                T.RandomGrayscale(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.train_transform = T.Compose([
                T.RandomResizedCrop(224, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.val_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
        ])

        self.train_set, self.test_set = get_loo_dr(
            root=self.data_dir,
            leave_out=leave_out,
            train_tf=self.train_transform,
            test_tf=self.val_transform
        )

        self.domain_mapper = DomainMapper()

        self.cfg = cfg
        self.num_classes = self.train_set.n_classes

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            pass
            
        elif stage == 'test':
            pass
        
        elif stage == 'predict':
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
    
def main():
    pass

if __name__ == '__main__':
    main()