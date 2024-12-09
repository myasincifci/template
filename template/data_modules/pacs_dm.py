import os
from typing import List, Dict

import torch
from torch import tensor
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from data_modules.pacs_h5_dataset import get_pacs_loo

class DomainMapper():
    def __init__(self):
        self.unique_domains = [0,1,2,3]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

class PacsDM(pl.LightningDataModule):
    def __init__(self, cfg, leave_out: List=None) -> None:
        super().__init__()
        self.data_dir = cfg.data.path
        self.batch_size = cfg.param.batch_size

        if cfg.data.color_aug:
            self.train_transform = BYOLView1Transform(
                input_size=224, 
                gaussian_blur=0.0
            )
        else:
            self.train_transform = BYOLView1Transform(
                input_size=224, 
                cj_prob=0.0,
                random_gray_scale=0.0,
                gaussian_blur=0.0,
                solarization_prob=0.0,
            )

        self.val_transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_set, self.test_set = get_pacs_loo(
            root=cfg.data.path,
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