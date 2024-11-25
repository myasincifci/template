from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import transforms as T
import pytorch_lightning as pl
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from wilds import get_dataset
from wilds.common.grouper import CombinatorialGrouper
from utils import DomainMapper

class CamelyonDM(pl.LightningDataModule):
    def __init__(self, cfg, unlabeled=False) -> None:
        super().__init__()
        self.data_dir = cfg.data.path
        self.unlabeled = cfg.unlabeled
        self.batch_size = cfg.param.batch_size

        if cfg.data.color_aug:
            self.train_transform = BYOLView1Transform(input_size=96, gaussian_blur=0.0)
        else:
            self.train_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"],
                    std=IMAGENET_NORMALIZE["std"],
                ),
            ])

        self.val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ])

        self.labeled_dataset = get_dataset(
            dataset='camelyon17',
            download=True, 
            root_dir=self.data_dir, 
            unlabeled=False
        )

        if self.unlabeled:
            self.unlabeled_dataset = get_dataset(
                dataset='camelyon17',
                download=True, 
                root_dir=self.data_dir, 
                unlabeled=True
            )

        self.grouper = CombinatorialGrouper(self.labeled_dataset, ['hospital'])

        self.cfg = cfg
        self.domain_mapper = DomainMapper().setup(
            self.labeled_dataset.get_subset("train").metadata_array[:, 0]
        )

        self.num_classes = self.labeled_dataset.n_classes

        self.train_set = self.labeled_dataset.get_subset(
                "train", 
                transform=self.train_transform
        )

        self.val_set_id = self.labeled_dataset.get_subset(
            "id_val", 
            transform=self.val_transform
        )

        self.val_set = self.labeled_dataset.get_subset(
            "val", 
            transform=self.val_transform
        )

        self.test_set = self.labeled_dataset.get_subset(
            "test",
            transform=self.val_transform
        )

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
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:    
        val_loader_id = DataLoader(
            self.val_set_id,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        return [
            val_loader_id,
            val_loader,
            test_loader
        ]