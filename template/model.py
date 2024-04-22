from functools import partial
from typing import Any

import pytorch_lightning as L
import torch
import torch.optim as optim
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
from torch.autograd import Function
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18, ResNet18_Weights
from torch.optim import lr_scheduler

class ResnetClf(L.LightningModule):
    def __init__(self, cfg, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if cfg.model.pretrained:
            self.model = resnet18(ResNet18_Weights.DEFAULT)
        else:
            self.model = resnet18()
        self.model.fc = nn.Linear(in_features=512, out_features=cfg.data.num_classes, 
            bias=True)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=cfg.data.num_classes)
        
        self.cfg = cfg

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        X, t, d = batch

        y = self.model(X)
        loss = self.criterion(y, t)
        self.log("train/loss", loss.item())

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        X, t, d = batch

        y = self.model(X)
        loss = self.criterion(y, t)
        self.accuracy(y, t)

        self.log("val/loss", loss.item())
        self.log("val/acc", self.accuracy, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optimizer = optim.SGD(params=self.parameters(), lr=self.cfg.param.lr, weight_decay=1e-4,momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, )

        return [optimizer], #[scheduler]