import os
import random
import hydra
import pytorch_lightning as L
import torch
import torch.nn as nn
from data_modules.pacs_dm import PacsDM
from data_modules.camelyon17_dm import CamelyonDM
from model import ResnetClf
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torchvision import transforms as T
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb

@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig) -> None:
    print(os.getcwd())
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("medium")

    logger = True
    if cfg.logging:
        wandb.init(
            project=cfg.logger.project,
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
        logger = WandbLogger()

    seed = random.randint(0,9999999)
    L.seed_everything(seed, workers=True)

    # Data
    match cfg.data.name:
        case 'pacs':
            data_module = PacsDM(cfg, leave_out=['sketch'])
        case 'domainnet':
            raise NotImplementedError
        case 'camelyon':
            data_module = CamelyonDM(cfg, unlabeled=False)

    # Model
    model = ResnetClf(cfg=cfg, dm=data_module)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        logger=logger,
        log_every_n_steps=cfg.logger.log_every_n_steps,
        callbacks=[LearningRateMonitor(logging_interval='step')]
    )

    trainer.fit(
        model=model,
        datamodule=data_module
    )


if __name__ == "__main__":
    main()
