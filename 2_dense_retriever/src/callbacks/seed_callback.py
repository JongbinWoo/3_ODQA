from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
from random import randint
class SeedCallback(Callback):
    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        trainer.datamodule.current_epochs = pl_module.current_epoch
        
        # trainer.train_dataloader = trainer.datamodule.train_dataloader(type)
        # trainer.val_dataloaders = trainer.datamodule.val_dataloader(type)