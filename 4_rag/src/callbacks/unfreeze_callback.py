from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
import time 
class SleepValidation(Callback):
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print('SLEEP VALIDATION FOR A MINUATE')
        time.sleep(10)
        return super().on_validation_start(trainer, pl_module)
    # def on_epoch_start(self, trainer, pl_module):
    #     if trainer.current_epoch == 0:
    #         for param in pl_module.c_encoder.parameters():
    #             param.requires_grad = False

    # def on_epoch_end(self, trainer, pl_module):
    #     if trainer.current_epoch == 0:
    #         for name, param in pl_module.c_encoder.named_parameters():
    #             if not (('embeddings' in name) or ('encoder.layer.0' in name) or ('encoder.layer.1.' in name) or ('encoder.layer.2' in name) or ('encoder.layer.3' in name) or ('encoder.layer.4' in name)):
    #                 param.requires_grad = True