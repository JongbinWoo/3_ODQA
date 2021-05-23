from pytorch_lightning.callbacks import Callback

class UnFreezeCallback(Callback):
    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            for param in pl_module.c_encoder.parameters():
                param.requires_grad = False

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            for name, param in pl_module.c_encoder.named_parameters():
                if not (('embeddings' in name) or ('encoder.layer.0' in name) or ('encoder.layer.1.' in name) or ('encoder.layer.2' in name) or ('encoder.layer.3' in name) or ('encoder.layer.4' in name)):
                    param.requires_grad = True