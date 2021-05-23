from typing import Any, List

from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy
from transformers.utils.dummy_pt_objects import ElectraModel
from src.models.retriever_model import DualEncoder
from transformers import AdamW, get_cosine_schedule_with_warmup
from src.metrics.aggregation import mean, precision_at_k

class Retriever(LightningModule):
    def __init__(
        self, 
        lr: float = 5e-05, 
        weight_decay: float = 0.0005, 
        num_negatives: int = 111095,
        encoder_momentum: float = 0.999,
        softmax_temperature: float = 0.25,
        num_workers: int = 4,
        batch_size: int = 16,
        max_epochs: int = 2,
        warmup_ratio: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder_q, self.encoder_c = self._load_pretrained_model()

        # for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        #     param_k.data.copy_(param_q.data)
        #     param_k.requires_grad = False

        for name, param in self.encoder_c.named_parameters():
            param.requires_grad = False

        self.register_buffer('queue', torch.zeros(256, 112620))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }
    
    def _load_pretrained_model(self):
        encoder = DualEncoder.load_from_checkpoint('/opt/ml/code/pytorch_lightning_examples/3_RETRIEVER/logs/runs/2021-05-18/09-21-14/checkpoints/epoch=18.ckpt')
        encoder_q = encoder.q_encoder
        encoder_c = encoder.c_encoder
        
        return encoder_q, encoder_c

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        print('queue change')
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, inputs):
        q_inputs = {
            "input_ids": inputs[0],
            "attention_mask": inputs[1],
            "token_type_ids": inputs[2],
        }
        c_inputs = {
            "input_ids": inputs[3],
            "attention_mask": inputs[4],
            "token_type_ids": inputs[5],
        }
        q = self.encoder_q(**q_inputs)

        with torch.no_grad():
            k = self.encoder_c(**c_inputs)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.hparams.softmax_temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # self._dequeue_and_enqueue(k)
        return logits, labels

    def training_step(self, batch, batch_idx):
        # self._momentum_update_key_encoder()
        logits, labels = self(batch)

        loss = self.criterion(logits.float(), labels.long())

        acc1, acc5 = precision_at_k(logits, labels, top_k=(1, 5))

        log = {'train_loss': loss, 'train_acc1': acc1, 'train_acc5': acc5}
        self.log_dict(log)
        return loss


    def validation_step(self, batch, batch_idx):
        output, target = self(batch)
        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {'val_loss': val_loss, 'val_acc1': val_acc1, 'val_acc5': val_acc5}
        self.log_dict(log)

    def configure_optimizers(self):
        param_optimzier = list(self.encoder_q.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimzier if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimzier if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False
        )

        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        num_training_steps = int(
            data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs
        )
        num_warmup_steps = int(num_training_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "loss",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler]
    
        
