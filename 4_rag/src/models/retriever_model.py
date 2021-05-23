from typing import Any, List

from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy
from src.models.modules.encoder import ElectraSmallEncoder, ElectraBaseEncoder
# from src.models.modules.projection import Projection
# from src.models.pretrain_encoder import Retriever
from transformers import AdamW, get_cosine_schedule_with_warmup


class DualEncoder(LightningModule):
    def __init__(self, lr: float = 5e-05, weight_decay: float = 0.0005, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        SMALL_NAME = "monologg/koelectra-small-v3-discriminator"
        BASE_NAME = "monologg/koelectra-base-v3-discriminator"
        # self.c_encoder = BertEncoder.from_pretrained(MODEL_NAME)
        # self.q_encoder = BertEncoder.from_pretrained(MODEL_NAME)
        # self.c_encoder = ElectraBaseEncoder.from_pretrained(BASE_NAME)
        self.q_encoder = ElectraSmallEncoder.from_pretrained(SMALL_NAME)
        self.c_encoder = ElectraSmallEncoder.from_pretrained(SMALL_NAME)
        
        # self.c_encoder.resize_token_embeddings(35000 + 2)

        self.criterion = nn.NLLLoss()

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def forward(self, inputs):
        q_inputs = {
            "input_ids": inputs[0],
            "attention_mask": inputs[1],
            # "token_type_ids": inputs[2],
        }
        c_inputs = {
            "input_ids": inputs[3],
            "attention_mask": inputs[4],
            # "token_type_ids": inputs[5],
        }

        c_outputs = self.c_encoder(**c_inputs)
        q_outputs = self.q_encoder(**q_inputs)
        return q_outputs, c_outputs

    def training_step(self, batch, batch_idx):
        q_outputs, c_outputs = self(batch)

        sim_scores = torch.matmul(q_outputs, torch.transpose(c_outputs, 0, 1))

        targets = torch.arange(0, c_outputs.size()[0]).long().to(self.device)

        sim_scores = F.log_softmax(sim_scores/self.hparams.softmax_temperature, dim=-1)

        loss = self.criterion(sim_scores, targets)

        acc = self.train_accuracy(torch.argmax(sim_scores, dim=1), targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"]
        )
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch, batch_idx):
        q_outputs, c_outputs = self(batch)

        sim_scores = torch.matmul(q_outputs, torch.transpose(c_outputs, 0, 1))

        targets = torch.arange(0, c_outputs.size()[0]).long().to(self.device)

        sim_scores = F.log_softmax(sim_scores, dim=-1)

        loss = self.criterion(sim_scores, targets)

        acc = self.val_accuracy(torch.argmax(sim_scores, dim=1), targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.c_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.c_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
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
