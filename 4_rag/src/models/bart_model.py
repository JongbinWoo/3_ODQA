from typing import Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from datasets import load_metric
from transformers import BartForConditionalGeneration
from transformers import AdamW, get_cosine_schedule_with_warmup

# from torch.optim.lr_scheduler import LambdaLR
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer


class KoBARTConditionalGeneration(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, lr: float = 0.001, weight_decay: float = 0.0005, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = BartForConditionalGeneration.from_pretrained(
            get_pytorch_kobart_model()
        )
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.tokenizer = get_kobart_tokenizer()
        self.model.resize_token_embeddings(self.tokenizer.vocab_size + 3)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
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
        # attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        # decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()

        return self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # decoder_input_ids=inputs['decoder_input_ids'],
            # decoder_attention_mask=decoder_attention_mask,
            labels=inputs["labels"],
            return_dict=True,
        )

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self(batch)
        loss = outputs.loss
        preds = F.softmax(outputs["logits"].transpose(1, 2), dim=-2)
        labels = batch["labels"]

        # log train metrics
        acc = self.train_accuracy(preds, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss}
        # return loss

    def training_epoch_end(self, outputs: List[Any]):
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(
            self.trainer.callback_metrics["train/loss"]
        )
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self(batch)
        loss = outputs.loss
        preds = F.softmax(outputs["logits"].transpose(1, 2), dim=-2)
        labels = batch["labels"]

        # log train metrics
        acc = self.train_accuracy(preds, labels)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        # log best so far val acc and val loss
        self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
        self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
        self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
        self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    def configure_optimizers(self):
        param_optimzier = list(self.model.named_parameters())
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
