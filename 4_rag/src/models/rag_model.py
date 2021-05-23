import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from src.models.utils_rag import *
from src.models.bart_model import KoBARTConditionalGeneration
from src.models.retriever_model import DualEncoder

from transformers import (
    AdamW,
    ElectraModel,
    ElectraTokenizer,
    BartForConditionalGeneration,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
    RagConfig,
    BatchEncoding
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model


class Rag(pl.LightningModule):
    loss_names = ['loss']
    metric_names = ['em']
    val_metric = 'em'

    def __init__(
        self,
        encoder_model_name,
        index_path,
        passages_path,
        encoder_path,
        generator_path,
        retrieval_batch_size,
        train_batch_size,
        eval_batch_size,
        n_docs,
        max_target_length,
        max_source_length,
        val_max_target_length,
        num_workers,
        max_epochs,
        accumulate_grad_batches,
        lr_scheduler,
        warmup_steps,
        weight_decay,
        learning_rate,
        adam_epsilon,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.step_count = 0

        self.dataset_size = 40000

        question_encoder, generator = self._load_pretrained_model()

        question_encoder_config = question_encoder.config
        generator_config = generator.config

        rag_config = RagConfig.from_question_encoder_generator_configs(
            question_encoder_config=question_encoder_config,
            generator_config=generator_config,
            index_name='custom',
            passages_path=passages_path,
            index_path=index_path,
            retrieval_batch_size=retrieval_batch_size,
            n_docs=n_docs,
            doc_sep="질문: ",
            title_sep=" 글: "
        )

        question_encoder_tokenizer = ElectraTokenizer.from_pretrained(encoder_model_name)
        generator_tokenizer = get_kobart_tokenizer()
        added_token_num = generator_tokenizer.add_special_tokens({"additional_special_tokens":["제목:", "글:", "질문:"]})


        retriever = RagRetriever(
            rag_config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer
        )

        self.model = RagSequenceForGeneration(
            config=rag_config,
            question_encoder=question_encoder,
            generator=generator,
            retriever=retriever
        )

        self.tokenizer = RagTokenizer(
            question_encoder=question_encoder_tokenizer,
            generator=generator_tokenizer
        )
        self.target_lens = {
            'train': self.hparams.max_target_length,
            'val': self.hparams.val_max_target_length,
            'test': self.hparams.val_max_target_length
        }
        self.output_dir = Path(self.hparams.output_dir)
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        # pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
    
    def _load_pretrained_model(self):
        question_encoder = DualEncoder.load_from_checkpoint(self.hparams.encoder_path).q_encoder
        generator = KoBARTConditionalGeneration.load_from_checkpoint(self.hparams.generator_path) #.model
        # generator.freeze()
        return question_encoder, generator.model

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)
    
    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch):
        source_ids, source_mask, target_ids = batch['input_ids'], batch['attention_mask'], batch['decoder_input_ids']

        decoder_input_ids = target_ids
        lm_labels = decoder_input_ids

        outputs = self(
            source_ids,
            attention_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            labels=lm_labels,
            reduce_loss=True,
            n_docs=self.hparams.n_docs
        )

        loss = outputs['loss']
        return (loss,)
    
    def training_step(self, batch, batch_idx):
        loss_tensors = self._step(batch)

        logs = {name:loss for name, loss in zip(self.loss_names, loss_tensors)}

        tgt_pad_token_id = (
            self.tokenizer.generator.pad_token_id
        )

        src_pad_token_id = (
            self.tokenizer.question_encoder.pad_token_id
        )

        logs['tpb'] = (
            batch['input_ids'].ne(src_pad_token_id).sum() + batch['decoder_input_ids'].ne(tgt_pad_token_id).sum()
        )
        self.log("train/loss", loss_tensors[0], on_step=True, on_epoch=True, prog_bar=False)
        return {'loss': loss_tensors[0], 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        # import time
        # time.sleep(60)
        return self._generative_step(batch)
    
    def validation_epoch_end(self, outputs, prefix='val'):
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean() for k in self.loss_names}
        loss = losses['loss']
        metric_names = ['gen_time', 'gen_len']
        metric_names.extend(self.metric_names)
        gen_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in metric_names
        }
        metrics_tensor = torch.tensor(gen_metrics[self.val_metric]).type_as(loss)
        gen_metrics.update({k: v.item() for k, v in losses.items()})

        losses.update(gen_metrics)
        metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        metrics["step_count"] = self.step_count
        self.save_metrics(metrics, prefix)  # writes to self.metrics_save_path
        preds = flatten_list([x["preds"] for x in outputs])
        self.log('em', gen_metrics['em'])
        return {"log": metrics, "preds": preds, f"{prefix}_loss": loss, f"{prefix}_{self.val_metric}": metrics_tensor}

    def save_metrics(self, latest_metrics, type_path) -> None:
        self.metrics[type_path].append(latest_metrics)
        # save_json(self.metrics, self.metrics_save_path)  
        print(self.metrics)

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_exact_match(preds, target)

    def _generative_step(self, batch):
        start_time = time.time()
        batch = BatchEncoding(batch).to(self.device)
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            do_deduplication=False,
            use_cache=True,
            min_length=1,
            max_length=self.target_lens['val'],
            n_docs=5
        )

        gen_time = (time.time() - start_time) / batch['input_ids'].shape[0]
        preds = self.ids_to_clean_text(generated_ids)
        targets = self.ids_to_clean_text(batch['decoder_input_ids'])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name, loss in zip(self.loss_names, loss_tensors)}
        gen_metrics: Dict = self.calc_generative_metrics(preds, targets)

        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len, preds=preds, target=targets, **gen_metrics)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix='test')
    
    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = 1
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches
        return (self.dataset_size / effective_batch_size) * self.hparams.max_epochs
        
    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps()
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # if self.hparams.adafactor:
        #     optimizer = Adafactor(
        #         optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False
        #     )

        # else:
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon
        )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]
    
    def get_dataset(self, type_path) -> Seq2SeqDataset:
        max_target_length = self.target_lens[type_path]
        dataset = Seq2SeqDataset(
            self.tokenizer,
            type_path=type_path,
            max_target_length=100,#max_target_length,
            max_source_length=self.hparams.max_source_length
        )
        return dataset
    
    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams.eval_batch_size)


