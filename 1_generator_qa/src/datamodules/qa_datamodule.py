from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from datasets import load_dataset, load_from_disk

from kobart import get_kobart_tokenizer
from transformers import DataCollatorForSeq2Seq


class QADataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = '/opt/ml/input/data/data/train_dataset_split',
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = False,
        max_source_length: int = 1024,
        max_target_length: int = 128,
        padding: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.tokenizer = get_kobart_tokenizer()

        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id, 
            pad_to_multiple_of=None,
        )

        self.train_dataset, self.val_dataset = get_dataset(
            path=self.data_dir,
            num_workers=self.num_workers,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            padding=self.padding,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.data_collator,
        )


def get_dataset(path, num_workers, max_target_length, max_source_length, padding):
    dataset = load_from_disk(path)
    column_names = dataset["train"].column_names

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    tokenizer = get_kobart_tokenizer()
    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":["제목:", "글:", "질문:"]})
    def preprocess_function(examples):
        inputs = [
            f"제목: {t} 질문: {q}  글: {c} </s>"
            for t, q, c in zip(examples['title'], examples["question"], examples["text"])
        ]
        targets = [f'{a} </s>' for a in examples["answers"]]
        model_inputs = tokenizer(
            inputs, max_length=max_source_length, padding=padding, truncation=True
        )  # , return_tensors='pt')

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding=padding, truncation=True
            )  # , return_tensors='pt')

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=12,
        remove_columns=column_names,
        load_from_cache_file=False,
    )

    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=12,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    return train_dataset, val_dataset
