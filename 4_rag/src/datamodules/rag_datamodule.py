from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from datasets import load_dataset

from kobart import get_kobart_tokenizer
from transformers import DataCollatorForSeq2Seq


class QADataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
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
        # self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.tokenizer = get_kobart_tokenizer()

        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            # model=model,
            label_pad_token_id=self.tokenizer.pad_token_id,  # label_pad_token_id,
            pad_to_multiple_of=None,
        )

        self.train_dataset, self.val_dataset = get_dataset(
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


def get_dataset(num_workers, max_target_length, max_source_length, padding):
    datasets = load_dataset("squad_kor_v1")
    column_names = datasets["train"].column_names

    train_dataset = datasets["train"]
    # train_dataset = train_dataset.select(range(16))
    val_dataset = datasets["validation"]

    tokenizer = get_kobart_tokenizer()

    def preprocess_function(examples):
        inputs = [
            f"질문: {q}  글: {c} </s>"
            for q, c in zip(examples["question"], examples["context"])
        ]
        targets = [f'{a["text"][0]} </s>' for a in examples["answers"]]
        model_inputs = tokenizer(
            inputs, max_length=max_source_length, padding=padding, truncation=True
        )  # , return_tensors='pt')

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding=padding, truncation=True
            )  # , return_tensors='pt')

        model_inputs["labels"] = labels["input_ids"]
        # model_inputs["example_id"] = []
        # for i in range(len(model_inputs["labels"])):
        # model_inputs["example_id"].append(examples["id"][i])
        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    # train_dataset.set_format(type='torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])

    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    # val_dataset.set_format(type='torch', columns=['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
    # train_dataset = QADataset(train_dataset)
    # val_dataset = QADataset(val_dataset)
    return train_dataset, val_dataset
