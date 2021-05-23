from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer, ElectraTokenizer
from tokenization_kobert import KoBertTokenizer
import random

class RetrieverDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        pin_memory: bool = False,
        **kwargs
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset, self.val_dataset = get_dataset(self.num_workers)
        self.train_sampler = HardNegativeSampler(self.train_dataset, self.batch_size)
        self.val_sampler = HardNegativeSampler(self.val_dataset, self.batch_size)
        # self.train_sampler = BasicSampler(self.train_dataset, self.batch_size)
        # self.val_sampler = BasicSampler(self.val_dataset, self.batch_size)
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            # shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.train_sampler,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            # shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.val_sampler,
            drop_last=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            # shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # sampler=self.val_sampler,
            drop_last=True,
        )


def get_dataset(num_workers):
    dataset = load_dataset("squad_kor_v1")
    dataset = dataset.sort("id")
    # dataset['train'] = dataset['train'].select(range(80))
    # dataset['validation'] = dataset['validation'].select(range(80))
    # corpus = list(set(example['context'] for example in dataset['train']))
    column_names = dataset["train"].column_names
    MODEL_NAME = "monologg/koelectra-small-v3-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

    def split_id(examples):
        ids = [example.split("-") for example in examples["id"]]
        new_examples = {
            "article_id": [int(id[0]) for id in ids],
            "paragraph_id": [int(id[1]) for id in ids],
        }
        return new_examples

    def preprocess_question(examples):
        new_examples = split_id(examples)
        new_examples.update(
            tokenizer(examples["question"], padding="max_length", truncation=True)
        )
        return new_examples

    def preprocess_context(examples):
        new_examples = split_id(examples)
        new_examples.update(
            tokenizer(examples["context"], padding="max_length", truncation=True)
        )
        return new_examples

    question_dataset = dataset.map(
        preprocess_question,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    question_dataset.set_format(
        type="torch",
        columns=["attention_mask", "input_ids", "token_type_ids", "article_id", "paragraph_id"],
    )
    context_dataset = dataset.map(
        preprocess_context,
        batched=True,
        num_proc=num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
    )
    # assert context_dataset['train']['article_id'] != question_dataset['train']['article_id'], Check

    context_dataset.set_format(
        type="torch",
        columns=["attention_mask", "input_ids", "token_type_ids", "article_id", "paragraph_id"],
    )
    train_dataset = TensorDataset(
        question_dataset["train"]["input_ids"],
        question_dataset["train"]["attention_mask"],
        question_dataset["train"]["token_type_ids"],
        context_dataset["train"]["input_ids"],
        context_dataset["train"]["attention_mask"],
        context_dataset["train"]["token_type_ids"],
        context_dataset["train"]["article_id"],
        context_dataset["train"]["paragraph_id"],
    )
    valid_dataset = TensorDataset(
        question_dataset["validation"]["input_ids"],
        question_dataset["validation"]["attention_mask"],
        question_dataset["validation"]["token_type_ids"],
        context_dataset["validation"]["input_ids"],
        context_dataset["validation"]["attention_mask"],
        context_dataset["validation"]["token_type_ids"],
        context_dataset["validation"]["article_id"],
        context_dataset["validation"]["paragraph_id"],
    )
    return train_dataset, valid_dataset


import torch
import torch.utils.data
from collections import defaultdict
import numpy as np


class BasicSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, batch_size):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.batch_size = batch_size
        # distribution of classes in the dataset
        self.label_to_idx = defaultdict(list)
        for idx in self.indices:
            label = self._get_label(idx)
            self.label_to_idx[label].append(idx)

    def _get_label(self, idx):
        return int(self.dataset[idx][-2].numpy())

    def __iter__(self):
        return (
            np.random.choice(self.label_to_idx[j], 1)[0]
            for i in range(self.num_samples // self.batch_size)
            for j in np.random.choice(
                list(self.label_to_idx.keys()), self.batch_size, replace=False
            )
        )

    def __len__(self):
        return self.num_samples


class HardNegativeSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, batch_size):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.batch_size = batch_size
        # distribution of classes in the dataset
        self.label_to_idx = defaultdict(lambda: defaultdict(list))
        for idx in self.indices:
            article = self._get_article(idx)
            passage = self._get_passage(idx)
            self.label_to_idx[article][passage].append(idx)
        self.random_num = random.randint(0, self.batch_size-1)
        
    def _get_article(self, idx):
        return int(self.dataset[idx][-2].numpy())

    def _get_passage(self, idx):
        return int(self.dataset[idx][-1].numpy())

    def __iter__(self):
        for i in range(self.num_samples // self.batch_size):
            r_list = np.random.choice(list(self.label_to_idx.keys()), self.batch_size, replace=False)
            for k, j in enumerate(r_list):
                if k % 2 == 0:
                    r = np.random.choice(list(self.label_to_idx[j].keys()), 1)[0]
                    yield np.random.choice(self.label_to_idx[j][r], 1)[0]
                else:
                    j = r_list[k-1]
                    while True:
                        r_ = np.random.choice(list(self.label_to_idx[j].keys()), 1)[0]
                        if r_ != r:
                            r = r_
                            break
                        if len(list(self.label_to_idx[j].keys())) == 1:
                            j = r_list[k]
                            r = np.random.choice(list(self.label_to_idx[j].keys()), 1)[0]
                            break
                    yield np.random.choice(self.label_to_idx[j][r], 1)[0]

    def __len__(self):
        return self.num_samples
