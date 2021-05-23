from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, Dataset
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer, ElectraTokenizer
import random
from pytorch_lightning.utilities.seed import seed_everything


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
        self.seed = 6
        self.train_dataset, self.val_dataset = get_dataset(self.num_workers)
        self.type = 'easy'
    def train_dataloader(self):
        self.seed += 1
        seed_everything(self.seed)
        self.type = 'easy' if self.seed % 2 else 'hard'
        
        if self.type == "hard":
            print("HARD SAMPLER")
            sampler = HardNegativeSampler(self.train_dataset, self.batch_size)
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=sampler,
                drop_last=True,
            )
        
        elif self.type == "easy":
            print("EASY SAMPLER")
            sampler = BasicSampler(self.train_dataset, self.batch_size)
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=sampler,
                drop_last=True,
            )

    def val_dataloader(self):
        if self.type == "hard":
            sampler = HardNegativeSampler(self.val_dataset, self.batch_size)
            return DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=sampler,
                drop_last=True,
            )
        elif self.type == "easy":
            sampler = BasicSampler(self.val_dataset, self.batch_size)
            return DataLoader(
                    dataset=self.val_dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    sampler=sampler,
                    drop_last=True,
                )
    def test_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )


def get_dataset(num_workers):
    dataset = load_from_disk('/opt/ml/input/data/data/retriever_dataset_final')
    print(dataset)
    SMALL_NAME = "monologg/koelectra-small-v3-discriminator"
    
    c_tokenizer = ElectraTokenizer.from_pretrained(SMALL_NAME)
    q_tokenizer = ElectraTokenizer.from_pretrained(SMALL_NAME)
    column_names = dataset['train'].column_names
    
    def preprocessing(example):
        example['title'] = '제목: ' + example['title'] + ' 글: '
        return example
    
    dataset = dataset.map(preprocessing, keep_in_memory=False) 


    def split_id(examples):
        ids = [example.split("-") for example in examples["document_id"]]
        new_examples = {
            "article_id": [int(id[0]) for id in ids],
            "paragraph_id": [int(id[1]) for id in ids],
        }
        return new_examples

    def preprocess_question(examples):
        new_examples = split_id(examples)
        new_examples.update(
            q_tokenizer(examples["question"], padding="max_length", truncation=True, max_length=256)
        )
        return new_examples

    def preprocess_context(examples):
        new_examples = split_id(examples)
        new_examples.update(
            c_tokenizer(examples['title'], examples["text"], padding="max_length", truncation=True)
        )
        return new_examples

    question_dataset = dataset.map(
        preprocess_question,
        batched=True,
        num_proc=12,
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
        num_proc=12,
        remove_columns=column_names,
        load_from_cache_file=False,
    )

    context_dataset.set_format(
        type="torch",
        columns=["attention_mask", "input_ids", "token_type_ids", "article_id", "paragraph_id"],
    )
    context_dataset.save_to_disk('/opt/ml/input/data/data/retriever_dataset_context')
    question_dataset.save_to_disk('/opt/ml/input/data/data/retriever_dataset_question')
    # context_dataset = load_from_disk('/opt/ml/input/data/data/final_dataset_split_context')
    # question_dataset = load_from_disk('/opt/ml/input/data/data/final_dataset_split_question')
    # context_dataset['train'] = context_dataset['train'].select(range(80))
    # context_dataset['test'] = context_dataset['test'].select(range(80))
    # question_dataset['train'] = question_dataset['train'].select(range(80))
    # question_dataset['test'] = question_dataset['test'].select(range(80))
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
        question_dataset["test"]["input_ids"],
        question_dataset["test"]["attention_mask"],
        question_dataset["test"]["token_type_ids"],
        context_dataset["test"]["input_ids"],
        context_dataset["test"]["attention_mask"],
        context_dataset["test"]["token_type_ids"],
        context_dataset["test"]["article_id"],
        context_dataset["test"]["paragraph_id"],
    )
    return train_dataset, valid_dataset
# 

import torch
import torch.utils.data
from collections import defaultdict
import numpy as np


class BasicSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size):

        self.indices = list(range(len(dataset)))

        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.batch_size = batch_size
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
    def __init__(self, dataset, batch_size):
        self.indices = list(range(len(dataset)))

        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.batch_size = batch_size
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
