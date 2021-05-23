from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer
import random
from datasets import load_from_disk, concatenate_datasets
from transformers import ElectraTokenizer, ElectraModel

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

        self.train_dataset, self.val_dataset, self.context_dataset = get_dataset(self.num_workers)
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # sampler=self.train_sampler,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # sampler=self.val_sampler,
            drop_last=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.context_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # sampler=self.val_sampler,
            drop_last=True,
        )


def get_dataset(num_workers):
    dataset = load_from_disk('/opt/ml/input/data/data/moco_dataset')
    # dataset = dataset.train_test_split(test_size=0.07, shuffle=True)
    SMALL_NAME = "monologg/koelectra-small-v3-discriminator"
    
    tokenizer = ElectraTokenizer.from_pretrained(SMALL_NAME)
    column_names = dataset['train'].column_names
    
    # wiki_dataset = load_from_disk('/opt/ml/input/data/data/')

    def preprocessing(example):
        example['title'] = '제목: ' + example['title'] + ' 글: '
        return example
    
    dataset = dataset.map(preprocessing, keep_in_memory=False) 


    def split_id(examples):
        ids = [example.split("-") for example in examples["document_id"]]
        # title = examples['title']
        new_examples = {
            "article_id": [int(id[0]) for id in ids],
            "paragraph_id": [int(id[1]) for id in ids],
        }
        return new_examples

    def preprocess_question(examples):
        new_examples = split_id(examples)
        new_examples.update(
            tokenizer(examples["question"], padding="max_length", truncation=True, max_length=256)
        )
        return new_examples

    def preprocess_context(examples):
        new_examples = split_id(examples)
        new_examples.update(
            tokenizer(examples['title'], examples["text"], padding="max_length", truncation=True)
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
    # context_dataset.save_to_disk('/opt/ml/input/data/data/moco_dataset_context')
    # question_dataset.save_to_disk('/opt/ml/input/data/data/moco_dataset_question')
    # context_dataset = load_from_disk('/opt/ml/input/data/data/moco_dataset_context')
    # question_dataset = load_from_disk('/opt/ml/input/data/data/moco_dataset_question')
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
        question_dataset["validation"]["input_ids"],
        question_dataset["validation"]["attention_mask"],
        question_dataset["validation"]["token_type_ids"],
        context_dataset["validation"]["input_ids"],
        context_dataset["validation"]["attention_mask"],
        context_dataset["validation"]["token_type_ids"],
        context_dataset["validation"]["article_id"],
        context_dataset["validation"]["paragraph_id"],
    )
    context_dataset = concatenate_datasets([context_dataset['train'], context_dataset['validation']])
    return train_dataset, valid_dataset, context_dataset