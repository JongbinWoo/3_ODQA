#%%
import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from typing import Optional, Tuple
from functools import partial
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from datasets import load_from_disk
from datasets import Features, Sequence, Value, load_dataset, Dataset
from typing import List, Optional
from src.models.retriever_model import DualEncoder
from transformers import ElectraTokenizer
import faiss
#%%

def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    # for title, text in zip(documents["title"], documents["context"]):
    title, text = documents['title'], documents['context']
    if text is not None:
        for passage in split_text(text):
            titles.append(title if title is not None else "")
            texts.append(passage)
    return {"title": titles, "text": texts}

def embed(documents: dict, ctx_encoder, ctx_tokenizer) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="max_length", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to('cuda'))
    return {"embeddings": embeddings.detach().cpu().numpy()}


# import pandas as pd
# import json
# import re
# with open('/opt/ml/code/pytorch_lightning_examples/5_RAG/all_wikipedia_documents.json', 'r') as f:
#     wiki_data = pd.DataFrame(json.load(f)).transpose()    

# # wiki_data['total'] = wiki_data['title'] + ' ' + wiki_data['text']

# wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\\n\\n',' '))
# wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\n\n',' '))
# wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\\n',' '))
# wiki_data['text'] = wiki_data['text'].apply(lambda x : x.replace('\n',' '))
# wiki_data['text'] = wiki_data['text'].apply(lambda x : ' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(x.lower().strip())).split()))

# wiki_data = wiki_data[['text','title']]
# dataset = Dataset.from_pandas(wiki_data)
# dataset = dataset.remove_columns('__index_level_0__')
#%%
from datasets import load_from_disk
dataset = load_from_disk('./remove_duplicate_dataset')
dataset = dataset.remove_columns(['question', 'answers', 'document_id'])
dataset = dataset.rename_column('context', 'text')
#%%
retriever = DualEncoder.load_from_checkpoint('/opt/ml/code/pytorch_lightning_examples/3_RETRIEVER/logs/runs/2021-05-13/18-28-04/checkpoints/epoch=19.ckpt').c_encoder
ctx_encoder = retriever.to('cuda')
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"
ctx_tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
new_features = Features(
        {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
)  # optional, save as float32 instead of float64 to save space
dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=64,
        features=new_features
)
dataset.save_to_disk('./index_folder/my_wiki')
#%%
index = faiss.IndexHNSWFlat(256, 100, faiss.METRIC_INNER_PRODUCT)
dataset.add_faiss_index("embeddings", custom_index=index)

dataset.get_index("embeddings").save('./index_folder/index')

# dataset.load_faiss_index("embeddings", './index_path')

# %%

# %%
# #%%
import pandas as pd
import json
import re
with open('/opt/ml/input/data/data/종빈님_wikipedia_documents.json', 'r') as f:
    wiki_data = pd.DataFrame(json.load(f)).transpose()  
# %%
