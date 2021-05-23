
# %%
from datasets import Features, Sequence, Value, load_from_disk, Dataset
from typing import List, Optional
from src.models.retriever_model import DualEncoder
from transformers import ElectraTokenizer
import faiss
from src.models.retriever_moco_model import Retriever
filtered_dataset = load_from_disk('/opt/ml/input/data/data/wikipedia_documents_split/')
def preprocessing(example):
        example['title'] = '제목: ' + example['title'] + ' 글: '
        return example
    
filtered_dataset = filtered_dataset.map(preprocessing, keep_in_memory=False) 
retriever = DualEncoder.load_from_checkpoint('/opt/ml/code/pytorch_lightning_examples/3_RETRIEVER/logs/runs/2021-05-20/07-26-35/checkpoints/epoch=08.ckpt').c_encoder
ctx_encoder = retriever.to('cuda')
MODEL_NAME = "monologg/koelectra-small-v3-discriminator"
ctx_tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
# added_token_num = ctx_tokenizer.add_special_tokens({"additional_special_tokens":["제목:", "글:"]})

#%%
new_features = Features(
        {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
) 
# %%
from functools import partial
def embed(documents: dict, ctx_encoder, ctx_tokenizer) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to('cuda'))
    return {"embeddings": embeddings.detach().cpu().numpy()}

index_dataset = filtered_dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=64,
        features=new_features,
        remove_columns=['document_id']
)
#%%
index_dataset.save_to_disk('/opt/ml/input/data/data/my_wikipedia_1')
index = faiss.IndexHNSWFlat(256, 128, faiss.METRIC_INNER_PRODUCT)
index_dataset.add_faiss_index("embeddings", custom_index=index)
index_dataset.get_index("embeddings").save('/opt/ml/input/data/data/my_wikipedia_1/index')
# %%
# from datasets import load_from_disk
# dataset = load_from_disk('/opt//ml/input/data/data/my_wikipedia')
# %%
# from datasets import load_from_disk
# index_dataset = load_from_disk('/opt/ml/input/data/data/my_wikipedia')
# def get_length(example):
#     example['len'] = len(example['text'])

# index_dataset = index_dataset.map(get_length, keep_in_memory=False)
# # %%



# def split_documents(documents: dict) -> dict:
#     """Split documents into passages"""
#     titles, texts = [], []
#     for title, text in zip(documents["title"], documents["text"]):
#         # title = '제목: ' + title + ' '
#         if text is not None:
#             for passage in split_text(text):
#                 titles.append(title if title is not None else "")
#                 texts.append(passage)
#     return {
#         "title": titles,
#         "text": texts,
#     }

# def split_text(text: str, n=100, character=" ") -> List[str]:
#     """Split the text every ``n``-th occurrence of ``character``"""
#     text = text.split(character)
#     result = []
#     for i in range(0, len(text), n):
#         result.append(character.join(text[i: i+n]).strip())
#     return result