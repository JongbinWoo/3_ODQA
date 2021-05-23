#%%
# from models.retriever_model import Retriever
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
from tqdm import tqdm, trange
from datasets import load_from_disk
from transformers import ElectraTokenizer, RagRetriever, BartTokenizer
from models.retriever_model import DualEncoder
from models.retriever_moco_model import Retriever
from datasets import Features, Sequence, Value, load_dataset, Dataset
from functools import partial
#%%
# v_dataset = load_from_disk('/opt/ml/input/data/data/train_dataset')
# v_dataset = v_dataset['validation']
# v_dataset = v_dataset.remove_columns(['answers', 'context', 'document_id', 'title', '__index_level_0__'])
#%%
dataset = load_from_disk('/opt/ml/input/data/data/test_dataset/validation')
# dataset = dataset['validation']
#%%
tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-small-v3-discriminator')
retriever = DualEncoder.load_from_checkpoint('/opt/ml/code/pytorch_lightning_examples/3_RETRIEVER/logs/runs/2021-05-20/07-26-35/checkpoints/epoch=08.ckpt').q_encoder
# retriever = Retriever.load_from_checkpoint('/opt/ml/code/pytorch_lightning_examples/4_MOCO/logs/runs/2021-05-20/13-22-49/checkpoints/epoch=08.ckpt').encoder_q

q_encoder = retriever.to('cuda')
new_features = Features(
    {"question": Value("string"), "id": Value("string"), "embeddings": Sequence(Value("float32"))})
#%%
def encode_line(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )
def embed(documents: dict) -> dict:
    """Compute the DPR embeddings of document passages"""
    source_inputs = encode_line(tokenizer, documents['question'], 256, "right")
    source_ids = source_inputs["input_ids"]
    src_mask = source_inputs["attention_mask"]

    # embeddings = q_encoder(source_ids, attention_mask=src_mask)
    embeddings = q_encoder(source_ids.to('cuda'), attention_mask=src_mask.to('cuda'))
    # documents['embeddings'] = embeddings.detach().numpy()[0]
    documents['embeddings'] = embeddings.detach().cpu().numpy()[0]
    return documents
#%%
dataset = dataset.map(
    embed,
    # batched=True,
    # batch_size=64,
    features=new_features   
)
# %%
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
from transformers import BartForConditionalGeneration, RagConfig
question_encoder_config = q_encoder.config
generator = BartForConditionalGeneration.from_pretrained(
            get_pytorch_kobart_model()
        )
generator_config = generator.config
index_path = "/opt/ml/input/data/data/my_wikipedia_1/index"
passages_path =  "/opt/ml/input/data/data/my_wikipedia_1"
rag_config = RagConfig.from_question_encoder_generator_configs(
    question_encoder_config=question_encoder_config,
    generator_config=generator_config,
    index_name='custom',
    passages_path=passages_path,
    index_path=index_path,
    retrieval_batch_size=4,
    n_docs=20
)

generator_tokenizer = get_kobart_tokenizer()
added_token_num = generator_tokenizer.add_special_tokens({"additional_special_tokens":["제목:", "글:", "질문:"]})


retriever = RagRetriever(
    rag_config,
    question_encoder_tokenizer=tokenizer,
    generator_tokenizer=generator_tokenizer
)
# %%
import numpy as np
dataset = dataset.to_pandas()
#%%
########## DENSE ONLY ###############
import numpy as np
outputs = {}
for _, data in tqdm(dataset.iterrows()):
    data_id = data['id']
    query = np.array([data['embeddings']], dtype=np.float32) 
    _, dense_doc_ids, docs = retriever.retrieve(query, 20)
    texts = docs[0]['text']
    embeddings = docs[0]['embeddings']

    dense_doc_scores = (embeddings@query.T).reshape(1, -1).tolist()[0]
    
    # dense_doc_ids
    outputs[data_id] = [texts, dense_doc_scores]
# %%
######### HYBRID ##############
# import numpy as np
# from tqdm import tqdm
# outputs = {}
# for _, data in tqdm(dataset.iterrows()):
#     data_id = data['id']
#     query = np.array([data['embeddings']], dtype=np.float32) 
#     _, dense_doc_ids, docs = retriever.retrieve(query, 50)
#     # texts = docs[0]['text']
#     embeddings = docs[0]['embeddings']

#     dense_doc_scores = (embeddings@query.T).reshape(1, -1).tolist()[0]
#     sparse_doc_ids = data['ids']
#     sparse_doc_scores = data['scores']

#     doc_ids, doc_scores = reranking_with_sparse(dense_doc_ids.tolist()[0], dense_doc_scores, sparse_doc_ids, sparse_doc_scores)
#     texts = []
#     for doc_id in doc_ids:
#         texts.append(wiki_dataset['text'][doc_id])

#     outputs[data_id] = [texts, doc_scores]
#%%
import json

with open('/opt/ml/input/data/data/dense_retrieval1.json', 'w') as outfile:
    json.dump(outputs, outfile)

# #%%
# import json

# with open('./dense_retrieval_10.json', 'w') as outfile:
#     json.dump(outputs, outfile)
# # %%
# for _, data in tqdm(dataset.iterrows()):
#     data_id = data['id']
#     doc_ids, doc_scores = reranking_with_sparse(outputs[id][0], outputs[id][1], sparse_doc_ids, sparse_doc_scores)
# %%
