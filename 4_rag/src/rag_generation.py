#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.models.bart_model import KoBARTConditionalGeneration
from src.models.retriever_model import DualEncoder
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model

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

def load_pretrained_model():
    encoder_path = "/opt/ml/code/pytorch_lightning_examples/3_RETRIEVER/logs/runs/2021-05-15/12-15-53/checkpoints/epoch=04.ckpt"
    generator_path = "/opt/ml/code/pytorch_lightning_examples/2_QA/logs/runs/2021-05-14/15-36-30/checkpoints/last.ckpt"

    question_encoder = DualEncoder.load_from_checkpoint(encoder_path).q_encoder
    generator = KoBARTConditionalGeneration.load_from_checkpoint(generator_path)
    return question_encoder, generator.model

question_encoder, generator = load_pretrained_model()

question_encoder_config = question_encoder.config
generator_config = generator.config

rag_config = RagConfig.from_question_encoder_generator_configs(
    question_encoder_config=question_encoder_config,
    generator_config=generator_config,
    index_name='custom',
    passages_path="/opt/ml/input/data/data/my_wikipedia",
    index_path="/opt/ml/input/data/data/my_wikipedia/index",
    retrieval_batch_size=1,
    n_docs=32,
    doc_sep="질문: ",
    title_sep=" 글: "
)

question_encoder_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
generator_tokenizer = get_kobart_tokenizer()
added_token_num = generator_tokenizer.add_special_tokens({"additional_special_tokens":["제목:", "글:", "질문:"]})


retriever = RagRetriever(
    rag_config,
    question_encoder_tokenizer=question_encoder_tokenizer,
    generator_tokenizer=generator_tokenizer
)
model = RagSequenceForGeneration(
    config=rag_config,
    question_encoder=question_encoder,
    generator=generator,
    retriever=retriever
)
tokenizer = RagTokenizer(
    question_encoder=question_encoder_tokenizer,
    generator=generator_tokenizer
)

# %%
from datasets import load_from_disk
dataset = load_from_disk('/opt/ml/input/data/data/test_dataset')
dataset = dataset['validation']
# %%
result = {}
model.to('cuda')
import torch
from tqdm import tqdm
for data in tqdm(dataset):
    with torch.no_grad():
        inputs_dict = model.retriever.question_encoder_tokenizer.batch_encode_plus([data['question']], return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs_dict['input_ids'].to('cuda')
        attention_mask = inputs_dict['attention_mask'].to('cuda')
        outputs = model.generate(
            input_ids,
            attention_mask,
            min_length=1,
            max_length=20,
            early_stopping=False,
            # num_beams=2,
            num_return_sequences=1,
            bad_words_ids=[[0, 0]]
        )
        answers = model.retriever.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    question = data['question']
    print(f'Q: {question} A: {answers}')
    result[data['id']] = answers
# %%
import json
with open('./prediction.json', 'w') as f:
    json.dump(result, f)

with open('/opt/ml/input/data/data/predictions.json', 'w') as f:
    f.write(json.dumps(result, indent=4, ensure_ascii=False) + '\n')
