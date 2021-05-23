#%%
from datasets import load_from_disk

r_dataset = load_from_disk('/opt/ml/input/data/data/generated_by_t')
# %%
import kss
import re
from typing import List
why = []
def preprocessing(examples):
    text = examples['text']
    text = [x.replace('\\n\\n',' ') for x in text]
    text = [x.replace('\n\n',' ') for x in text]
    text = [x.replace('\\n',' ') for x in text]
    text = [x.replace('\n',' ') for x in text]
    # text = [' '.join(re.sub(r'''[^ \r\nㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9~₩!@#$%^&*()_+|{}:"<>?`\-=\\[\];',.\/]''', ' ', str(x.lower().strip())).split()) for x in text]
    # text = [' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(x.lower().strip())).split()) for x in text]
    examples['text'] = text 
    return examples
#%%
r_dataset = r_dataset.rename_column('context', 'text')
r_dataset = r_dataset.map(
    preprocessing,
    num_proc=12,
    batched=True
)
r_dataset = r_dataset.sort('document_id')
# %%
why = []
import kss
def split_document(documents):
    # print(document)
    titles, texts, answers, document_ids, questions = [], [], [], [], []
    # title, text, answer, document_id, question = documents['title'], documents['context'], documents['answers'], documents['document_id'], documents['question']
    for title, text, answer, document_id, question in zip(documents['title'], documents['context'], documents['answers'], documents['document_id'], documents['question']):
        try:
            chunks = [chunk for chunk in kss.split_chunks(text, max_length=470)]
            start_idxes = [chunk[0] for chunk in chunks]
            answer_passage_idx = check_answer(start_idxes, answer['answer_start'][0])
            for i, chunk in enumerate(chunks):
                titles.append(title)
                texts.append(chunk[1])
                document_ids.append(f"{document_id}-{i}")
                if i == answer_passage_idx:
                    answers.append(answer['text'][0])
                    questions.append(question)
                else:
                    answers.append('')
                    questions.append('')
        except:
            why.append(document_id)
            titles.append(title)
            texts.append(text)
            answers.append(answer['text'][0])
            questions.append(question)
            document_ids.append(f"{document_id}-0")
    return {
        "title": titles,
        "context": texts,
        "answers": answers,
        "document_id": document_ids,
        "question": questions
    }
    
def check_answer(indexes, start_idx):
    for i, idx in enumerate(indexes):
        if idx > start_idx:
            return i-1
    return i


# %%
r_dataset = r_dataset.rename_column('text', 'context')
r_dataset = r_dataset.map(
    split_document,
    # batch_size=1,
    remove_columns=['id'],
    num_proc=12,
    batched=True
)
r_dataset = r_dataset.rename_column('context', 'text')
# %%
def check(example):
    if example:
        return True
    else:
        return False
r_dataset = r_dataset.filter(check, keep_in_memory=False, input_columns=['question'])
# %%
g_dataset = load_from_disk('/opt/ml/input/data/data/generated')

# %%
g_dataset = g_dataset.rename_column('context', 'text')
def modify_answer(example):
    answer = example['answers']['text'][0]
    example['answers'] = answer
    return example
#%%
g_dataset = g_dataset.map(modify_answer)
# %%
from datasets import concatenate_datasets
rg_dataset = concatenate_datasets([r_dataset, g_dataset])
# %%
r_dataset.save_to_disk('/opt/ml/input/data/data/retriever_dataset')

# %%
