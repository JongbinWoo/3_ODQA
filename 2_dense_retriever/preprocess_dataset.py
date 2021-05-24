#%%
############# WIKI ############## 진짜 마지막
import pandas as pd
import json
import re
with open('/opt/ml/input/data/data/wikipedia_documents.json', 'r') as f:
    wiki_data = pd.DataFrame(json.load(f)).transpose()  

# %%
from datasets import Dataset
wiki_data = wiki_data[['text','title', 'document_id']]
dataset = Dataset.from_pandas(wiki_data)
dataset = dataset.remove_columns('__index_level_0__')
#%%
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
def split_document_(documents):
    # print(document)
    titles, texts, document_ids = [], [], []
    for title, text, document_id in zip(documents['title'], documents['text'], documents['document_id']):
        try:
            chunks = [chunk for chunk in kss.split_chunks(text, max_length=470)]
            for i, chunk in enumerate(chunks):
                titles.append(title)
                texts.append(chunk[1])
                document_ids.append(f"{document_id}-{i}")
        except:
            # titles, texts = split_documents(documents)
            # why.append(title)
            titles.append(title)
            texts.append(text)
            document_ids.append(f"{document_id}-0")

    return {
        "title": titles,
        "text": texts,
        "document_id": document_ids
    }

#%%
dataset = dataset.sort('document_id')
dataset = dataset.map(
    preprocessing,
    num_proc=12,
    batched=True
)
#%%
###### 같은 제목, 같은 글인데 document_id가 다른경우 처리
dataset = dataset.to_pandas()
from collections import defaultdict
dict = defaultdict(set)
for _, data in dataset.iterrows():
    dict[data['title']+data['text']].add(data['document_id'])
#%%
duplicate_dict = {}
for k,v in dict.items():
    if len(v) > 1:
        duplicate_dict[k] = v
#%%
change_dict = {}
for v in duplicate_dict.values():
    id_list = list(v)
    id_list = sorted(id_list)
    for id in id_list[1:]:
        change_dict[id] = id_list[0]

remove_ids = list(change_dict.keys())
#%%
def remove_duplicate(example):
    if example in remove_ids:
        return False
    else:
        return True

dataset = Dataset.from_pandas(dataset)
dataset = dataset.filter(remove_duplicate, input_columns=['document_id'])
##################### 중복제거 완료
#%%
#이제 중복없는 진짜 text+context : document_id 쌍 완성
dataset = dataset.to_pandas()
from collections import defaultdict
dict = defaultdict(set)
for _, data in dataset.iterrows():
    dict[data['title']+data['text']].add(data['document_id'])

#%%

# 서일님 생성 데이터 document_id 맞추기
from datasets import load_from_disk

r_dataset = load_from_disk('/opt/ml/input/data/data/generated_by_t')
#%%
# r_dataset = r_dataset.rename_column('context', 'text')
#%%
r_dataset = r_dataset.rename_column('context', 'text')
r_dataset = r_dataset.map(
    preprocessing,
    num_proc=12,
    batched=True
)
r_dataset = r_dataset.sort('document_id')
#%%
def filtering_wiki(example):
    if example['document_id'] > 56736:
        return False
    else:
        return True
def filtering_wiki_(example):
    if example['document_id'] <= 56736:
        return False
    else:
        return True
koquard_dataset = r_dataset.filter(filtering_wiki_)
r_filtered_dataset = r_dataset.filter(filtering_wiki)
#%%
why = []  ### 얘네는 korquad인가?
def fix_document_id(example):
    try:
        new_id = list(dict[example['title']+example['text']])[0]
    except:
        new_id = -1#example['document_id']
        why.append(example)
    example['document_id'] = new_id
    return example
#%%
r_filtered_dataset = r_dataset.map(fix_document_id)
#%%
def filter_why(example):
    if example == -1:
        # print(example)
        return False
    else:
        return True
r_filtered_dataset = r_filtered_dataset.filter(filter_why, input_columns=['document_id'])





#%%
dataset = Dataset.from_pandas(dataset)
dataset = dataset.sort('document_id')
split_dataset = dataset.map(
    split_document_,
    batched=True,
    # batch_size=1,
    num_proc=12
)
#%% 맨위에서 중복처리가 안 되고 split한 후에 중복이 처리되는 경우가 있음
split_dataset = split_dataset.to_pandas()
from collections import defaultdict
split_dict = defaultdict(set)
for _, data in split_dataset.iterrows():
    split_dict[data['title']+data['text']].add(data['document_id'])

duplicate_dict_ = {}
for k,v in split_dict.items():
    if len(v) > 1:
        duplicate_dict_[k] = v
#%%
change_dict_ = {}
for v in duplicate_dict_.values():
    id_list = list(v)
    id_list_ = list(map(lambda x: int(x.split('-')[0]), id_list))
    max_idx = id_list_.index(max(id_list_))
    max_value = id_list.pop(max_idx)
    change_dict_[max_value] = id_list[0]
    if len(id_list) == 2:
        change_dict_[max_value] = id_list[0]
        change_dict_[id_list[1]] = id_list[0]

#%%
remove_ids_ = list(change_dict_.keys())
def remove_duplicate(example):
    if example in remove_ids_:
        return False
    else:
        return True

split_dataset = Dataset.from_pandas(split_dataset)
split_dataset_ = split_dataset.filter(remove_duplicate, input_columns=['document_id'])
#%%
split_dataset_.save_to_disk('/opt/ml/input/data/data/wikipedia_documents_split')

##################################################################################3
#%%
# split된 wiki의 찐 id
split_dataset = split_dataset_.to_pandas()
split_dict = defaultdict(set)
for _, data in split_dataset.iterrows():
    split_dict[data['title']+data['text']].add(data['document_id'])

# #########################  RETRIEVER
#%%
r_filtered_dataset = r_filtered_dataset.rename_column('text', 'context')
r_filtered_split = r_filtered_dataset.map(
    split_document,
    # batch_size=1,
    remove_columns=['id'],
    num_proc=12,
    batched=True
)
#%%
def check(example):
    if example:
        return True
    else:
        return False
#%%
r_filtered_split = r_filtered_split.filter(check, keep_in_memory=False, input_columns=['question'])
#%%
r_filtered_split = r_filtered_split.rename_column('context', 'text')
def change_document_id(example):
    new_id = list(split_dict[example['title']+example['text']])[0]
    example['document_id'] = new_id
    return example
r_filtered_split = r_filtered_split.map(change_document_id)
#%%
#### 내가 생성한 데이터 합치기
g_dataset = load_from_disk('/opt/ml/input/data/data/generated')
#%%
def change_document_id(example):
    new_id = list(split_dict[example['title']+example['context']])[0]
    example['document_id'] = new_id
    return example

g_dataset = g_dataset.map(change_document_id)
g_dataset = g_dataset.rename_column('context', 'text')
def modify_answer(example):
    answer = example['answers']['text'][0]
    example['answers'] = answer
    return example
#%%
g_dataset = g_dataset.map(modify_answer)
#%%
from datasets import concatenate_datasets
rg_dataset = concatenate_datasets([r_filtered_split, g_dataset])
#%%
# rg_dataset.save_to_disk('/opt/ml/input/data/data/moco_dataset')
# %%
koquard_dataset = koquard_dataset.remove_columns(['id'])
# koquard_dataset = koquard_dataset.map(modify_answer)
koquard_dataset = koquard_dataset.rename_column('text', 'context')
#%%
koquard_dataset_split = koquard_dataset.map(
    split_document,
    # batch_size=1,
    num_proc=12,
    batched=True
)
koquard_dataset_split = koquard_dataset_split.rename_column('context', 'text')
koquard_dataset_split = koquard_dataset_split.filter(check, keep_in_memory=False, input_columns=['question'])
#%%
retriever_dataset = concatenate_datasets([rg_dataset, koquard_dataset_split])
retriever_dataset.save_to_disk('/opt/ml/input/data/data/retriever_dataset')





















#%%
r_processed_dataset = r_dataset.map(
    preprocessing,
    num_proc=12,
    batched=True
)
# %%
def change_document_id(example):
    doc_id = example['document_id']
    if doc_id in remove_ids:
        example['document_id'] = change_dict[doc_id]
    return example

r_dataset_changed = r_processed_dataset.map(change_document_id)
r_dataset_changed = r_dataset_changed.sort('document_id')
# %%
# def filtering_wiki(example):
#     if example['document_id'] > 56736:
#         return False
#     else:
#         return True

# r_filtered_dataset = r_processed_dataset.filter(filtering_wiki)
#%%
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
r_split_dataset = r_dataset_changed.map(
    split_document,
    # batch_size=1,
    remove_columns=['id'],
    num_proc=12,
    batched=True
)
r_split_dataset = r_split_dataset.rename_column('context', 'text')
# %%
def check(example):
    if example:
        return True
    else:
        return False
r_dataset = r_split_dataset.filter(check, keep_in_memory=False, input_columns=['question'])
#%%
r_dataset.save_to_disk('/opt/ml/input/data/data/retriever_final_split')
# ng = r_split_dataset.filter(check, keep_in_memory=False, input_columns=['question'])
#%%
from datasets import load_from_disk
# r_dataset = load_from_disk('/opt/ml/input/data/data/retriever_final_split')
r_dataset_df = r_dataset.to_pandas() 
#%%
###### 같은 제목, 같은 글인데 document_id가 다른경우 처리
from collections import defaultdict
dict = defaultdict(set)
for _, data in r_dataset_df.iterrows():
    dict[data['title']+data['text']].add(data['document_id'])
#%%
duplicate_dict = {}
for k,v in dict.items():
    if len(v) > 1:
        duplicate_dict[k] = v
#%%
change_dict = {}
for v in duplicate_dict.values():
    id_list = list(v)
    id_list_ = list(map(lambda x: int(x.split('-')[0]), id_list))
    max_idx = id_list_.index(max(id_list_))
    max_value = id_list.pop(max_idx)
    change_dict[max_value] = id_list[0]
    if len(id_list) == 2:
        change_dict[max_value] = id_list[0]
        change_dict[id_list[1]] = id_list[0]
#%%
for _, data in r_dataset_df.iterrows():
    if data['document_id'] in change_dict.keys():
        data['document_id'] = change_dict[data['document_id']]
from datasets import Dataset
r_dataset = Dataset.from_pandas(r_dataset_df)
r_dataset.save_to_disk('/opt/ml/input/data/data/retriever_final_split')
######### wiki에서 질문이 생성안된 지문을 생성해와서 합쳐준다.

## 위에 거로 거르고 합쳐야한다!!

############## 

####생성안된 지문 찾기
#%%
ids = set()
def check_duplicate(example):
    context = example['title'] + example['text']
    if context in ids:
        return False
    else:
        ids.add(context)
        return True

ng_unique = ng.filter(check_duplicate, keep_in_memory=False)
ng_id_list = list(ng_unique['document_id'])
# %%
def check(example):
    if example:
        return True
    else:
        return False

exist = r_split_dataset.filter(check, keep_in_memory=False, input_columns=['question'])
# %%
exist_unique = exist.filter(check_duplicate, keep_in_memory=False)
# %%
exist_id_list = list(exist_unique['document_id'])
# %%
ng_set = set(ng_id_list)
e_set = set(exist_id_list)
ng_real_list = list(ng_set.difference(e_set))
# %%
def get_ng(example):
    if example in ng_real_list:
        return True
    else:
        return False

ng_dataset = filtered_dataset.filter(get_ng, input_columns=['document_id'])
# %%
ng_dataset.save_to_disk('/opt/ml/input/data/data/generation_needed')

# %%
###### MOCO       document_id 기준으로 정렬
from datasets import load_from_disk
## 위에 거로 거르고 합쳐야한다!!


m_dataset = load_from_disk('/opt/ml/input/data/data/retriever_final_split')
# %%
def extract_document_id(example):
    document_id = int(example['document_id'].split('-')[0])
    example['id_num'] = document_id
    return example

m_dataset = m_dataset.map(extract_document_id)
# %%
def filtering_wiki(example):
    if example > 56736:
        return False
    else:
        return True

m_dataset_filtered = m_dataset.filter(filtering_wiki, input_columns='id_num')
# %%
m_dataset_sorted = m_dataset_filtered.sort('id_num')
# %%
m_dataset_sorted = m_dataset_sorted.remove_columns(['id_num'])
# %%
m_dataset_sorted.save_to_disk('/opt/ml/input/data/data/moco_dataset')
# %%


############### validation split
v_dataset = load_from_disk('/opt/ml/input/data/data/train_dataset/validation')
v_questions = set(v_dataset['question'])
def validation_filtering(example):
    if example['question'] in v_questions:
        return False
    else:
        return True

r_dataset_filtered = retriever_dataset.filter(validation_filtering)

def validation_filtering(example):
    if example['question'] in v_questions:
        return True
    else:
        return False

v_dataset = retriever_dataset.filter(validation_filtering)

from datasets import DatasetDict
retriever = DatasetDict()
retriever['train'] = r_dataset_filtered
retriever['test'] = v_dataset
retriever.save_to_disk('/opt/ml/input/data/data/retriever_dataset_final')