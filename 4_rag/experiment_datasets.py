#%%

# ######################### 1번 TRAIN
from datasets import load_from_disk

dataset = load_from_disk('/opt/ml/input/data/data/retriever_dataset')
# dataset = load_from_disk('/opt/ml/input/data/data/final_dataset')
#%%
import re
def preprocessing(examples):
    text = examples['context']
    text = [x.replace('\\n\\n',' ') for x in text]
    text = [x.replace('\n\n',' ') for x in text]
    text = [x.replace('\\n',' ') for x in text]
    text = [x.replace('\n',' ') for x in text]
    # wiki_data['text'] = wiki_data['text'].apply(lambda x : ' '.join(re.sub(r'''[^ \r\nㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9~₩!@#$%^&*()_+|{}:"<>?`\-=\\[\];',.\/]''', ' ', str(x.lower().strip())).split()))
    text = [' '.join(re.sub(r'''[^ \r\nㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9~₩!@#$%^&*()_+|{}:"<>?`\-=\\[\];',.\/]''', ' ', str(x.lower().strip())).split()) for x in text]
    examples['context'] = text 
    return examples
#%%
processed_dataset = dataset.map(
    preprocessing,
    num_proc=12,
    batched=True
)
# %%
why = []
import kss
def split_document(documents):
    # print(document)
    titles, texts, answers, document_ids, questions = [], [], [], [], []
    # title, text, answer, document_id, question = documents['title'], documents['context'], documents['answers'], documents['document_id'], documents['question']
    for title, text, answer, document_id, question in zip(documents['title'], documents['context'], documents['answers'], documents['document_id'], documents['question']):
        try:
            chunks = [chunk for chunk in kss.split_chunks(text, max_length=500)]
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
split_dataset = processed_dataset.map(
    split_document,
    # batch_size=1,
    remove_columns=['id'],
    num_proc=12,
    batched=True
)
split_dataset = split_dataset.rename_column('context', 'text')
# %%
# import numpy as np
# import pandas as pd
# len_list = [len(i) for i in split_dataset['train']['text']]
# len_list = np.array(len_list)
# len_list_s = pd.Series(len_list)

#%%
def check(example):
    if example:
        return False
    else:
        return True

filtered_dataset_ = split_dataset.filter(check, keep_in_memory=False, input_columns=['question'])
#%%
filtered_dataset_.save_to_disk('/opt/ml/input/data/data/generation_needed')
# filtered_dataset.save_to_disk('/opt/ml/input/data/data/retriever_dataset_split')
#%%
# ######################### 2번 RETRIEVER
# from datasets import load_from_disk

# dataset = load_from_disk('/opt/ml/input/data/data/retriever_dataset')
# #%%
# def preprocessing(examples):
#     text = examples['context']
#     text = [x.replace('\\n\\n',' ') for x in text]
#     text = [x.replace('\n\n',' ') for x in text]
#     text = [x.replace('\\n',' ') for x in text]
#     text = [x.replace('\n',' ') for x in text]
#     # text = [' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(x.lower().strip())).split()) for x in text]
#     examples['context'] = text 
#     return examples
# #%%
# processed_dataset = dataset.map(
#     preprocessing,
#     num_proc=12,
#     batched=True
# )
# # %%
# why = []
# import kss
# def split_document(documents):
#     # print(document)
#     titles, texts, answers, document_ids, questions = [], [], [], [], []
#     # title, text, answer, document_id, question = documents['title'], documents['context'], documents['answers'], documents['document_id'], documents['question']
#     for title, text, answer, document_id, question in zip(documents['title'], documents['context'], documents['answers'], documents['document_id'], documents['question']):
#         try:
#             chunks = [chunk for chunk in kss.split_chunks(text, max_length=500)]
#             start_idxes = [chunk[0] for chunk in chunks]
#             answer_passage_idx = check_answer(start_idxes, answer['answer_start'][0])
#             for i, chunk in enumerate(chunks):
#                 titles.append(title)
#                 texts.append(chunk[1])
#                 document_ids.append(f"{document_id}-{i}")
#                 if i == answer_passage_idx:
#                     answers.append(answer['text'][0])
#                     questions.append(question)
#                 else:
#                     answers.append('')
#                     questions.append('')
#         except:
#             why.append(document_id)
#             titles.append(title)
#             texts.append(text)
#             answers.append(answer['text'][0])
#             questions.append(question)
#             document_ids.append(f"{document_id}-0")
#     return {
#         "title": titles,
#         "context": texts,
#         "answers": answers,
#         "document_id": document_ids,
#         "question": questions
#     }
    
# def check_answer(indexes, start_idx):
#     for i, idx in enumerate(indexes):
#         if idx > start_idx:
#             return i-1
#     return i

# # %%
# split_dataset = processed_dataset.map(
#     split_document,
#     # batch_size=1,
#     remove_columns=['id'],
#     num_proc=12,
#     batched=True
# )
# split_dataset = split_dataset.rename_column('context', 'text')
# # %%
# import numpy as np
# import pandas as pd
# len_list = [len(i) for i in split_dataset['text']]
# len_list = np.array(len_list)
# len_list_s = pd.Series(len_list)

# #%%
# def check(example):
#     if example:
#         return True
#     else:
#         return False

# filtered_dataset = split_dataset.filter(check, keep_in_memory=False, input_columns=['question'])
# #%%
# filtered_dataset.save_to_disk('/opt/ml/input/data/data/retriever_dataset_split')

# # %%
#%%

# ######################### 1번 TRAIN
from datasets import load_from_disk

dataset = load_from_disk('/opt/ml/input/data/data/retriever_dataset')
# dataset = load_from_disk('/opt/ml/input/data/data/final_dataset')
#%%
import re
def preprocessing(examples):
    text = examples['context']
    text = [x.replace('\\n\\n',' ') for x in text]
    text = [x.replace('\n\n',' ') for x in text]
    text = [x.replace('\\n',' ') for x in text]
    text = [x.replace('\n',' ') for x in text]
    # wiki_data['text'] = wiki_data['text'].apply(lambda x : ' '.join(re.sub(r'''[^ \r\nㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9~₩!@#$%^&*()_+|{}:"<>?`\-=\\[\];',.\/]''', ' ', str(x.lower().strip())).split()))
    text = [' '.join(re.sub(r'''[^ \r\nㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9~₩!@#$%^&*()_+|{}:"<>?`\-=\\[\];',.\/]''', ' ', str(x.lower().strip())).split()) for x in text]
    examples['context'] = text 
    return examples
#%%
processed_dataset = dataset.map(
    preprocessing,
    num_proc=12,
    batched=True
)
# %%
def filtering_wiki(example):
    if example['document_id'] > 56736:
        return False
    else:
        return True

filtered_dataset = processed_dataset.filter(filtering_wiki)

#%%

why = []
import kss
def split_document(documents):
    # print(document)
    titles, texts, answers, document_ids, questions = [], [], [], [], []
    # title, text, answer, document_id, question = documents['title'], documents['context'], documents['answers'], documents['document_id'], documents['question']
    for title, text, answer, document_id, question in zip(documents['title'], documents['context'], documents['answers'], documents['document_id'], documents['question']):
        try:
            chunks = [chunk for chunk in kss.split_chunks(text, max_length=500)]
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
split_dataset = filtered_dataset.map(
    split_document,
    # batch_size=1,
    remove_columns=['id'],
    num_proc=12,
    batched=True
)
split_dataset = split_dataset.rename_column('context', 'text')
# %%
def check(example):
    if example:
        return False
    else:
        return True

filtered_dataset_ = split_dataset.filter(check, keep_in_memory=False, input_columns=['question'])
# %%
id_list = set()
def check(example):
    text = example['title'] + example['text']
    if text in id_list:
        return False
    else:
        id_list.add(text)
        return True             

filtered_dataset = filtered_dataset_.filter(check, keep_in_memory=False)
#%%
filtered_dataset.save_to_disk('/opt/ml/input/data/data/generation_needed')
# %%
