#%%
from transformers import RagRetriever
from datasets import load_from_disk

class HybridRetriever(RagRetriever):
    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
        super().__init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True)
        self.test_dataset = load_from_disk()
    
    def __call__(self, question_id, question_input_ids, question_hidden_states, prefix, n_docs, return_tensors):
        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        sparse_doc_ids, sparse_scores = self._get_bm25_top_k(question_id)

    def _get_bm25_top_k(self, question_id):
        data = self.test_dataset[question_id]
        return data[''], data['']
    
    def _rerank(self, dense_doc_ids, dense_scores, sparse_doc_ids, sparse_scores):
        pass