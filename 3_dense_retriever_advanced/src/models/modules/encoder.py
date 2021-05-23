# from transformers import BertPreTrainedModel, BertModel
from transformers import ElectraPreTrainedModel, ElectraModel
import torch.nn.functional as F
import torch.nn as nn 
# from src.models.modules.projection import Projection

# class BertEncoder(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertEncoder, self).__init__(config)

#         self.bert = BertModel(config)
#         self.init_weights()
#         for name, param in self.bert.named_parameters():
#             if ('embeddings' in name) or ('encoder.layer.0' in name) or ('encoder.layer.1.' in name) or ('encoder.layer.2' in name) or ('encoder.layer.3' in name) or ('encoder.layer.4' in name):
#                 param.requires_grad = False
#         # self.projection = Projection()


#     def forward(self, input_ids, attention_mask=None, token_type_ids=None):
#         outputs = self.bert(
#             input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
#         )
#         pooled_output = outputs[1]
#         # pooled_output = self.projection(pooled_output)
#         # first_token_embedding = outputs[0][:, 0, :]
#         return pooled_output
#         # return F.normalize(pooled_output, dim=1)

class ElectraSmallEncoder(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraSmallEncoder, self).__init__(config)
        self.electra = ElectraModel(config)
        self.init_weights()
        for name, param in self.electra.named_parameters():
            if ('embeddings' in name):
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return outputs[0][:,0,:]

class ElectraBaseEncoder(ElectraPreTrainedModel):
    def __init__(self, config):
        super(ElectraBaseEncoder, self).__init__(config)
        self.electra = ElectraModel(config)
        self.projection = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Linear(768, 256)
        )
        self.init_weights()
        for name, param in self.electra.named_parameters():
            if ('embeddings' in name) or ('encoder.layer.0' in name) or ('encoder.layer.1.' in name) or ('encoder.layer.2' in name) or ('encoder.layer.3' in name) or ('encoder.layer.4' in name):
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.electra(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        first_token = outputs[0][:,0,:]
        return self.projection(first_token)