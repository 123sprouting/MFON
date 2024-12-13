from transformers import BertModel, BertTokenizer
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, config, name=None):
        super(TextEncoder, self).__init__()
        self.name = name
        language = config.SIMS.downStream.language
        if language =='en':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
            self.extractor = BertModel.from_pretrained('bert-base-uncased')
        elif language == 'cn':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  
            self.extractor = BertModel.from_pretrained('bert-base-chinese')
        self.device = config.DEVICE

    def forward(self, text):
        x = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        x = self.extractor(**x)
        last_hidden_state = x['last_hidden_state']
        return last_hidden_state
        # [bs, seq, h]
