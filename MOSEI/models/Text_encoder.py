from transformers import BertModel, BertTokenizer
from torch import nn
from models.classifier import BaseClassifier
import torch
import os


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class TextEncoder(nn.Module):
    def __init__(self, config, name=None):
        super(TextEncoder, self).__init__()
        self.name = name
        language = config.MOSEI.downStream.language
        if language =='en':
            self.tokenizer = BertTokenizer.from_pretrained('/home/zxm/pycharm/data/pretrain_bert/bert-base-uncased')  
            self.extractor = BertModel.from_pretrained('/home/zxm/pycharm/data/pretrain_bert/bert-base-uncased')
        elif language == 'cn':
            self.tokenizer = BertTokenizer.from_pretrained('/home/zxm/pycharm/data/pretrain_bert/bert_cn')  
            self.extractor = BertModel.from_pretrained('/home/zxm/pycharm/data/pretrain_bert/bert_cn')
        self.device = config.DEVICE

    def forward(self, text):
        x = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        x = self.extractor(**x)
        last_hidden_state = x['last_hidden_state']
        return last_hidden_state
        # [bs, seq, h]
        
        
class TextPretrain(nn.Module):
    def __init__(self, config, encoder_fea_dim=None):
        super(TextPretrain, self).__init__()
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim
        self.encoder = TextEncoder(config)
        self.classifier = BaseClassifier(
            input_size=encoder_fea_dim,
            hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 8)],
            output_size=1, name='TextClassifier',
        )
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.config = config
        self.model_path = config.MOSEI.path.encoder_path + str(config.seed) + '/'
        check_dir(self.model_path)

    def forward(self, text, label, return_loss=True, device=None):
        if device is None:
            device = self.device
        x = self.encoder(text)
        pred = self.classifier(x).squeeze()

        if return_loss:
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, loss
        else:
            return pred

    def save_model(self, name='best_loss'):
        # save all modules
        encoder_path = self.model_path + name + '_text_encoder.pt'
        decoder_path = self.model_path + name + '_text_decoder.pt'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), decoder_path)
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)

    def load_model(self, name='best_loss', module=None):
        encoder_path = self.model_path + name + '_text_encoder.pt'
        decoder_path = self.model_path + name + '_text_decoder.pt'
        print('model loaded from:')

        if module == 'encoder':
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(encoder_path)
        if module == 'decoder':
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(decoder_path)
        if module == 'all' or module is None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(encoder_path)
            print(decoder_path)


