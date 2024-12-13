
        
import torch
from torch import nn
import torch.nn.functional as F
import os

from models.trans.transformer import TransformerEncoder
from models.classifier import BaseClassifier

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()

        self.encoder_fea_dim = config.MOSI.downStream.encoder_fea_dim
        self.audio_fea_dim = config.MOSI.downStream.audio_fea_dim
        self.audio_seq_len = config.MOSI.downStream.audio_seq_len
        self.audio_nhead = config.MOSI.downStream.audio_nhead
        self.audio_tf_num_layers = config.MOSI.downStream.audio_tf_num_layers
        self.attn_dropout = config.MOSI.downStream.audio_drop_out
        self.attn_mask = config.MOSI.downStream.audio_attn_mask
        
        self.proj_a = nn.Linear(self.audio_fea_dim, self.encoder_fea_dim)
        
        self.trans_encoder_a = TransformerEncoder(embed_dim=self.encoder_fea_dim, 
                                num_heads=self.audio_nhead, # 8
                                layers=self.audio_tf_num_layers, # 2
                                attn_dropout=self.attn_dropout, 
                                relu_dropout=self.attn_dropout, 
                                res_dropout=self.attn_dropout, 
                                embed_dropout=self.attn_dropout,
                                attn_mask=self.attn_mask) # True
            
    def forward(self,audio):
        
        proj_x_a = self.proj_a(audio) # [bs, seq, h]
        proj_x_a = proj_x_a.permute(1, 0, 2)
        
        h_as = self.trans_encoder_a(proj_x_a) 
        if type(h_as) == tuple:
            h_as = h_as[0]# [seq, bs,h]
        last_h_a = h_as[0]   # Take the last output for prediction # [bs, h]
        return last_h_a
        
    def set_froze(self):
        for param in self.parameters():
            param.requires_grad = False


class AudioPretrain(nn.Module):
    def __init__(self, config, encoder_fea_dim=None):
        super(AudioPretrain, self).__init__()
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSI.downStream.encoder_fea_dim
        self.encoder = AudioEncoder(config)
        self.classifier = BaseClassifier(
            input_size=encoder_fea_dim,
            hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 8)],
            output_size=1, name='AudioRegClassifier',
        )
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.config = config
        self.model_path = config.MOSI.path.encoder_path + str(config.seed) + '/'
        check_dir(self.model_path)

    def forward(self, audio, label, return_loss=True, device=None, mode='train'):
        if device is None:
            device = self.device
        x = self.encoder(audio)
        pred = self.classifier(x).squeeze()

        if return_loss:
            loss = self.criterion(pred.squeeze(), label.squeeze())
            if mode=='train':
                return pred, loss
            else:
                return pred, loss, x
        else:
            return pred

    def save_model(self, name='best_loss'):
        # save all modules
        encoder_path = self.model_path + name + '_audio_encoder.pt'
        decoder_path = self.model_path + name + '_audio_decoder.pt'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), decoder_path)
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)

    def load_model(self, name='best_loss', module=None):
        encoder_path = self.model_path + name + '_audio_encoder.pt'
        decoder_path = self.model_path + name + '_audio_decoder.pt'
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
