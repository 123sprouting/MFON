
        
import torch
from torch import nn
import torch.nn.functional as F
import os
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

from models.trans.transformer import TransformerEncoder
from models.classifier import BaseClassifier

class VisionEncoder(nn.Module):
    def __init__(self, config):
        super(VisionEncoder, self).__init__()
        self.vision_fea_dim = config.MOSEI.downStream.vision_fea_dim
        self.vision_seq_len = config.MOSEI.downStream.vision_seq_len
        
        self.encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim
        self.vision_nhead = config.MOSEI.downStream.vision_nhead
        self.vision_tf_num_layers = config.MOSEI.downStream.vision_tf_num_layers
        self.attn_dropout = config.MOSEI.downStream.vision_drop_out
        self.attn_mask = config.MOSEI.downStream.vision_attn_mask
        
        self.proj_a = nn.Linear(self.vision_fea_dim, self.encoder_fea_dim)
        
        self.trans_encoder_a = TransformerEncoder(
                embed_dim=self.encoder_fea_dim, 
                num_heads=self.vision_nhead, # 8
                layers=self.vision_tf_num_layers, # 2
                attn_dropout=self.attn_dropout, 
                relu_dropout=self.attn_dropout, 
                res_dropout=self.attn_dropout, 
                embed_dropout=self.attn_dropout,
                attn_mask=self.attn_mask # True
        ) 
            
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



class VisionPretrain(nn.Module):
    def __init__(self, config, encoder_fea_dim=None):
        super(VisionPretrain, self).__init__()
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim
        self.encoder = VisionEncoder(config)
        self.classifier = BaseClassifier(
            input_size=encoder_fea_dim,
            hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 8)],
            output_size=1, name='VisionRegClassifier',
        )
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.config = config
        self.model_path = config.MOSEI.path.encoder_path + str(config.seed) + '/'
        check_dir(self.model_path)

    def forward(self, audio, label, return_loss=True, device=None):
        if device is None:
            device = self.device
        x = self.encoder(audio)
        pred = self.classifier(x).squeeze()

        if return_loss:
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, loss
        else:
            return pred

    def save_model(self, name='best_loss'):
        # save all modules
        encoder_path = self.model_path + name + '_vision_encoder.pt'
        decoder_path = self.model_path + name + '_vision_decoder.pt'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), decoder_path)
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)

    def load_model(self, name='best_loss', module=None):
        encoder_path = self.model_path + name + '_vision_encoder.pt'
        decoder_path = self.model_path + name + '_vision_decoder.pt'
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

