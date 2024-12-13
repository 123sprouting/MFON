import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import os
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
sys.path.append(os.path.dirname(path))

from classifier import BaseClassifier
from Text_encoder import TextEncoder
from Vision_encoder import VisionEncoder
from Audio_encoder  import AudioEncoder
from models.trans.transformer import TransformerEncoder
from models.classifier import BaseClassifier


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))



class TVA_fusion(nn.Module):
    def __init__(self, config):
        super(TVA_fusion, self).__init__()
        self.config = config
        
        self.text_dropout = config.MOSEI.downStream.text_drop_out
        
        encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim
        audio_text_nhead = config.MOSEI.downStream.audio_text_nhead
        audio_text_tf_num_layers = config.MOSEI.downStream.audio_text_tf_num_layers
        
        self.audio_fea_dim = config.MOSEI.downStream.audio_fea_dim
        self.a_len = config.MOSEI.downStream.audio_seq_len
        
        vision_text_nhead = config.MOSEI.downStream.vision_text_nhead
        vision_text_tf_num_layers = config.MOSEI.downStream.vision_text_tf_num_layers
        
        attn_dropout= config.MOSEI.downStream.drop_out
        attn_mask = config.MOSEI.downStream.attn_mask
        
        audio_fea_dim = config.MOSEI.downStream.audio_fea_dim
        vision_fea_dim = config.MOSEI.downStream.vision_fea_dim
        text_fea_dim = config.MOSEI.downStream.text_fea_dim
        
        self.vlen, self.alen = config.MOSEI.downStream.vlen, config.MOSEI.downStream.alen
        
        # 加权重的提示 
        self.prompta_m = nn.Parameter(torch.rand(self.alen, encoder_fea_dim))
        self.promptv_m = nn.Parameter(torch.rand(self.vlen, encoder_fea_dim)) 
        
        self.text_encoder = TextEncoder(config=config)
        self.proj_t = nn.Linear(text_fea_dim, encoder_fea_dim)
        
        self.proj_v = nn.Linear(vision_fea_dim, encoder_fea_dim)
        
        # self.proj_t = nn.Conv1d(text_fea_dim, encoder_fea_dim, kernel_size=1, padding=0, bias=False)
        # self.proj_a = nn.Conv1d(vision_fea_dim, encoder_fea_dim, kernel_size=1, padding=0, bias=False)
        # self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        self.vision_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim, num_heads=vision_text_nhead, layers=vision_text_tf_num_layers, 
            attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
            attn_mask=attn_mask
        ) # Q:text, KV:vision
        
        self.proj_a = nn.Linear(audio_fea_dim, encoder_fea_dim)
        self.audio_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim, num_heads=audio_text_nhead, layers=audio_text_tf_num_layers, 
            attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
            attn_mask=attn_mask
        ) # Q:text, KV:audio
        
        self.vision_encoder_froze = VisionEncoder(config=config)
        self.audio_encoder_froze = AudioEncoder(config=config)
        
        self.TVA_decoder = BaseClassifier(
            input_size=encoder_fea_dim * 3,
            hidden_size=[encoder_fea_dim, encoder_fea_dim//2, encoder_fea_dim//8],
            output_size=1
        )
        self.device = config.DEVICE
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.model_path = config.MOSEI.path.model_path + str(config.seed) + '/'
        check_dir(self.model_path)
     
    def load_froze(self):
        model_path = self.config.MOSEI.path.encoder_path + str(self.config.seed) + '/'
        self.audio_encoder_froze.load_state_dict(torch.load(model_path+ 'best_loss_audio_encoder.pt', map_location=self.device))
        self.vision_encoder_froze.load_state_dict(torch.load(model_path + 'best_loss_vision_encoder.pt', map_location=self.device))
        self.audio_encoder_froze.set_froze()
        self.vision_encoder_froze.set_froze()
       
    def forward(self, text, vision, audio, mode='train'):
        last_hidden_text  = self.text_encoder(text)   # [bs, seq, h] [bs, h]
        last_hidden_text = F.dropout(self.proj_t(last_hidden_text.permute(1, 0, 2)), 
            p=self.text_dropout, training=self.training
        )
        x_t_embed = last_hidden_text[0]
        
        proj_vision = self.proj_v(vision).permute(1, 0, 2) + self.promptv_m.unsqueeze(1)
        h_tv = self.vision_with_text(last_hidden_text, proj_vision, proj_vision)    # [seq-v, bs, 768] [seq-t, bs, 768]--> [seq, bs,h]
        
        proj_audio = self.proj_a(audio).permute(1, 0, 2) + self.prompta_m.unsqueeze(1)
        h_ta = self.audio_with_text(last_hidden_text, proj_audio, proj_audio)
        
        x_v_embed = h_tv[0]   
        
        x_a_embed = h_ta[0]   
        
        x = torch.cat([x_t_embed, x_v_embed, x_a_embed], dim=-1) # [bs, 3h]
        pred = self.TVA_decoder(x).view(-1) # [bs]
        
        loss_v = loss_a = 0
        loss_nce = 0
        if mode == 'train':
            x_v_embed_froze = self.vision_encoder_froze(vision).squeeze()
            x_a_embed_froze = self.audio_encoder_froze(audio).squeeze()# [bs, h], [bs, h]
            loss_v = self.get_KL_loss(x_v_embed, x_v_embed_froze)
            loss_a = self.get_KL_loss(x_a_embed, x_a_embed_froze) 
            loss_nce = self.get_InfoNCE_loss(x_v_embed, x_t_embed) + self.get_InfoNCE_loss(x_a_embed, x_t_embed) 
        else:
            x_tva = torch.stack([x_t_embed, x_v_embed, x_a_embed])
            model_tva = self.TVA_decoder.MLP[2:]
            pred_tva = model_tva(x_tva)
            return pred, pred_tva
            
        return pred, (loss_v, loss_a, loss_nce)
        
    def save_model(self,name=None):
        # save all modules
        if name==None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = self.model_path + str(name)+'TVA_fusion' + '_model.pt'
        print('model saved at:\n', mode_path)
        torch.save(self.state_dict(), mode_path)

    def load_model(self, name=None):
        if name==None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = name
        print('model loaded from:\n', mode_path)
        # self.load_state_dict(torch.load(mode_path, map_location=self.device))
        checkpoint = torch.load(mode_path, map_location=self.device)
        model_state_dict = self.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        self.load_state_dict(filtered_checkpoint, strict=False)

    def get_distill_loss(self, input1, input2):
        diff_loss = torch.mean((input1-input2)*(input1-input2))
        return diff_loss
    
    def get_KL_loss(self, x_embed, x_embed_target):
        x_embed1 = F.log_softmax(x_embed, dim=1)
        x_embed_target1 = F.softmax(x_embed_target, dim=1)
        loss = self.criterion(x_embed1, x_embed_target1)
        return loss
    
    def get_InfoNCE_loss(self, input1, input2):
        
        x1 = input1 / input1.norm(dim=1, keepdim=True)
        x2 = input2 / input2.norm(dim=1, keepdim=True)

        pos = torch.sum(x1*x2, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x1, x2.t()), dim=-1)   # bs
        nce_loss = -(pos - neg).mean()
        
        return nce_loss
    
    