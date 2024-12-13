import torch
import os

seed = 1111
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = os.path.dirname(__file__)

LOGPATH = os.path.join(root_path, 'log/')

if not os.path.exists(LOGPATH): 
    os.makedirs(LOGPATH)
       

class MOSEI:
    class path:
        raw_data_path = '/home/zxm/pycharm/data/MOSEI/unaligned_50.pkl'
        model_path = os.path.join(root_path, 'save_models/all_model/MOSEI/')
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        encoder_path = os.path.join(root_path, 'save_models/uni_fea_encoder/MOSEI/')
        if not os.path.exists(encoder_path): 
            os.makedirs(encoder_path)

    class downStream:
        language = 'en'
      
        encoder_fea_dim = 768
        
        text_fea_dim = 768
        
        vision_fea_dim = 35
        vision_seq_len = 500
        
        audio_fea_dim = 74
        audio_seq_len = 500
        
        vision_drop_out, audio_drop_out = 0., 0
        vision_nhead, audio_nhead = 8, 8
        vision_dim_feedforward = encoder_fea_dim 
        audio_dim_feedforward = encoder_fea_dim 
        vision_tf_num_layers, audio_tf_num_layers = 4, 2
        vision_attn_mask, audio_attn_mask = False, False
        
        audio_text_nhead, vision_text_nhead = 8, 8
        vision_text_tf_num_layers, audio_text_tf_num_layers = 2, 3
        drop_out = 0.1
        text_drop_out = 0.3
        attn_mask = True
                
        alen=audio_seq_len
        vlen=vision_seq_len 
        batch_size = 32
        update_epochs = 4
        
        class visionPretrain:
            lr = 1e-4
            epoch = 25
            decay = 1e-3
	
        class audioPretrain:
            lr = 1e-4
            epoch = 25
            decay = 1e-3
     
        class TVAtrain:	
            text_lr = 5e-5
            audio_lr = 1e-3
            vision_lr = 1e-4
            other_lr = 1e-3
            
            text_decay = 1e-2
            audio_decay = 1e-3
            vision_decay = 1e-3
            other_decay = 1e-3
            
            epoch = 25
            
            delta_va = 0.3
            delta_nce = 0.001
           