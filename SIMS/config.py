import torch
import os

seed = 1111
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = os.path.dirname(__file__)

LOGPATH = os.path.join(root_path, 'log/')

if not os.path.exists(LOGPATH): 
    os.makedirs(LOGPATH)
                

class SIMS:
    class path:
        raw_data_path = 'data/SIMS/unaligned_39.pkl'
        model_path = os.path.join(root_path, 'save_models/all_model/SIMS/')
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        encoder_path = os.path.join(root_path, 'save_models/uni_fea_encoder/SIMS/')
        if not os.path.exists(encoder_path): 
            os.makedirs(encoder_path)

    class downStream:
        language = 'cn'
      
        encoder_fea_dim = 768
        
        text_fea_dim = 768
        
        vision_fea_dim = 709
        vision_seq_len = 55
        
        audio_fea_dim = 33
        audio_seq_len = 400
        
        vision_drop_out, audio_drop_out = 0.1, 0.5
        vision_nhead, audio_nhead = 8, 8
        vision_dim_feedforward = encoder_fea_dim 
        audio_dim_feedforward = encoder_fea_dim 
        vision_tf_num_layers, audio_tf_num_layers = 5, 3
        vision_attn_mask, audio_attn_mask = True, True
        
        audio_text_nhead, vision_text_nhead = 8, 8
        vision_text_tf_num_layers, audio_text_tf_num_layers = 5, 2
        drop_out = 0.6
        text_drop_out = 0.5
        attn_mask = True

        batch_size = 32
        update_epochs = 4
        
        alen=audio_seq_len
        vlen=vision_seq_len 
        p_len= 3
        
        class visionPretrain:
            lr = 1e-4
            epoch = 25
            decay = 1e-3
         
        class audioPretrain:
            lr = 1e-3
            epoch = 25
            decay = 1e-3
      
        class TVAtrain:	
            text_lr = 5e-5
            audio_lr = 1e-3
            vision_lr = 1e-3
            other_lr = 1e-3
            
            text_decay = 1e-3
            audio_decay = 1e-3
            vision_decay = 1e-2
            other_decay = 1e-2
            
            epoch = 25
            
            delta_va = 0.5
            delta_tva = 0.5
            delta_nce = 0.005
           
