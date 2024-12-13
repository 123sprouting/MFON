import torch
import datetime
from tqdm import tqdm
from utils import write_log, set_random_seed
from models.model import TVA_fusion
from utils import write_config
from models.classifier import BaseClassifier


def TVA_train_fusion(config, metrics, seed, train_data, valid_data):
    print('---------------TVA_EXP---------------')
    
    set_random_seed(seed)
    
    update_epochs = config.SIMS.downStream.update_epochs
    
    text_lr = config.SIMS.downStream.TVAtrain.text_lr
    audio_lr = config.SIMS.downStream.TVAtrain.audio_lr
    vision_lr = config.SIMS.downStream.TVAtrain.vision_lr
    other_lr = config.SIMS.downStream.TVAtrain.other_lr
    
    text_decay = config.SIMS.downStream.TVAtrain.text_decay
    audio_decay = config.SIMS.downStream.TVAtrain.audio_decay
    vision_decay = config.SIMS.downStream.TVAtrain.vision_decay
    other_decay = config.SIMS.downStream.TVAtrain.other_decay
            
    delta_va = config.SIMS.downStream.TVAtrain.delta_va 
    delta_nce = config.SIMS.downStream.TVAtrain.delta_nce
    
    model = TVA_fusion(config).to(config.DEVICE)

    model.load_froze()
    
    text_params = list(model.proj_t.named_parameters()) + list(model.text_encoder.named_parameters())
    text_params = [p for _, p in text_params] 
    vision_params = list(model.proj_v.named_parameters()) +\
                list(model.vision_with_text.named_parameters()) 
    vision_params = [p for _, p in vision_params] + [model.promptv_m]
    audio_params = list(model.proj_a.named_parameters()) +\
                list(model.audio_with_text.named_parameters())
    audio_params = [p for _, p in audio_params] + [model.prompta_m]
    model_params_other = [p for n, p in list(model.named_parameters()) if '_decoder' in n] 

    optimizer_grouped_parameters = [
        {'params': text_params, 'weight_decay': text_decay, 'lr': text_lr},
        {'params': audio_params, 'weight_decay': audio_decay, 'lr': audio_lr},
        {'params': vision_params, 'weight_decay': vision_decay, 'lr': vision_lr},
        {'params': model_params_other, 'weight_decay': other_decay, 'lr': other_lr}
    ]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters)
   
    loss, best_loss  = 0, 1e8
    loss_a = loss_v = pred_loss = loss_nce = 0
    device = config.DEVICE  
    total_epoch = config.SIMS.downStream.TVAtrain.epoch
    best_epoch = 1
    for epoch in range(1, total_epoch + 1):
        
        model.train()
        left_epochs = update_epochs
        bar = tqdm(train_data, disable=False)
        for index, batch_data in enumerate(bar):
            try:
                bar.set_description("Epoch:%d|loss:%s|pred_loss:%s|loss_v:%s|loss_a:%s|loss_nce:%s" % (
                    epoch, loss.item(), pred_loss.item(), loss_v.item(), loss_a.item(),  loss_nce.item()
                    )
                )
            except:
                bar.set_description(
                    "Epoch:%d|loss:%s|pred_loss:%s|loss_v:%s|loss:%s|loss_nce:%s" % (epoch, loss, pred_loss, loss_v, loss_a, loss_nce)
                )
            if left_epochs == update_epochs:
                optimizer.zero_grad()
            left_epochs -= 1

            text = batch_data['raw_text']
            vision = batch_data['vision'].clone().detach().to(device).float()
            audio = batch_data['audio'].clone().detach().to(device).float()
            label = batch_data['labels']['M'].clone().detach().to(device)
        
            pred, (loss_v, loss_a, loss_nce) = model(text, vision, audio, mode='train')
            
            pred_loss = torch.mean((pred-label)*(pred-label))  # [bs]
            
            loss = pred_loss + delta_va * (loss_v + loss_a) + delta_nce * loss_nce
            
            loss.backward()
           
            if not left_epochs:
                optimizer.step()
                left_epochs = update_epochs
                
        if not left_epochs:
            optimizer.step()

        _, result_loss = eval(model, metrics, valid_data, device)
        
        if result_loss < best_loss:
            best_loss = result_loss
            model.save_model()


def eval(model, metrics, eval_data, device):
    model.eval()
    with torch.no_grad():
        pred, truth = [], []
        loss = 0
        lens = 0 
        for index, batch_data in enumerate(eval_data):
            text = batch_data['raw_text']
            vision = batch_data['vision'].clone().detach().to(device).float()
            audio = batch_data['audio'].clone().detach().to(device).float()
            label = batch_data['labels']['M'].clone().detach().view(-1).to(device).float()
            _pred, _ = model(text, vision, audio, mode='test')
            pred.append(_pred.view(-1))
            truth.append(label)
            _loss = torch.mean((label-_pred)*(label-_pred))
            loss += _loss.item() * len(label)
            lens += len(label)
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu')).squeeze()
        eval_results = metrics.eval_sims_regression(truth, pred)
        eval_results['Loss'] = loss / lens
    model.train()
    return eval_results, loss / lens


def TVA_test_fusion(config, metric, test_data,mode_path=None):
    
    seed = config.seed
    log_path = config.LOGPATH + "SIMS_TVA_Test." + datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S') + '.log'

    write_config(config, log_path)
    
    model = TVA_fusion(config=config)

    device = config.DEVICE
    model.to(device)
    
    model.load_model(mode_path)
    result, loss = eval(model,metric, test_data, device)
   
    log = '\nTVA_Test result\n\tacc_2:%s\n\tacc_3:%s\n\tacc_7:%s\n\t' \
        'F1_score:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
        '------------------------------------------' % (
        result["Mult_acc_2"], result["Mult_acc_3"],result["Mult_acc_5"], result["F1_score"], 
        result['MAE'], result['Corr'], loss
    )
    write_log(log, log_path)
    print(log)
    
    return result




