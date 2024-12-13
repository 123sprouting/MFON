import torch
from tqdm import tqdm
import datetime
from models.Audio_encoder import AudioPretrain
from utils import write_log, set_random_seed
from utils import write_config


    
def Atrain(config, metrics, seed, train_data, valid_data):
    print('---------------AudioPretrain---------------')
    set_random_seed(seed)

    model = AudioPretrain(config=config)
    
    device = config.DEVICE
    total_epoch = config.MOSEI.downStream.audioPretrain.epoch
    
    lr = config.MOSEI.downStream.audioPretrain.lr
    decay = config.MOSEI.downStream.audioPretrain.decay
    update_epochs = config.MOSEI.downStream.update_epochs
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, amsgrad=True, weight_decay=decay)
    
    model.to(device)
    
    loss = 0
    best_loss = 1e8
    for epoch in range(1, total_epoch + 1):
        model.train()
        left_epochs = update_epochs
        bar = tqdm(train_data, disable=False)
        for index, sample in enumerate(bar):
            bar.set_description("Epoch:%d|Loss:[%s]|" % (epoch, loss))
             
            if left_epochs == update_epochs:
                optimizer.zero_grad()
            left_epochs -= 1
            audio = sample['audio'].clone().detach().to(device).float()
            label = sample['labels']['M'].clone().detach().to(device).float()
           
            _, loss = model(audio, label.float().squeeze(), return_loss=True)
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
    lens = 0.0
    with torch.no_grad():
        pred = []
        truth = []
        loss = 0
        for index, sample in enumerate(eval_data):
            audio = sample['audio'].clone().detach().to(device).float()
            label = sample['labels']['M'].clone().detach().to(device).float()
            _pred,_loss = model(audio, label.float().squeeze(), return_loss=True)
            pred.append(_pred.view(-1))
            truth.append(label)
            loss += _loss.item() * len(label)
            lens +=len(label)
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))
        eval_results = metrics.eval_mosei_regression(truth, pred)
        eval_results['Loss'] = loss / lens
    model.train()
    return eval_results, loss / lens


def Atest(config, metrics, test_data):
    
    log_path = config.LOGPATH + "MOSEI_AudioPretrain_Test." +\
            datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-') + '.log'
    write_config(config, log_path)
    
    model = AudioPretrain(config=config)
    device = config.DEVICE
    model.to(device)
    
    model.load_model(name='best_loss')
    result, loss = eval(model, metrics, test_data, device)
        
    log = '\nAudio_Test result\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2:%s\n\t' \
        'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
        '------------------------------------------' % (
        result['Has0_acc_2'], result['Has0_F1_score'],result['Non0_acc_2'], result['Non0_F1_score'], 
        result['Mult_acc_5'],result['Mult_acc_7'], result['MAE'], result['Corr'], loss
    )
    print(log)
    write_log(log, log_path)
