import torch
from tqdm import tqdm
import datetime
from models.Vision_encoder import VisionPretrain
from utils import write_log, set_random_seed
from utils import write_config

    
def Vtrain(config, metrics, seed, train_data, valid_data):
    print('---------------VisionPretrain---------------')
    set_random_seed(seed)
    
    model = VisionPretrain(config=config)
    
    device = config.DEVICE
    total_epoch = config.SIMS.downStream.visionPretrain.epoch
    
    lr = config.SIMS.downStream.visionPretrain.lr
    decay = config.SIMS.downStream.visionPretrain.decay
    
    update_epochs = config.SIMS.downStream.update_epochs

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
            vision = sample['vision'].clone().detach().to(device).float()
            
            label = sample['labels']['V'].clone().detach().to(device).float()
            _, loss = model(vision, label.float().squeeze(), return_loss=True)
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
            vision = sample['vision'].clone().detach().to(device).float()
            label = sample['labels']['V'].clone().detach().to(device).float()
            _pred,_loss = model(vision, label.float().squeeze(), return_loss=True)
            pred.append(_pred.view(-1))
            truth.append(label)
            loss += _loss.item() * len(label)
            lens +=len(label)
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))
        eval_results = metrics.eval_sims_regression(truth, pred)
        eval_results['Loss'] = loss / lens
    model.train()
    return eval_results, loss / lens


def Vtest(config, metrics, test_data, num=0):
    
    log_path = config.LOGPATH + "SIMS_VisionPretrain_Test." +\
            datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S-')+ '.log'
    
    write_config(config, log_path)
    
    model = VisionPretrain(config=config)
    device = config.DEVICE
    model.to(device)
    
    model.load_model(name='best_loss')
    result, loss = eval(model, metrics, test_data, device)
    print(result)
    log = '\nVision_Test result\n\tacc_2:%s\n\tacc_3:%s\n\tacc_7:%s\n\t' \
        'F1_score:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
        '------------------------------------------' % (
        result["Mult_acc_2"], result["Mult_acc_3"],result["Mult_acc_5"], result["F1_score"], 
        result['MAE'], result['Corr'], loss
    )
    print(log)
    write_log(log, log_path)