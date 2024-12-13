import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

import random 
import os

def write_log(log, path):
    if isinstance(log, dict):
        tmp = ''
        for k in log.keys():
            tmp = tmp + str(k) + ':' + str(log[k]) + '\t'
        with open(path, 'a') as f:
            f.write(tmp+'\n')
    else:
        with open(path, 'a') as f:
            f.writelines(log + '\n')


def write_config(config, log_path):
    d = {}
    d['path']={}
    for key in dir(config.SIMS.path):
        if key.startswith('__') or key.startswith('_') :
            continue
        d['path'][key] = getattr(config.SIMS.path, key)
    write_log(d, path=log_path)
    
    d = {}
    d['downStream']={}
    for key in dir(config.SIMS.downStream):
        if key.startswith('__') or key.startswith('_')  or key.endswith('train'):
            continue
        d['downStream'][key] = getattr(config.SIMS.downStream, key)
    write_log(d, path=log_path)
    
    d = {}
    d['TVAtrain'] = {}
    for key in dir(config.SIMS.downStream.TVAtrain):
        if key.startswith('__') or key.startswith('_') :
            continue
        d['TVAtrain'][key] = getattr(config.SIMS.downStream.TVAtrain, key)
    write_log(d, path=log_path)
    
    d = {}
    d['audioPretrain'] = {}
    for key in dir(config.SIMS.downStream.audioPretrain):
        if key.startswith('__') or key.startswith('_') :
            continue
        d['audioPretrain'][key] = getattr(config.SIMS.downStream.audioPretrain, key)
    write_log(d, path=log_path)
    
    d = {}
    d['visionPretrain'] = {}
    for key in dir(config.SIMS.downStream.visionPretrain):
        if key.startswith('__') or key.startswith('_') :
            continue
        d['visionPretrain'][key] = getattr(config.SIMS.downStream.visionPretrain, key)
    write_log(d, path=log_path)
   


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)


def multiclass_acc(y_pred, y_true):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param y_pred: Float array representing the predictions, dimension (N,)
    :param y_true: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

class Metrics:

    def eval_mosei_regression(self, y_true, y_pred, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = multiclass_acc(test_preds_a3, test_truth_a3)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_truth, non_zeros_binary_preds)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_truth, binary_preds)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        eval_results = {
            "Has0_acc_2": round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2": round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            # "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }# Has的要小于Non的，不然感觉是违背常理的。。。
        return eval_results

    def eval_sims_regression(self, y_true, y_pred):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        # for i in range(5):
        #     test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
        # for i in range(5):
        #     test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds < ms_5[i + 1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth < ms_5[i + 1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = np.sum(np.round(test_preds_a2) == np.round(test_truth_a2)) / float(len(test_truth_a2))
        mult_a3 = np.sum(np.round(test_preds_a3) == np.round(test_truth_a3)) / float(len(test_truth_a3))
        mult_a5 = np.sum(np.round(test_preds_a5) == np.round(test_truth_a5)) / float(len(test_truth_a5))

        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": round(mult_a2, 4),
            "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "F1_score": round(f_score, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),  # Correlation Coefficient
        }
        return eval_results


