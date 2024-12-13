import torch
import numpy as np
import _pickle as pk
from torch.utils.data import Dataset, DataLoader


class MOSIDataset(Dataset):
    def __init__(self, mode, raw_data_path):
        with open(raw_data_path, 'rb') as f:
            self.data = pk.load(f)[mode]
        
        self.data['raw_text'] = np.array(self.data['raw_text'])
       
        self.size = len(self.data['raw_text'])
        self.data['index'] = torch.tensor(range(self.size))
        self.vision_fea_size = self.data['vision'][0].shape
        self.audio_fea_size = self.data['audio'][0].shape
         
        self.data['vision'][self.data['vision'] != self.data['vision']] = 0
        self.data['audio'][self.data['audio'] != self.data['audio']] = 0
        
        del self.data['text_bert'], self.data['annotations'], self.data['classification_labels'], self.data['id']
        if 'audio_lengths'   in self.data.keys():
            del self.data['audio_lengths']
        if 'vision_lengths'   in self.data.keys():
            del self.data['vision_lengths']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        samples = {}
        for key in self.data.keys():
            if key == 'regression_labels':
                samples['labels'] = {}
                samples['labels']['M'] = self.data[key][idx]
            else:
                samples[key] = self.data[key][idx]
        return samples


def MOSIDataloader(mode, raw_data_path, batch_size, shuffle=True):
    dataset = MOSIDataset(mode, raw_data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


class MOSEIDataset(Dataset):
    def __init__(self, mode, raw_data_path):
        with open(raw_data_path, 'rb') as f:
            self.data = pk.load(f)[mode]
        
        self.data['raw_text'] = np.array(self.data['raw_text'])
        self.size = len(self.data['raw_text'])
        self.data['index'] = torch.tensor(range(self.size))
        self.vision_fea_size = self.data['vision'][0].shape
        self.audio_fea_size = self.data['audio'][0].shape
         
        self.data['vision'][self.data['vision'] != self.data['vision']] = 0
        self.data['audio'][self.data['audio'] != self.data['audio']] = 0
        
        del self.data['text_bert'], self.data['annotations'], self.data['classification_labels'], self.data['id']
        if 'audio_lengths'  in self.data.keys():
            del self.data['audio_lengths']
        if 'vision_lengths'  in self.data.keys():
            del self.data['vision_lengths']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        samples = {}
        for key in self.data.keys():
            if key == 'regression_labels':
                samples['labels'] = {}
                samples['labels']['M'] = self.data[key][idx]
            else:
                samples[key] = self.data[key][idx]
        return samples


def MOSEIDataloader(mode, raw_data_path, batch_size, shuffle=True):
    dataset = MOSEIDataset(mode, raw_data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


class SIMSDataset(Dataset):
    def __init__(self, mode, raw_data_path):
        with open(raw_data_path, 'rb') as f:
            self.data = pk.load(f)[mode]
        self.data['raw_text'] = np.array(self.data['raw_text'])
       
        self.size = len(self.data['raw_text'])
        self.data['index'] = torch.tensor(range(self.size))
        self.vision_fea_size = self.data['vision'][0].shape
        self.audio_fea_size = self.data['audio'][0].shape
         
        self.data['vision'][self.data['vision'] != self.data['vision']] = 0
        self.data['audio'][self.data['audio'] != self.data['audio']] = 0
        
        del self.data['classification_labels_A'],self.data['classification_labels_V']
        del self.data['classification_labels_T'],self.data['classification_labels']
        del self.data['text_bert'], self.data['id']
        del self.data['audio_lengths'], self.data['vision_lengths']
           
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        samples = {}
        samples['labels'] = {}
        for key in self.data.keys():
            if key == 'regression_labels':
                samples['labels']['M'] = self.data[key][idx]
            elif key == 'regression_labels_A':
                samples['labels']['A'] = self.data[key][idx]
            elif key == 'regression_labels_V':
                samples['labels']['V'] = self.data[key][idx]
            elif key == 'regression_labels_T':
                samples['labels']['T'] = self.data[key][idx]
            else:
                samples[key] = self.data[key][idx]
        return samples
    

def SIMSDataloader(mode, raw_data_path,  batch_size=None, shuffle=True):
    dataset = SIMSDataset(mode, raw_data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

