import torch
from torch.utils.data import Dataset
import json, logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

class SequenceDataset(Dataset):
    def __init__(self, data_file, label_list_file, label_name='ec', logger=None):
        self.logger = logger if logger is not None else get_logger('SequenceDataset')
        self.raw_data = torch.load(data_file)
        self.pids = list(self.raw_data.keys())
        self.embeddings = [self.raw_data[pid]['embedding'] for pid in self.pids]
        self.raw_labels = [self.raw_data[pid][label_name] for pid in self.pids]
        with open(label_list_file, 'r') as f:
            self.label_list = json.load(f)
        self.label2idx = {l: i for i, l in enumerate(self.label_list)}
        self.labels = torch.zeros(len(self.raw_labels), len(self.label_list))
        for i, label in enumerate(self.raw_labels):
            for l in label:
                self.labels[i, self.label2idx[l]] = 1
        self.logger.info(f'Loaded {len(self)} sequences with {label_name} labels')
        self.num_labels = len(self.label_list)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
    

class MultiSequenceDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
    def __len__(self):
        return
    
    def __getitem__(self, idx):
        return
    

if __name__ == '__main__':
    
    dataset = MultiSequenceDataset()