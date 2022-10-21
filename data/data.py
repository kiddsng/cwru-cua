# import modules and libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import numpy as np
import os

def to_np(x):
    return x.detach().cpu().numpy()

class LoadDataset(Dataset):
    def __init__(self, dataset):
        super(LoadDataset, self).__init__()

        X = dataset['samples']
        y = dataset['labels']

        if len(X.shape) < 3: # change shape of tensor from ([8184, 5120]) to ([8184, 1, 5120])
            X = X.unsqueeze(1)
        
        self.data, self.labels = X, y
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return(len(self.data))

def generate_dataloaders(path, domain, batch_size):
    dataset = torch.load(os.path.join(path, domain + '.pt'))
    num_samples = dataset['samples'].shape[0]
    num_val = int(0.2 * num_samples)
    num_test = int(0.4 * num_samples)
    np.random.seed(4154)
    perm = np.random.permutation(num_samples)

    val_dataset = {'samples': dataset['samples'][perm[0:num_val]], 'labels': dataset['labels'][perm[0:num_val]]}
    test_dataset = {'samples': dataset['samples'][perm[num_val:num_test]], 'labels': dataset['labels'][perm[num_val:num_test]]}
    train_dataset = {'samples': dataset['samples'][perm[num_test: ]], 'labels': dataset['labels'][perm[num_test: ]]}

    train_dataset = LoadDataset(train_dataset)
    val_dataset = LoadDataset(val_dataset)
    test_dataset = LoadDataset(test_dataset)

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader

# create replay data


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def update(self, data):
        x, y, = data
        self.x = np.concatenate([self.x, x], 0)
        self.y = np.concatenate([self.y, y], 0)
        return self


class ReplayDataset(Dataset):
    def __init__(self, train_dataset):
        # generate replays from source data
        tmp = {'x': [], 'y': []}
        for data in train_dataset:
            tmp['x'] += [data[0].numpy()]
            tmp['y'] += [data[1]]
        tmp['x'] = np.array(tmp['x'])
        tmp['y'] = np.array(tmp['y'])

        self.replay_dataset = SimpleDataset(np.array(tmp['x']), np.array(tmp['y']))

    @staticmethod
    def rand_sample(dataset):
        n = len(dataset)
        i = np.random.randint(n)
        return dataset[i]

    def __getitem__(self, index):
        x_rpy, y_rpy = self.rand_sample(self.replay_dataset)
        return x_rpy, y_rpy

    def __len__(self):
        return len(self.replay_dataset)

def generate_replay_dataloader(train_dataset, batch_size):
    replay_dataset = ReplayDataset(train_dataset=train_dataset)   
    replay_dataloader = DataLoader(dataset=replay_dataset, shuffle=True, batch_size=batch_size)
    return replay_dataloader