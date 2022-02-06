import torch
import numpy as np


class EmbDataset(torch.utils.data.Dataset):
    def __init__(self, source_matrix, tgt_matrix, len_neg=1):
        self.source_matrix = source_matrix
        self.tgt_matrix = tgt_matrix
        self.len_neg = len_neg

    def __getitem__(self, idx):
        neg_idx = np.random.randint(len(self))
        while neg_idx == idx:
            neg_idx = np.random.randint(len(self))
        anchor = torch.tensor(self.source_matrix[idx], dtype=torch.float32)
        positive = torch.tensor(self.tgt_matrix[idx], dtype=torch.float32)
        negative = torch.tensor(self.tgt_matrix[neg_idx], dtype=torch.float32)
        return anchor, positive, negative

    def __len__(self):
        return self.source_matrix.shape[0]
