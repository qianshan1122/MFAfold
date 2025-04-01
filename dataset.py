from itertools import product
import torch
from torch.utils.data import Dataset
from utils import bp2matrix, prob_mat, seq2emb
from createfeature import DPCP, ncp_nd
import pandas as pd
import json


class SeqDataset(Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path)
        self.seqs = data['sequence'].tolist()
        self.ids = data.id.tolist()
        self.base_pairs = [
            json.loads(data.base_pairs.iloc[i]) for i in range(len(data))
        ]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seqid = self.ids[idx]

        sequence = ''.join(self.seqs[idx])
        L = len(sequence)
        Mc = None
        if self.base_pairs is not None:
            Mc = bp2matrix(L, self.base_pairs[idx])
        interaction_prior = prob_mat(sequence, L).unsqueeze(0)
        one_hot = seq2emb(sequence, L)

        data_fcn = torch.zeros((16, L, L))
        for n, cord in enumerate(list(product(torch.arange(4), torch.arange(4)))):
            i, j = cord
            data_fcn[n, :, :] = torch.matmul(one_hot[:, i].reshape(-1, 1),
                                             one_hot[:, j].reshape(1, -1))
        input1 = torch.cat((data_fcn, interaction_prior), dim=0)
        dpcp_emb = DPCP(sequence)
        ncp_emb = ncp_nd(sequence)  # (l,4)

        item = {"one_hot": one_hot, "contact": Mc, "length": L, "sequence": sequence, "input1": input1,
                "interaction_prior": interaction_prior, 'seqid': seqid, "ncp": ncp_emb, "dpcp": dpcp_emb}
        return item
