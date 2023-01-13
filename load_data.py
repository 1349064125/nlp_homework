import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import config

def loaddata():
    word = []
    data = pd.read_csv(config.PATH+"train.tsv", sep='\t', names=["seq1", "seq2", "target"]).dropna()

    seq1 = [[j for j in str(i).replace(" ", "")] for i in data["seq1"]]
    seq2 = [[j for j in str(i).replace(" ", "")] for i in data["seq2"]]

    for i in seq1:
        word.extend(i)
    for i in seq2:
        word.extend(i)


    data = pd.read_csv(config.PATH+"dev.tsv", sep='\t', names=["seq1", "seq2", "target"]).dropna()

    seq1 = [[j for j in str(i).replace(" ", "")] for i in data["seq1"]]
    seq2 = [[j for j in str(i).replace(" ", "")] for i in data["seq2"]]

    for i in seq1:
        word.extend(i)
    for i in seq2:
        word.extend(i)


    data = pd.read_csv(config.PATH+"test.tsv", sep='\t', names=["seq1", "seq2", "target"]).dropna()

    seq1 = [[j for j in str(i).replace(" ", "")] for i in data["seq1"]]
    seq2 = [[j for j in str(i).replace(" ", "")] for i in data["seq2"]]

    for i in seq1:
        word.extend(i)
    for i in seq2:
        word.extend(i)

    vacob = {"PAD":0,"CLA":1,"SEP":2,"UNK":3,"SEP":4,}
    mset = set(word)

    for i  in  mset:
        vacob[i] = len(vacob)

    print(vacob)
    np.save('data/vocab.npy', vacob)

class dataset(Dataset):
    def __init__(self,path):
        data = pd.read_csv(path, sep='\t', names=["seq1", "seq2", "target"]).dropna()
        #排序
        my_index = (data.seq1.str.len()+data.seq2.str.len()).sort_values().index
        #data = data.reindex(my_index)


        seq1 = [[j for j in str(i).replace(" ", "")] for i in data["seq1"]]
        seq2 = [[j for j in str(i).replace(" ", "")] for i in data["seq2"]]

        self.seq1 = seq1
        self.seq2 =seq2
        self.target = data["target"]
        self.vocab = np.load("data/vocab.npy", allow_pickle=True).tolist()

    def __getitem__(self, index):
        seq1 = [self.vocab.get(i, self.vocab["UNK"]) for i in self.seq1[index]]
        seq2 = [self.vocab.get(i, self.vocab["UNK"]) for i in self.seq2[index]]

        return seq1,seq2,self.target[index]
    def __len__(self):
        return len(self.seq1)

    def batch_data_pro(self, batch_datas):
        DEVICE = config.DEVICE
        seq=[]
        target = []
        mask = []
        max_len = 0
        for i, j, k in batch_datas:
            if (len(i)+len(j)) > max_len:
                max_len = (len(i)+len(j))

        for i, j, k in batch_datas:
            seq.append([self.vocab["CLA"]]+i+[self.vocab["SEP"]]+j + [self.vocab["PAD"]] * (max_len - len(j) - len(i)))
            target.append(k)




        seq = torch.tensor(seq, device=DEVICE)
        target = torch.tensor(target, device=DEVICE)
        mask = get_attn_pad_mask(seq, seq)



        return seq,target,mask


def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token 
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

'''
dataset = dataset(config.PATH+"train.tsv")
dataloader = DataLoader(dataset, batch_size=config.BatchSize, shuffle=False,
                                  collate_fn=dataset.batch_data_pro)

for seq ,mask, target in dataloader:
    print(seq)
    print(mask)
    print(target)
    input()
'''