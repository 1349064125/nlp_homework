import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

import config
import model
import load_data

dataset = load_data.dataset(config.PATH+"train.tsv")
loader = DataLoader(dataset, batch_size=config.BatchSize, shuffle=False,
                                  collate_fn=dataset.batch_data_pro)
model = torch.load("model.pkl").eval()
print("aaa")
for seq, target, mask in loader:
    '''
    enc_inputs: [batch_size, src_len]
    dec_inputs: [batch_size, tgt_len]
    dec_outputs: [batch_size, tgt_len]
    '''
    # outputs: [batch_size * tgt_len, tgt_vocab_size]
    outputs = model(seq, mask)

    pre_label = torch.argmax(outputs[:,0,:],-1)
    acc = accuracy_score(target.cpu().to(torch.int16),pre_label.cpu())
    print("acc:",acc)
    input()