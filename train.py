import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
import model
import load_data


dataset = load_data.dataset(config.PATH+"train.tsv")
loader = DataLoader(dataset, batch_size=config.BatchSize, shuffle=True,
                                  collate_fn=dataset.batch_data_pro)


model = model.BERT().to(config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
i=0
for epoch in range(15):
    trainloss = 0
    trainnum = 0
    for seq, target ,mask in loader:
      '''
      enc_inputs: [batch_size, src_len]
      dec_inputs: [batch_size, tgt_len]
      dec_outputs: [batch_size, tgt_len]
      '''
      # outputs: [batch_size * tgt_len, tgt_vocab_size]
      outputs = model(seq, mask)

      loss = criterion(outputs[:,0,:],target)

      i = (i + 1) % 100
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      trainloss +=loss.item()*len(target)
      trainnum+=len(target)

      if i % 100 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(trainloss/trainnum))

torch.save(model,"model.pkl")