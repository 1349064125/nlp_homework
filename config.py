import torch
PATH = "data\\"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PAD = 0
CLA = 1
SEP = 2
UNK = 3
SEP = 4
BatchSize = 26
EmbedSize = 150
d_ff = 1024
N_Layers =24
Vocab_Size=5059
N_Heads = 3
Dropout =0.1