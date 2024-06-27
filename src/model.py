import torch
from torch import nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, label_size):
        super(TextClassifier, self).__init__()
        self.Emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.Linear = nn.Linear(hidden_size, label_size)
        
        nn.init.uniform_(self.Emb.weight, a=-1/hidden_size, b=1/hidden_size)
        nn.init.uniform_(self.Linear.weight, a=-1/hidden_size, b=1/hidden_size)
        
    def forward(self, x):
        # x - (batch, max lenth)
        h = self.Emb(x).mean(dim=1) #batch, max length, hidden
        h = self.Linear(h) #batch, label size
        return h