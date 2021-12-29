import torch
from torch import nn
from const import *
# todo: vocab, embedding 渡して vocab_size, embed_dim は渡さないようにする

class LSTMClassifier(nn.Module):
    
    def __init__(self, embedding, h_dim, class_dim):
        super(LSTMClassifier, self).__init__()
        torch.manual_seed(SEED)
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.lstm = nn.LSTM(embedding.shape[1], h_dim, batch_first=True)
        self.linear = nn.Linear(h_dim, class_dim)
        
    def forward(self, texts):
        '''Calculate an output sequence for input text
        
        Args:
          texts: 2 rank array of 
        Returns:
        
        '''
        text_embedding = self.embedding(texts)
        o, (h_n, c_n) = self.lstm(text_embedding)
        out = self.linear(h_n)
        return out

if __name__ == "__main__":
    pass