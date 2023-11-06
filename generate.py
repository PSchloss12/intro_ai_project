import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import torchtext

import numpy as np
from tqdm import tqdm
import math
from random import randint

from utils import generate
# from model import LTSM
from data import get_vocab_tokenizer

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate,
                tie_weights):

        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                    dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # The purpose of this is to make the embedding layer share weights with the output layer. This helps reduce the number of parameters
        if tie_weights:
            assert embedding_dim == hidden_dim, 'cannot tie, check dims'
            self.embedding.weight = self.fc.weight
        self.init_weights()

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

    def init_weights(self):
      '''
      initialize the embedding weights uniformly in the range [-0.1, 0.1]
      and all other layers uniformly in the range [-1/sqrt(H), 1/sqrt(H)]
      '''
      init_range_emb = 0.1
      init_range_other = 1/math.sqrt(self.hidden_dim)
      self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
      self.fc.weight.data.uniform_(-init_range_other, init_range_other)
      self.fc.bias.data.zero_()
      for i in range(self.num_layers):
          self.lstm.all_weights[i][0] = torch.FloatTensor(self.embedding_dim,
                  self.hidden_dim).uniform_(-init_range_other, init_range_other)
          self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_dim,
                  self.hidden_dim).uniform_(-init_range_other, init_range_other)

    def init_hidden(self, batch_size, device):
        '''
        set the LSTM’s hidden and cell state to zero
        '''
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        '''
        need this function while training to explicitly tell PyTorch that hidden states due to different sequences are independent
        '''
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

punctuation = set(['.',',','"',"'"])

# set the embedding and hidden dimensions as the same value because we will use weight tying
embedding_dim = 1024             # 400 in the paper
hidden_dim = 1024                # 1150 in the paper
num_layers = 2                   # 3 in the paper
dropout_rate = 0.65
tie_weights = True
lr = 1e-3                        # They used 30 and a different optimizer

def evaluate():
    # evaluate model
    n_epochs = 50
    seq_len = 50
    clip = 0.25
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
    test_loss = evaluate(model, test_data, criterion, batch_size, seq_len, device)
    print(f'Test Perplexity: {math.exp(test_loss):.3f}')

def respond(prompt, max_len, temp, seed):
    generation = generate(prompt, max_len, temp, model, tokenizer,
                                vocab, device, seed)
    for i in range(len(generation)):
        if generation[i] in punctuation and i>0:
            generation[i-1]+=generation[i]
            generation[i]=""
            i-=1
    return ' '.join(generation)

if __name__ == "__main__":
    device = "cpu"

    vocab, tokenizer = get_vocab_tokenizer()
    vocab_size = len(vocab)

    # initialize the model, optimizer and loss criterion
    model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # load neural net
    model.load_state_dict(torch.load('model-lstm_lm.pt',  map_location=device))

    while True:
        prompt = input("Enter prompt>> ")
        for t in [0.7,0.75,0.9,1]:
            seed = randint(0,vocab_size)
            print("seed:",seed)
            response = respond(prompt,10,t,seed)
            print(response, end="\n\n")