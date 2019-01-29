import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self, params):
        super(Net, self).__init__()

      	# LSTM F
        self.lstm_f = nn.LSTM(input_size=params.lstm_f.input_size, hidden_size=params.lstm_f.hidden_size,
        	batch_first=True, num_layers=params.lstm_f.num_layers, dropout=params.lstm_f.dropout)
        self.hidden_f = Variable()

        #LSTM Q
        self.lstm_q = nn.LSTM(input_size=params.lstm_q.input_size, hidden_size=params.lstm_q.hidden_size,
        	batch_first=True, num_layers=params.lstm_q.num_layers, dropout=params.lstm_q.dropout)

        #LSTM R
        self.lstm_r = nn.LSTM(input_size=params.lstm_r.input_size, hidden_size=params.lstm_r.hidden_size,
        	batch_first=True, num_layers=params.lstm_r.num_layers, dropout=params.lstm_r.dropout)

        
    def forward(self, s):
        s = self.embedding(s)            # dim: batch_size x seq_len x embedding_dim
        s, _ = self.lstm(s)              # dim: batch_size x seq_len x lstm_hidden_dim

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        s = s.view(-1, s.shape[2])       # dim: batch_size*seq_len x lstm_hidden_dim

        s = self.fc(s)                   # dim: batch_size*seq_len x num_tags

        return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags
