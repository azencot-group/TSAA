import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # for seasonal naive seasonality>0
        self.seasonality = min(configs.seasonality, configs.pred_len)
        self.channels = configs.enc_in
        self.individual = configs.individual

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.naive_forecast(x[:,:,i])
            x = output
        else:
            x = self.naive_forecast(x)
        return x # [Batch, Output length, Channel]
    

    def naive_forecast(self, X,  noise_coef=0.1):
        # batch x lookback x variates
        # forecast is the last obtained input, Y[:, j] = X[:, -1] + noise
        if len(X.shape) == 2:
            X = X.unsqueeze(2)
        b, d = X.shape[0], X.shape[-1]

        Y = torch.zeros((b, self.pred_len, d)).to(X.device)
        for jj in range(self.pred_len):
            if self.seasonality>0:
                Y[:, jj] = X[:, jj % self.seasonality - self.seasonality]		# this code breaks if s > l
            else:
                Y[:, jj] = X[:, -1]				# naive forecast

        # generate noise
        E = torch.mean(torch.abs(X)) * noise_coef * torch.rand(Y.shape).to(X.device)

        # batch x horizon x variates
        return Y + E