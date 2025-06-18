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
        self.alpha= 0.1
        self.window = 95
        # window must be less than seq_len

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.exp_smoothing(x[:,:,i])
            x = output
        else:
            x = self.exp_smoothing(x)
        return x # [Batch, Output length, Channel]
    

    def exp_smoothing(self,X,  noise_coef=0.0):
        # batch x lookback x variates
        # forecast is a weighted average of inputs: Y[:, 1] = \sum_j alpha (1-alpa)^j X[:, j]
        coefs = (1-self.alpha) ** torch.linspace(0, self.window+1, steps=self.window+1).to(X.device)
        coefs = torch.flip(coefs, dims=(0,))
        coefs  = coefs*self.alpha
  
        Y = X
        for jj in range(self.pred_len):
            y = torch.sum( Y[:, -(self.window+1):] * coefs[None, ..., None], dim=1)	# broadcasting
            Y = torch.cat((Y, y.unsqueeze(1)), dim=1)

        # generate noise
        E = torch.mean(torch.abs(X)) * noise_coef * torch.randn(Y[:, -self.pred_len:].shape).to(X.device) 

        # batch x horizon x variates
        return Y[:, -self.pred_len:] + E