import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        
        if self.individual:
            self.weights = nn.ParameterList([
                nn.Parameter(torch.zeros(self.seq_len, self.pred_len), requires_grad=False)
                for _ in range(self.channels)
            ])
            self.biases = nn.ParameterList([
                nn.Parameter(torch.zeros(self.pred_len), requires_grad=False)
                for _ in range(self.channels)
            ])
        else:
            self.weights = nn.Parameter(torch.zeros(self.seq_len, self.pred_len), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.pred_len), requires_grad=False)
        
        self.fitted = False
    
    def fit(self, X_all, Y_all, device='cuda'):
        N = X_all.shape[0]
        
        if self.individual:
            for c in range(self.channels):
                X_c = X_all[:, :, c]
                Y_c = Y_all[:, :, c]
                X_aug = np.concatenate([X_c, np.ones((N, 1))], axis=1)
                W_aug = np.linalg.lstsq(X_aug, Y_c, rcond=None)[0]
                W_c = W_aug[:-1, :]
                b_c = W_aug[-1, :]
                self.weights[c].data = torch.from_numpy(W_c).float().to(device)
                self.biases[c].data = torch.from_numpy(b_c).float().to(device)
        else:
            X_flat = X_all.transpose(0, 2, 1).reshape(-1, self.seq_len)
            Y_flat = Y_all.transpose(0, 2, 1).reshape(-1, self.pred_len)
            X_aug = np.concatenate([X_flat, np.ones((X_flat.shape[0], 1))], axis=1)
            W_aug = np.linalg.lstsq(X_aug, Y_flat, rcond=None)[0]
            W = W_aug[:-1, :]
            b = W_aug[-1, :]
            self.weights.data = torch.from_numpy(W).float().to(device)
            self.bias.data = torch.from_numpy(b).float().to(device)
        
        self.fitted = True
        print(f"[Closed-Form] Weights computed.")
    
    def forward(self, x):
        if self.individual:
            batch_size = x.size(0)
            output = torch.zeros(batch_size, self.pred_len, self.channels, device=x.device, dtype=x.dtype)
            for c in range(self.channels):
                output[:, :, c] = torch.matmul(x[:, :, c], self.weights[c]) + self.biases[c]
            return output
        else:
            x = x.permute(0, 2, 1)
            out = torch.matmul(x, self.weights) + self.bias
            return out.permute(0, 2, 1)
