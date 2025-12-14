import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    """
    Linear model with closed-form (analytic) solution.
    Instead of iterative SGD training, weights are computed 
    directly using the Normal Equation.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len      # L: lookback window length
        self.pred_len = configs.pred_len    # T: prediction horizon
        self.channels = configs.enc_in      # Number of variables
        self.individual = configs.individual
        
        # Initialize weight matrices (will be set analytically)
        if self.individual:
            # Separate weights for each channel: list of (L, T) matrices
            self.weights = nn.ParameterList([
                nn.Parameter(torch.zeros(self.seq_len, self.pred_len), requires_grad=False)
                for _ in range(self.channels)
            ])
            self.biases = nn.ParameterList([
                nn.Parameter(torch.zeros(self.pred_len), requires_grad=False)
                for _ in range(self.channels)
            ])
        else:
            # Shared weights across channels: (L, T) matrix
            self.weights = nn.Parameter(torch.zeros(self.seq_len, self.pred_len), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.pred_len), requires_grad=False)
        
        self.fitted = False
    
    def fit(self, train_loader, device='cuda'):
        """
        Compute optimal weights using closed-form solution (Normal Equation).
        W* = (X^T X)^{-1} X^T Y  or equivalently  W* = pinv(X) @ Y
        """
        # Collect all training data
        X_list = []
        Y_list = []
        
        for batch_x, batch_y, _, _ in train_loader:
            # batch_x: [B, seq_len, channels]
            # batch_y: [B, pred_len, channels] (we need label_len handling)
            X_list.append(batch_x.numpy())
            Y_list.append(batch_y[:, -self.pred_len:, :].numpy())  # Take last pred_len steps
        
        X_all = np.concatenate(X_list, axis=0)  # [N, seq_len, channels]
        Y_all = np.concatenate(Y_list, axis=0)  # [N, pred_len, channels]
        
        N = X_all.shape[0]
        
        if self.individual:
            # Solve separately for each channel
            for c in range(self.channels):
                X_c = X_all[:, :, c]  # [N, seq_len]
                Y_c = Y_all[:, :, c]  # [N, pred_len]
                
                # Add bias term: augment X with column of ones
                X_aug = np.concatenate([X_c, np.ones((N, 1))], axis=1)  # [N, seq_len+1]
                
                # Closed-form solution: W = pinv(X) @ Y
                W_aug = np.linalg.lstsq(X_aug, Y_c, rcond=None)[0]  # [seq_len+1, pred_len]
                
                # Extract weights and bias
                W_c = W_aug[:-1, :]  # [seq_len, pred_len]
                b_c = W_aug[-1, :]   # [pred_len]
                
                self.weights[c].data = torch.from_numpy(W_c).float().to(device)
                self.biases[c].data = torch.from_numpy(b_c).float().to(device)
        else:
            # Shared weights: reshape and solve
            # Treat all channels together
            X_flat = X_all.transpose(0, 2, 1).reshape(-1, self.seq_len)  # [N*channels, seq_len]
            Y_flat = Y_all.transpose(0, 2, 1).reshape(-1, self.pred_len)  # [N*channels, pred_len]
            
            # Add bias term
            X_aug = np.concatenate([X_flat, np.ones((X_flat.shape[0], 1))], axis=1)
            
            # Closed-form solution
            W_aug = np.linalg.lstsq(X_aug, Y_flat, rcond=None)[0]
            
            W = W_aug[:-1, :]  # [seq_len, pred_len]
            b = W_aug[-1, :]   # [pred_len]
            
            self.weights.data = torch.from_numpy(W).float().to(device)
            self.bias.data = torch.from_numpy(b).float().to(device)
        
        self.fitted = True
        print(f"Closed-form solution computed. Weight matrix shape: {self.weights.shape if not self.individual else self.weights[0].shape}")
    
    def forward(self, x):
        """
        Forward pass: y = xW + b
        x: [Batch, seq_len, channels]
        """
        if self.individual:
            batch_size = x.size(0)
            output = torch.zeros(batch_size, self.pred_len, self.channels, device=x.device)
            for c in range(self.channels):
                # x[:, :, c]: [B, seq_len]
                # weights[c]: [seq_len, pred_len]
                output[:, :, c] = torch.matmul(x[:, :, c], self.weights[c]) + self.biases[c]
            return output
        else:
            # x: [B, seq_len, channels] -> permute to [B, channels, seq_len]
            # matmul with weights [seq_len, pred_len] -> [B, channels, pred_len]
            x = x.permute(0, 2, 1)  # [B, channels, seq_len]
            out = torch.matmul(x, self.weights) + self.bias  # [B, channels, pred_len]
            return out.permute(0, 2, 1)  # [B, pred_len, channels]
