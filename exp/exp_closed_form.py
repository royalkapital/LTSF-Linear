import os
import time
import torch
import numpy as np
from data_provider.data_factory import data_provider
from models.Linear_ClosedForm import Model

class Exp_ClosedForm:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(args).to(self.device)
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        
        print("=" * 50)
        print("Collecting training data...")
        
        X_list = []
        Y_list = []
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            X_list.append(batch_x.numpy())
            Y_list.append(batch_y[:, -self.args.pred_len:, :].numpy())
        
        X_all = np.concatenate(X_list, axis=0)
        Y_all = np.concatenate(Y_list, axis=0)
        
        print(f"Training data shape: X={X_all.shape}, Y={Y_all.shape}")
        print("Computing closed-form solution...")
        
        start_time = time.time()
        self.model.fit(X_all, Y_all, device=self.device)
        train_time = time.time() - start_time
        
        print(f"Training complete! Time: {train_time:.4f} seconds")
        print("=" * 50)
        
        return self.model, train_time
    
    def test(self):
        test_data, test_loader = self._get_data(flag='test')
        
        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].float()
                
                outputs = self.model(batch_x)
                
                preds.append(outputs.cpu().numpy())
                trues.append(batch_y.numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        
        print("=" * 50)
        print("TEST RESULTS (Closed-Form Solution)")
        print("=" * 50)
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print("=" * 50)
        
        return mse, mae
