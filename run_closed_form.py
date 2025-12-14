import argparse
import torch
import random
import numpy as np
from exp.exp_closed_form import Exp_ClosedForm

seed = 2021
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='LTSF-Linear Closed-Form Solution')
    
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M', help='M, S, MS')
    parser.add_argument('--target', type=str, default='OT', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='freq')
    
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
    
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--individual', action='store_true', default=False, help='individual channel weights')
    
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='num workers')
    parser.add_argument('--embed', type=str, default='timeF', help='time encoding')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("   LTSF-Linear: CLOSED-FORM SOLUTION")
    print("   (Analytic global optimum - No iterative training)")
    print("=" * 60)
    print(f"Dataset:           {args.data}")
    print(f"Sequence Length:   {args.seq_len}")
    print(f"Prediction Length: {args.pred_len}")
    print("=" * 60 + "\n")
    
    exp = Exp_ClosedForm(args)
    model, train_time = exp.train()
    mse, mae = exp.test()
    
    print("\n" + "=" * 60)
    print("   FINAL RESULTS")
    print("=" * 60)
    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Test MSE:      {mse:.6f}")
    print(f"Test MAE:      {mae:.6f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
