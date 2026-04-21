import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy.linalg import sqrtm
import warnings
import os
import sys
sys.path.append('.')

warnings.filterwarnings('ignore')

def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2: return 0.0
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return np.abs(np.mean(group1) - np.mean(group2)) / s if s > 0 else 0.0

def process_and_evaluate(df_all, features, active_label='stress'):
    """
    Evaluates normalization methods ensuring strict isolation of train/test data 
    and proper dimension-matching.
    """
    subjects = df_all['subject_id'].unique()
    
    # Pre-split contiguous blocks for ALL subjects to allow fitting Global Scaler properly
    data_dict = {}
    X_train_global_list = []
    
    for subj in subjects:
        subj_df = df_all[df_all['subject_id'] == subj].copy()
        subj_df['label_name'] = subj_df['label']
        
        df_eval = subj_df[subj_df['label_name'].isin(['baseline', active_label])].copy()
        if df_eval.empty or len(df_eval['label_name'].unique()) < 2:
            continue
            
        base_idx = np.where(df_eval['label_name'] == 'baseline')[0]
        act_idx = np.where(df_eval['label_name'] == active_label)[0]
        
        # 50/50 Block Split (First half train, Second half test) - NO RANDOM SHUFFLING
        split_b = int(len(base_idx) * 0.5)
        split_a = int(len(act_idx) * 0.5)
        
        train_idx = np.concatenate([base_idx[:split_b], act_idx[:split_a]])
        test_idx = np.concatenate([base_idx[split_b:], act_idx[split_a:]])
        
        X_raw = df_eval[features].values
        y_raw = df_eval['label_name'].values
        
        # We need the indices of the 'baseline' strictly inside the train split for DBA
        base_train_mask = (y_raw[train_idx] == 'baseline')
        
        # Compute proper causal rolling Z-score over the continuous stream
        # Rolling only uses past data, so it's intrinsically leakage-free at any time t
        rolling_window = 6000
        df_rolling = df_eval[features].copy()
        for col in features:
            mean_r = df_eval[col].rolling(window=rolling_window, min_periods=1).mean()
            std_r = df_eval[col].rolling(window=rolling_window, min_periods=1).std().replace(0, 1)
            df_rolling[col] = (df_eval[col] - mean_r) / std_r
        X_rolling = df_rolling.fillna(0).values
        
        data_dict[subj] = {
            'X_raw': X_raw,
            'y_raw': y_raw,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'base_train_mask': base_train_mask,
            'X_rolling': X_rolling
        }
        
        # Add to global train set
        X_train_global_list.append(X_raw[train_idx])
        
    if not X_train_global_list:
        return pd.DataFrame()
        
    results = []
    
    for subj, d in data_dict.items():
        X_train_raw = d['X_raw'][d['train_idx']]
        X_test_raw = d['X_raw'][d['test_idx']]
        y_train = d['y_raw'][d['train_idx']]
        y_test = d['y_raw'][d['test_idx']]
        
        # 1. Subject-wise Z-Score (Fit only on this subject's train chunk)
        subj_scaler = StandardScaler().fit(X_train_raw)
        X_train_sz = subj_scaler.transform(X_train_raw)
        X_test_sz = subj_scaler.transform(X_test_raw)
        
        # 3. Rolling Z-Score
        X_train_rz = d['X_rolling'][d['train_idx']]
        X_test_rz = d['X_rolling'][d['test_idx']]
        
        # 4. DBA (Mahalanobis Whitening) -> n-dimensional output
        X_train_base = X_train_raw[d['base_train_mask']]
        
        mu_b = X_train_base.mean(axis=0)
        cov_b = np.cov(X_train_base, rowvar=False)
        cov_b += np.eye(cov_b.shape[0]) * 1e-6 # Epsilon for stability
        
        try:
            cov_b_inv = np.linalg.inv(cov_b)
            # Whitening matrix W = Sigma^(-1/2)
            W = sqrtm(cov_b_inv).real 
            X_train_dba = (X_train_raw - mu_b) @ W
            X_test_dba = (X_test_raw - mu_b) @ W
        except np.linalg.LinAlgError:
            continue
            
        methods = {
            'Subject_Z': (X_train_sz, X_test_sz),
            'Rolling_Z': (X_train_rz, X_test_rz),
            'DBA': (X_train_dba, X_test_dba)
        }
        
        res = {'Subject': subj}
        
        for m_name, (X_tr, X_te) in methods.items():
            # Linear Classification completely on strictly held-out block
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(X_tr, y_train)
            preds = clf.predict(X_te)
            f1 = f1_score(y_test, preds, pos_label=active_label)
            
            # Compute Effect Size (Cohen's d) on the Test Set distribution
            idx_base_te = (y_test == 'baseline')
            idx_act_te = (y_test == active_label)
            
            b_data = X_te[idx_base_te]
            a_data = X_te[idx_act_te]
            
            d_score = np.mean([compute_cohens_d(b_data[:, i], a_data[:, i]) for i in range(X_te.shape[1])])
            
            res[f'{m_name}_F1'] = f1
            res[f'{m_name}_d'] = d_score
            
        results.append(res)
        
    return pd.DataFrame(results)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--active_label', type=str, default='stress')
    args = parser.parse_args()

    print(f"Loading raw features from {args.input}...")
    df_all = pd.read_csv(args.input)
    features = ['HRV_Inst_LF', 'HRV_Inst_HF', 'EDA_Phasic', 'EDA_Tonic']
    
    print(f"Evaluating subjects preventing temporal leakage using Blocked Splits...")
    df_res = process_and_evaluate(df_all, features, active_label=args.active_label)
    
    if not df_res.empty:
        print("\n--- RESULTS OVERVIEW ---")
        
        from scipy.stats import wilcoxon
        print("Mean +/- Std:")
        print(f"Subject_Z F1: {df_res['Subject_Z_F1'].mean():.2f} +/- {df_res['Subject_Z_F1'].std():.2f}")
        print(f"Rolling_Z F1: {df_res['Rolling_Z_F1'].mean():.2f} +/- {df_res['Rolling_Z_F1'].std():.2f}")
        print(f"DBA F1: {df_res['DBA_F1'].mean():.2f} +/- {df_res['DBA_F1'].std():.2f}")
        
        # Paired Wilcoxon Test
        stat, p = wilcoxon(df_res['DBA_F1'], df_res['Rolling_Z_F1'])
        print(f"\nWilcoxon Test (DBA vs Rolling): p-value = {p:.2e}")
        
        df_res.to_csv(args.output, index=False)
        print(f"\nSaved detailed results to {args.output}")
    else:
        print("No valid results computed.")
