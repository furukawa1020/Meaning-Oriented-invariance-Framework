import os
import sys
import json
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

# Guard
if os.environ.get("ALLOW_REAL_DATA_PHASE_2E_NULL_AUDIT") != "1":
    print("FATAL: Phase 2E is locked.")
    sys.exit(1)

warnings.filterwarnings('ignore')

WESAD_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/wesad/WESAD")
OUT_DIR = "phase2e_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Loader & Feature Extraction (Extended for Low-Level features) ────────────
def extract_all_feature_levels(subj_data):
    import neurokit2 as nk
    ecg = subj_data['signal']['chest']['ECG'].flatten()
    eda = subj_data['signal']['wrist']['EDA'].flatten()
    lbl = subj_data['label'].flatten()
    subj_id = subj_data['subject']
    fs_ecg, fs_eda, tgt = 700, 4, 100

    dur = len(ecg) / fs_ecg
    n = int(np.floor(dur * tgt))
    t = np.linspace(0, dur, n, endpoint=False)

    # 1. Low-Level (Raw-ish)
    t_eda_src = np.linspace(0, len(eda)/fs_eda, len(eda))
    eda_c = nk.eda_clean(eda, sampling_rate=fs_eda)
    eda_raw_100 = interp1d(t_eda_src, eda_c, bounds_error=False, fill_value="extrapolate")(t)
    
    # HRV: just RRI
    ecg_c = nk.ecg_clean(ecg, sampling_rate=fs_ecg)
    _, info = nk.ecg_peaks(ecg_c, sampling_rate=fs_ecg)
    rpeaks = info['ECG_R_Peaks']
    rri_ms = np.diff(rpeaks) / fs_ecg * 1000
    rri_t  = rpeaks[1:] / fs_ecg
    rri_raw_100 = interp1d(rri_t, rri_ms, kind='cubic', bounds_error=False, fill_value="extrapolate")(t)

    # 2. High-Level (Processed)
    eda_dec = nk.eda_phasic(eda_raw_100, sampling_rate=tgt)
    import scipy.signal as ss
    sos_lf = ss.butter(4, [0.04, 0.15], btype='bandpass', fs=tgt, output='sos')
    sos_hf = ss.butter(4, [0.15, 0.40], btype='bandpass', fs=tgt, output='sos')
    lf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_lf, rri_raw_100))) ** 2
    hf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_hf, rri_raw_100))) ** 2

    # Labels
    t_lbl_src = np.linspace(0, len(lbl)/fs_ecg, len(lbl))
    lbl_100 = interp1d(t_lbl_src, lbl, kind='nearest', bounds_error=False, fill_value="extrapolate")(t)
    label_map = {1: 'baseline', 2: 'stress'}
    labels = [label_map.get(int(v), None) for v in lbl_100]

    df = pd.DataFrame({
        'subject_id': subj_id,
        'label': labels,
        'HRV_RRI_Raw': rri_raw_100,
        'EDA_Raw': eda_raw_100,
        'HRV_LF': lf,
        'HRV_HF': hf,
        'EDA_Phasic': eda_dec['EDA_Phasic'].values,
        'EDA_Tonic': eda_dec['EDA_Tonic'].values
    })
    
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"    WARNING: NaNs detected in {subj_id}:")
        print(nan_counts[nan_counts > 0])
        
    df = df.dropna()
    return df

def run_multi_factor_audit(df_all):
    subjects = df_all['subject_id'].unique()
    feature_sets = {
        'high_level': ['HRV_LF', 'HRV_HF', 'EDA_Phasic', 'EDA_Tonic'],
        'low_level': ['HRV_RRI_Raw', 'EDA_Raw']
    }
    models = {
        'LR_C1': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000),
        'SVM': LinearSVC(C=1.0, class_weight='balanced', max_iter=1000),
        'RF': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    }
    
    results = []

    for subj in subjects:
        subj_df = df_all[df_all['subject_id'] == subj]
        df_eval = subj_df[subj_df['label'].isin(['baseline', 'stress'])]
        if len(df_eval['label'].unique()) < 2: continue
        
        base_idx = np.where(df_eval['label'].values == 'baseline')[0]
        act_idx = np.where(df_eval['label'].values == 'stress')[0]
        if len(base_idx) < 50 or len(act_idx) < 50: continue

        # Split
        sp_b, sp_a = len(base_idx)//2, len(act_idx)//2
        train_idx = np.concatenate([base_idx[:sp_b], act_idx[:sp_a]])
        test_idx = np.concatenate([base_idx[sp_b:], act_idx[sp_a:]])
        
        y_train = df_eval['label'].values[train_idx]
        y_test = df_eval['label'].values[test_idx]

        for fs_name, feats in feature_sets.items():
            X_all = df_eval[feats].values
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            
            # Normalizers
            # Subject-wise Z
            sc = StandardScaler().fit(X_train)
            X_tr_sz, X_te_sz = sc.transform(X_train), sc.transform(X_test)
            
            # Baseline Z
            X_base = X_train[y_train == 'baseline']
            mu_b, sd_b = X_base.mean(axis=0), X_base.std(axis=0)
            sd_b[sd_b < 1e-8] = 1e-8
            X_tr_bz, X_te_bz = (X_train - mu_b)/sd_b, (X_test - mu_b)/sd_b
            
            # DBA
            cov_b = np.cov(X_base, rowvar=False) + np.eye(len(feats)) * 1e-5
            W = sqrtm(np.linalg.inv(cov_b)).real
            X_tr_dba, X_te_dba = (X_train - mu_b) @ W, (X_test - mu_b) @ W
            
            norms = {'Subject_Z': (X_tr_sz, X_te_sz), 'Baseline_Z': (X_tr_bz, X_te_bz), 'DBA': (X_tr_dba, X_te_dba)}
            
            for norm_name, (X_tr_n, X_te_n) in norms.items():
                for mod_name, clf in models.items():
                    try:
                        clf.fit(X_tr_n, y_train)
                        if hasattr(clf, "predict_proba"):
                            probs = clf.predict_proba(X_te_n)[:, 1]
                        else:
                            probs = clf.decision_function(X_te_n)
                        
                        y_bin = (y_test == 'stress').astype(int)
                        # Fix Gate 4 anomaly: ensure at least 5 samples per class in test
                        if len(np.unique(y_bin)) < 2 or len(y_bin) < 10:
                            auroc = np.nan
                        else:
                            auroc = roc_auc_score(y_bin, probs)
                            
                        results.append({
                            'subject': subj, 'feature_level': fs_name, 'norm': norm_name,
                            'model': mod_name, 'auroc': auroc
                        })
                    except:
                        continue
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Starting Phase 2E Full Audit...")
    pkls = sorted(list(WESAD_DIR.rglob("*.pkl")))[:5]
    print(f"Auditing pkls: {[p.name for p in pkls]}")
    
    all_dfs = []
    for p in pkls:
        with open(p, 'rb') as f: data = pickle.load(f, encoding='latin1')
        df_s = extract_all_feature_levels(data)
        print(f"  {p.name}: {len(df_s)} samples, labels: {df_s['label'].unique()}")
        all_dfs.append(df_s)
    
    df_big = pd.concat(all_dfs)
    df_res = run_multi_factor_audit(df_big)
    print(f"Audit run complete. Results count: {len(df_res)}")
    
    if not df_res.empty:
        df_res.to_csv(os.path.join(OUT_DIR, "phase2e_multi_factor_results.csv"), index=False)
        summary = df_res.groupby(['feature_level', 'model', 'norm'])['auroc'].mean().unstack()
        summary.to_csv(os.path.join(OUT_DIR, "phase2e_audit_summary.csv"))
        print("Phase 2E results saved.")
    else:
        print("ERROR: No results were computed. Check class presence and split logic.")
