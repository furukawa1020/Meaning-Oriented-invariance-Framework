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
from sklearn.metrics import roc_auc_score, matthews_corrcoef, balanced_accuracy_score

# Guard
if os.environ.get("ALLOW_REAL_DATA_PHASE_2F_VERIFICATION") != "1":
    print("FATAL: Phase 2F is locked.")
    sys.exit(1)

warnings.filterwarnings('ignore')

WESAD_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/wesad/WESAD")
CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase2f_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Loader & Window-Level Extraction ──────────────────────────────────────────
def get_windowed_data(df, window_size=100, label_col='label'):
    """Aggregate 100Hz samples into 1-second non-overlapping windows."""
    feats = ['HRV_LF', 'HRV_HF', 'EDA_Phasic', 'EDA_Tonic']
    n = len(df)
    n_windows = n // window_size
    if n_windows == 0: return None
    
    # Reshape features to (n_windows, window_size, n_feats) and take mean
    X_raw = df[feats].values[:n_windows * window_size]
    X_windowed = X_raw.reshape(n_windows, window_size, -1).mean(axis=1)
    
    # Majority vote for labels
    y_raw = df[label_col].values[:n_windows * window_size]
    y_reshaped = y_raw.reshape(n_windows, window_size)
    y_windowed = []
    for i in range(n_windows):
        vals, counts = np.unique(y_reshaped[i], return_counts=True)
        y_windowed.append(vals[np.argmax(counts)])
    
    return X_windowed, np.array(y_windowed)

def process_subject_wesad(pkl_path):
    import neurokit2 as nk
    with open(pkl_path, 'rb') as f: data = pickle.load(f, encoding='latin1')
    ecg = data['signal']['chest']['ECG'].flatten()
    eda = data['signal']['wrist']['EDA'].flatten()
    lbl = data['label'].flatten()
    fs_ecg, fs_eda, tgt = 700, 4, 100
    dur = len(ecg)/fs_ecg
    n = int(np.floor(dur*tgt))
    t = np.linspace(0, dur, n, endpoint=False)
    
    # Simple clean + resample
    eda_c = nk.eda_clean(eda, sampling_rate=fs_eda)
    eda_100 = interp1d(np.linspace(0, len(eda)/fs_eda, len(eda)), eda_c, fill_value="extrapolate")(t)
    rpeaks = nk.ecg_peaks(nk.ecg_clean(ecg, sampling_rate=fs_ecg), sampling_rate=fs_ecg)[1]['ECG_R_Peaks']
    rri_ms = np.diff(rpeaks)/fs_ecg*1000
    rri_100 = interp1d(rpeaks[1:]/fs_ecg, rri_ms, kind='cubic', fill_value="extrapolate")(t)
    
    eda_dec = nk.eda_phasic(eda_100, sampling_rate=tgt)
    import scipy.signal as ss
    sos_lf = ss.butter(4, [0.04, 0.15], btype='bandpass', fs=tgt, output='sos')
    sos_hf = ss.butter(4, [0.15, 0.40], btype='bandpass', fs=tgt, output='sos')
    lf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_lf, rri_100)))**2
    hf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_hf, rri_100)))**2
    
    lbl_100 = interp1d(np.linspace(0, len(lbl)/fs_ecg, len(lbl)), lbl, kind='nearest', fill_value="extrapolate")(t)
    lmap = {1: 'baseline', 2: 'stress'}
    labels = [lmap.get(int(v), None) for v in lbl_100]
    
    df = pd.DataFrame({'HRV_LF': lf, 'HRV_HF': hf, 'EDA_Phasic': eda_dec['EDA_Phasic'].values, 
                       'EDA_Tonic': eda_dec['EDA_Tonic'].values, 'label': labels}).dropna()
    return df

def process_subject_case(sub_id):
    import neurokit2 as nk
    p_path = CASE_DIR / 'data' / 'interpolated' / 'physiological' / f"{sub_id}.csv"
    if not p_path.exists(): return None
    df_p = pd.read_csv(p_path)
    tgt = 100
    t_p = df_p['daqtime'].values / 1000.0
    t = np.linspace(0, t_p[-1], int(t_p[-1]*tgt), endpoint=False)
    
    eda_c = nk.eda_clean(df_p['gsr'].values, sampling_rate=tgt)
    eda_100 = interp1d(t_p, eda_c, fill_value="extrapolate")(t)
    eda_dec = nk.eda_phasic(eda_100, sampling_rate=tgt)
    
    ecg_c = nk.ecg_clean(df_p['ecg'].values, sampling_rate=tgt)
    rpeaks = nk.ecg_peaks(ecg_c, sampling_rate=tgt)[1]['ECG_R_Peaks']
    rri_ms = np.diff(rpeaks)/tgt*1000
    rri_100 = interp1d(rpeaks[1:]/tgt, rri_ms, kind='cubic', fill_value="extrapolate")(t)
    
    import scipy.signal as ss
    sos_lf = ss.butter(4, [0.04, 0.15], btype='bandpass', fs=tgt, output='sos')
    sos_hf = ss.butter(4, [0.15, 0.40], btype='bandpass', fs=tgt, output='sos')
    lf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_lf, rri_100)))**2
    hf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_hf, rri_100)))**2
    
    vid_100 = interp1d(t_p, df_p['video'].values, kind='nearest', fill_value="extrapolate")(t)
    lmap = {10: 'baseline', 1: 'stress', 2: 'stress'}
    labels = [lmap.get(int(v), None) for v in vid_100]
    
    df = pd.DataFrame({'HRV_LF': lf, 'HRV_HF': hf, 'EDA_Phasic': eda_dec['EDA_Phasic'].values, 
                       'EDA_Tonic': eda_dec['EDA_Tonic'].values, 'label': labels}).dropna()
    return df

# ── Evaluation Logic ─────────────────────────────────────────────────────────
def run_verification():
    models = {
        'RF': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'LR': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000),
        'SVM': LinearSVC(C=1.0, class_weight='balanced', max_iter=1000)
    }
    
    results = []
    
    # 1. WESAD Subjects
    wesad_pkls = sorted(list(WESAD_DIR.rglob("*.pkl")))
    for p in wesad_pkls:
        print(f"Processing WESAD {p.name}...")
        df_full = process_subject_wesad(p)
        res = evaluate_subject(df_full, p.stem, 'WESAD', models)
        results.extend(res)
        
    # 2. CASE Subjects
    for i in range(1, 31):
        sid = f"sub_{i}"
        print(f"Processing CASE {sid}...")
        df_full = process_subject_case(sid)
        if df_full is not None:
            res = evaluate_subject(df_full, sid, 'CASE', models)
            results.extend(res)
            
    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "phase2f_full_results.csv"), index=False)

def evaluate_subject(df_full, sid, dataset, models):
    # Windowing (100 samples = 1 second)
    windowed = get_windowed_data(df_full)
    if windowed is None: return []
    X_win, y_win = windowed
    
    # Split (First half train, second half test)
    base_idx = np.where(y_win == 'baseline')[0]
    act_idx = np.where(y_win == 'stress')[0]
    if len(base_idx) < 10 or len(act_idx) < 10: return []
    
    sp_b, sp_a = len(base_idx)//2, len(act_idx)//2
    train_idx = np.concatenate([base_idx[:sp_b], act_idx[:sp_a]])
    test_idx = np.concatenate([base_idx[sp_b:], act_idx[sp_a:]])
    
    X_tr, X_te = X_win[train_idx], X_win[test_idx]
    y_tr, y_te = y_win[train_idx], y_win[test_idx]
    
    # Normalizers
    # 1. Subject-wise Z
    sc = StandardScaler().fit(X_tr)
    # 2. Baseline Z
    X_base = X_tr[y_tr == 'baseline']
    mu_b, sd_b = X_base.mean(axis=0), X_base.std(axis=0)
    sd_b[sd_b < 1e-8] = 1e-8
    # 3. Covariance Calibration
    cov_b = np.cov(X_base, rowvar=False) + np.eye(X_win.shape[1])*1e-5
    W = sqrtm(np.linalg.inv(cov_b)).real
    
    norms = {
        'subject_z': (sc.transform(X_tr), sc.transform(X_te)),
        'baseline_z': ((X_tr - mu_b)/sd_b, (X_te - mu_b)/sd_b),
        'covariance_calibration': ((X_tr - mu_b)@W, (X_te - mu_b)@W)
    }
    
    subj_results = []
    for norm_name, (X_tr_n, X_te_n) in norms.items():
        for mod_name, clf in models.items():
            try:
                clf.fit(X_tr_n, y_tr)
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_te_n)[:, 1]
                else:
                    probs = clf.decision_function(X_te_n)
                
                y_bin = (y_te == 'stress').astype(int)
                if len(np.unique(y_bin)) < 2: continue
                
                auroc = roc_auc_score(y_bin, probs)
                y_pred = clf.predict(X_te_n)
                mcc = matthews_corrcoef(y_te, y_pred)
                b_acc = balanced_accuracy_score(y_te, y_pred)
                
                subj_results.append({
                    'dataset': dataset, 'subject_id': sid, 'norm': norm_name, 'model': mod_name,
                    'auroc': auroc, 'mcc': mcc, 'balanced_accuracy': b_acc
                })
            except: continue
    return subj_results

if __name__ == "__main__":
    print(f"--- Phase 2F Verification: Window-Level Audit ---")
    print(f"Start: {datetime.datetime.now()}")
    run_verification()
    print(f"End: {datetime.datetime.now()}")
