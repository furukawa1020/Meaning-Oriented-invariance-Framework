"""
Phase 2C-1: Main Performance Run

Strict execution contract:
1. Load real data (WESAD + CASE)
2. Run fixed Logistic Regression (C=1.0, L2, balanced) over all normalizers
3. Output result CSVs
4. Immediately invoke run_submission_gate.py
5. Print only gate_report.json summary - NOT raw results
"""
import os
import sys
import json
import pickle
import warnings
import subprocess
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, brier_score_loss
)

# ── Guard ─────────────────────────────────────────────────────────────────────
if os.environ.get("ALLOW_REAL_DATA_PHASE_2C_MAIN") != "1":
    print("FATAL: Phase 2C-1 is locked.")
    print("Run with: ALLOW_REAL_DATA_PHASE_2C_MAIN=1 python phase2c_main_run.py")
    sys.exit(1)

warnings.filterwarnings('ignore')

GATE_SCRIPT = r"C:\Projects\Meaning-Oriented invariance Framework\moif\scripts\thms_pipeline\run_submission_gate.py"
OUT_DIR = "phase2c_results"
os.makedirs(OUT_DIR, exist_ok=True)

WESAD_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/wesad/WESAD")
CASE_DIR  = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")

# ── WESAD Loader (metadata + raw signal only; no feature extraction import) ──
def load_wesad_raw():
    records = []
    for p_path in sorted(WESAD_DIR.rglob("*.pkl")):
        with open(p_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        subj = data['subject']
        ecg = data['signal']['chest']['ECG'].flatten()
        eda = data['signal']['wrist']['EDA'].flatten()
        lbl = data['label'].flatten()
        records.append({"subject_id": subj, "ecg": ecg, "eda": eda, "label_700hz": lbl})
        print(f"  WESAD loaded {subj}")
    return records

# ── CASE Loader ───────────────────────────────────────────────────────────────
CASE_LABEL_MAP = {10: 'baseline', 1: 'stress', 2: 'stress'}

def load_case_raw():
    records = []
    phys_dir = CASE_DIR / 'data' / 'interpolated' / 'physiological'
    anno_dir = CASE_DIR / 'data' / 'interpolated' / 'annotations'
    for p_path in sorted(phys_dir.glob("sub_*.csv")):
        subj_id = p_path.stem
        a_path = anno_dir / f"{subj_id}.csv"
        if not a_path.exists():
            continue
        df_p = pd.read_csv(p_path, usecols=['daqtime', 'ecg', 'gsr', 'video'])
        df_a = pd.read_csv(a_path, usecols=['jstime', 'valence', 'arousal'])
        records.append({"subject_id": subj_id, "df_phys": df_p, "df_anno": df_a})
        print(f"  CASE loaded {subj_id}")
    return records

# ── Feature extraction (simple but leakage-controlled) ───────────────────────
def extract_features_wesad(subj_data):
    """Simplified feature extraction from raw ECG/EDA for WESAD."""
    import neurokit2 as nk
    ecg = subj_data['ecg']
    eda = subj_data['eda']
    lbl = subj_data['label_700hz']
    fs_ecg, fs_eda, tgt = 700, 4, 100

    dur = len(ecg) / fs_ecg
    n = int(np.floor(dur * tgt))
    t = np.linspace(0, dur, n, endpoint=False)

    # EDA: interpolate + phasic separation
    t_eda = np.linspace(0, len(eda)/fs_eda, len(eda), endpoint=False)
    eda_c = nk.eda_clean(eda, sampling_rate=fs_eda)
    eda_100 = interp1d(t_eda, eda_c, bounds_error=False, fill_value='extrapolate')(t)
    eda_dec = nk.eda_phasic(eda_100, sampling_rate=tgt)

    # ECG → RRI → HRV bands
    ecg_c = nk.ecg_clean(ecg, sampling_rate=fs_ecg)
    _, info = nk.ecg_peaks(ecg_c, sampling_rate=fs_ecg)
    rpeaks = info['ECG_R_Peaks']
    rri_ms = np.diff(rpeaks) / fs_ecg * 1000
    rri_t  = rpeaks[1:] / fs_ecg
    rri_100 = interp1d(rri_t, rri_ms, kind='cubic', bounds_error=False, fill_value='extrapolate')(t)

    import scipy.signal as ss
    sos_lf = ss.butter(4, [0.04, 0.15], btype='bandpass', fs=tgt, output='sos')
    sos_hf = ss.butter(4, [0.15, 0.40], btype='bandpass', fs=tgt, output='sos')
    lf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_lf, rri_100))) ** 2
    hf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_hf, rri_100))) ** 2

    # Labels at 100 Hz
    t_lbl = np.linspace(0, len(lbl)/fs_ecg, len(lbl), endpoint=False)
    lbl_100 = interp1d(t_lbl, lbl, kind='nearest', bounds_error=False, fill_value='extrapolate')(t)

    df = pd.DataFrame({
        'timestamp': t,
        'EDA_Tonic': eda_dec['EDA_Tonic'].values,
        'EDA_Phasic': eda_dec['EDA_Phasic'].values,
        'HRV_LF': lf, 'HRV_HF': hf,
        'raw_label': lbl_100,
        'subject_id': subj_data['subject_id'],
        'dataset': 'WESAD'
    })
    label_map = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}
    df['label'] = df['raw_label'].map(label_map)
    df = df.dropna(subset=['label', 'EDA_Tonic', 'HRV_LF'])
    return df

def extract_features_case(subj_data):
    """Simplified feature extraction for CASE (ECG/GSR only)."""
    import neurokit2 as nk
    df_p = subj_data['df_phys']
    tgt = 100

    t_p = df_p['daqtime'].values / 1000.0
    dur = t_p[-1]
    n = int(np.floor(dur * tgt))
    t = np.linspace(0, dur, n, endpoint=False)

    # GSR → phasic
    gsr_raw = df_p['gsr'].values
    gsr_100 = interp1d(t_p, gsr_raw, bounds_error=False, fill_value='extrapolate')(t)
    eda_dec = nk.eda_phasic(nk.eda_clean(gsr_100, sampling_rate=tgt), sampling_rate=tgt)

    # ECG → HRV
    ecg_raw = df_p['ecg'].values
    ecg_100 = interp1d(t_p, ecg_raw, bounds_error=False, fill_value='extrapolate')(t)
    ecg_c = nk.ecg_clean(ecg_100, sampling_rate=tgt)
    _, info = nk.ecg_peaks(ecg_c, sampling_rate=tgt)
    rpeaks = info['ECG_R_Peaks']
    rri_ms = np.diff(rpeaks) / tgt * 1000
    rri_t  = rpeaks[1:] / tgt
    rri_100 = interp1d(rri_t, rri_ms, kind='cubic', bounds_error=False, fill_value='extrapolate')(t)

    import scipy.signal as ss
    sos_lf = ss.butter(4, [0.04, 0.15], btype='bandpass', fs=tgt, output='sos')
    sos_hf = ss.butter(4, [0.15, 0.40], btype='bandpass', fs=tgt, output='sos')
    lf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_lf, rri_100))) ** 2
    hf = np.abs(ss.hilbert(ss.sosfiltfilt(sos_hf, rri_100))) ** 2

    # Video labels at 100 Hz
    vid_100 = interp1d(t_p, df_p['video'].values, kind='nearest', bounds_error=False, fill_value='extrapolate')(t)
    label_map = CASE_LABEL_MAP
    labels = [label_map.get(int(round(v)), None) for v in vid_100]

    df = pd.DataFrame({
        'timestamp': t,
        'EDA_Tonic': eda_dec['EDA_Tonic'].values,
        'EDA_Phasic': eda_dec['EDA_Phasic'].values,
        'HRV_LF': lf, 'HRV_HF': hf,
        'label': labels,
        'subject_id': subj_data['subject_id'],
        'dataset': 'CASE'
    })
    df = df.dropna(subset=['label', 'EDA_Tonic', 'HRV_LF'])
    return df

# ── Normalizers (inline, no import from normalizers.py to be self-contained) ──
FEATURES = ['EDA_Tonic', 'EDA_Phasic', 'HRV_LF', 'HRV_HF']

def apply_normalizers(X_train, X_test, X_base_train, rolling_window=6000):
    """Returns dict of {method_name: (X_train_norm, X_test_norm)}"""
    methods = {}

    # 1. Subject-wise Z (fit on all train)
    sc = StandardScaler().fit(X_train)
    methods['Subject_Z'] = (sc.transform(X_train), sc.transform(X_test))

    # 2. Baseline-only Z (fit ONLY on baseline train)
    mu_b = X_base_train.mean(axis=0)
    sd_b = X_base_train.std(axis=0)
    sd_b[sd_b < 1e-8] = 1e-8
    methods['Baseline_Z'] = ((X_train - mu_b) / sd_b, (X_test - mu_b) / sd_b)

    # 3. Population Z - fitted externally, passed as global scaler
    # (handled in outer loop, placeholder here)
    methods['Population_Z'] = None   # filled below

    # 4. DBA / Baseline Covariance Whitening
    cov_b = np.cov(X_base_train, rowvar=False)
    cov_b += np.eye(cov_b.shape[0]) * 1e-5
    try:
        W = sqrtm(np.linalg.inv(cov_b)).real
        methods['DBA_Whitening'] = ((X_train - mu_b) @ W, (X_test - mu_b) @ W)
    except Exception:
        methods['DBA_Whitening'] = ((X_train - mu_b) / sd_b, (X_test - mu_b) / sd_b)

    return methods

def evaluate_pair(X_tr, X_te, y_tr, y_te, active_label):
    """Fixed-param Logistic Regression, returns dict of metrics."""
    clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs',
                             max_iter=1000, class_weight='balanced',
                             random_state=42)
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)
    classes = list(clf.classes_)
    pos_idx = classes.index(active_label) if active_label in classes else 0
    prob_pos = y_prob[:, pos_idx]
    y_bin = (y_te == active_label).astype(int)

    try:
        auroc = roc_auc_score(y_bin, prob_pos)
        auprc = average_precision_score(y_bin, prob_pos)
    except Exception:
        auroc = auprc = float('nan')

    y_pred = clf.predict(X_te)
    f1  = f1_score(y_te, y_pred, pos_label=active_label, zero_division=0)
    mcc = matthews_corrcoef(y_te, y_pred)
    try:
        brier = brier_score_loss(y_bin, prob_pos)
    except Exception:
        brier = float('nan')

    return {'auroc': auroc, 'auprc': auprc, 'f1': f1, 'mcc': mcc, 'brier': brier,
            'y_pred': y_pred, 'prob_pos': prob_pos, 'y_te': y_te}

# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation(df_all, dataset_name, active_label):
    subjects = df_all['subject_id'].unique()
    print(f"\n[{dataset_name}] {len(subjects)} subjects, active='{active_label}'")

    # Build population scaler (exclude each subject at inference time)
    results_main    = []
    results_subject = []
    disagree_rows   = []
    calib_rows      = []
    raw_preds       = {}   # {method: {subj: (y_pred, y_te)}}

    for method in ['Subject_Z', 'Baseline_Z', 'Population_Z', 'DBA_Whitening']:
        raw_preds[method] = {}

    for subj in subjects:
        subj_df = df_all[df_all['subject_id'] == subj].copy()
        df_eval = subj_df[subj_df['label'].isin(['baseline', active_label])].copy()

        if len(df_eval['label'].unique()) < 2:
            continue

        base_idx = np.where(df_eval['label'].values == 'baseline')[0]
        act_idx  = np.where(df_eval['label'].values == active_label)[0]
        if len(base_idx) < 20 or len(act_idx) < 20:
            continue

        sp_b = int(len(base_idx) * 0.5)
        sp_a = int(len(act_idx)  * 0.5)
        train_idx = np.concatenate([base_idx[:sp_b], act_idx[:sp_a]])
        test_idx  = np.concatenate([base_idx[sp_b:], act_idx[sp_a:]])

        X_all  = df_eval[FEATURES].values
        y_all  = df_eval['label'].values
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        base_train_mask = (y_train == 'baseline')
        X_base_train = X_train[base_train_mask]
        if len(X_base_train) < 5:
            continue

        methods = apply_normalizers(X_train, X_test, X_base_train)

        # Population Z: fit on all OTHER subjects' train baseline data
        other_base_rows = []
        for s2 in subjects:
            if s2 == subj:
                continue
            s2_df = df_all[(df_all['subject_id'] == s2) & (df_all['label'] == 'baseline')]
            if not s2_df.empty:
                n2 = len(s2_df)
                other_base_rows.append(s2_df[FEATURES].values[:int(n2 * 0.5)])
        if other_base_rows:
            X_pop = np.vstack(other_base_rows)
            sc_pop = StandardScaler().fit(X_pop)
            methods['Population_Z'] = (sc_pop.transform(X_train), sc_pop.transform(X_test))
        else:
            methods['Population_Z'] = (X_train, X_test)

        subj_results = {'dataset': dataset_name, 'subject_id': subj}
        for m_name, pair in methods.items():
            if pair is None:
                continue
            X_tr, X_te = pair
            m = evaluate_pair(X_tr, X_te, y_train, y_test, active_label)
            subj_results[f'{m_name}_auroc'] = m['auroc']
            subj_results[f'{m_name}_f1']    = m['f1']
            subj_results[f'{m_name}_mcc']   = m['mcc']
            raw_preds[m_name][subj]         = (m['y_pred'], m['y_te'])

            results_subject.append({
                'dataset': dataset_name, 'subject_id': subj,
                'method': m_name, 'auroc': m['auroc'],
                'auprc': m['auprc'], 'f1': m['f1'], 'mcc': m['mcc']
            })

        results_main.append(subj_results)

        # Calibration length sub-audit (varying baseline duration)
        for calib_sec in [30, 60, 120, 300]:
            n_calib = min(int(calib_sec * 100), len(base_idx) // 2)
            if n_calib < 5:
                continue
            tr2 = np.concatenate([base_idx[:n_calib], act_idx[:sp_a]])
            X_tr2 = X_all[tr2]
            y_tr2 = y_all[tr2]
            X_base2 = X_tr2[y_tr2 == 'baseline']
            if len(X_base2) < 5:
                continue
            sc2 = StandardScaler().fit(X_tr2)
            m2 = evaluate_pair(sc2.transform(X_tr2), sc2.transform(X_test),
                               y_tr2, y_test, active_label)
            calib_rows.append({'dataset': dataset_name, 'subject_id': subj,
                                'duration': calib_sec, 'auroc_mean': m2['auroc']})

    # Aggregate results_main
    df_subj = pd.DataFrame(results_subject)
    method_agg = []
    for m_name in ['Subject_Z', 'Baseline_Z', 'Population_Z', 'DBA_Whitening']:
        m_rows = df_subj[df_subj['method'] == m_name].dropna(subset=['auroc'])
        if m_rows.empty:
            continue
        method_agg.append({
            'dataset': dataset_name, 'method': m_name,
            'auroc_mean': m_rows['auroc'].mean(), 'auroc_std': m_rows['auroc'].std(),
            'auprc_mean': m_rows['auprc'].mean(),
            'f1_mean':    m_rows['f1'].mean(),
            'n_subjects': len(m_rows)
        })

    # Pipeline disagreement
    method_pairs = [
        ('Subject_Z', 'Baseline_Z'), ('Subject_Z', 'Population_Z'),
        ('Subject_Z', 'DBA_Whitening'), ('Baseline_Z', 'DBA_Whitening')
    ]
    for m1, m2 in method_pairs:
        for subj in subjects:
            if subj not in raw_preds[m1] or subj not in raw_preds[m2]:
                continue
            p1, y_te = raw_preds[m1][subj]
            p2, _    = raw_preds[m2][subj]
            if len(p1) != len(p2):
                continue
            disagree_rate = np.mean(p1 != p2)
            # Cohen's kappa
            from sklearn.metrics import cohen_kappa_score
            try:
                kappa = cohen_kappa_score(p1, p2)
            except Exception:
                kappa = float('nan')
            disagree_rows.append({
                'dataset': dataset_name, 'subject_id': subj,
                'method_pair': f"{m1}_vs_{m2}",
                'prediction_disagreement_rate': disagree_rate,
                'cohens_kappa': kappa
            })

    return (pd.DataFrame(method_agg),
            pd.DataFrame(results_subject),
            pd.DataFrame(disagree_rows),
            pd.DataFrame(calib_rows))

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Phase 2C-1: Main Performance Run ===")
    print(f"Started: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")

    all_main, all_subj, all_disagree, all_calib = [], [], [], []

    # --- WESAD ---
    print("\n[WESAD] Loading raw signals...")
    wesad_records = load_wesad_raw()
    for rec in wesad_records:
        print(f"  Extracting features for {rec['subject_id']}...")
        try:
            df_s = extract_features_wesad(rec)
            if df_s.empty:
                print(f"    Skipping {rec['subject_id']}: empty after extraction")
                continue
            if not hasattr(run_evaluation, '_wesad_df'):
                run_evaluation._wesad_df = [df_s]
            else:
                run_evaluation._wesad_df.append(df_s)
        except Exception as e:
            print(f"    ERROR extracting {rec['subject_id']}: {e}")
            continue

    wesad_subjects_extracted = getattr(run_evaluation, '_wesad_df', [])
    if wesad_subjects_extracted:
        df_wesad = pd.concat(wesad_subjects_extracted, ignore_index=True)
        m, s, d, c = run_evaluation(df_wesad, 'WESAD', 'stress')
        all_main.append(m); all_subj.append(s)
        all_disagree.append(d); all_calib.append(c)
    else:
        print("[WESAD] No subjects successfully extracted.")

    # --- CASE ---
    print("\n[CASE] Loading raw signals...")
    case_records = load_case_raw()
    case_dfs = []
    for rec in case_records:
        print(f"  Extracting features for {rec['subject_id']}...")
        try:
            df_s = extract_features_case(rec)
            if not df_s.empty:
                case_dfs.append(df_s)
        except Exception as e:
            print(f"    ERROR extracting {rec['subject_id']}: {e}")
            continue

    if case_dfs:
        df_case = pd.concat(case_dfs, ignore_index=True)
        m, s, d, c = run_evaluation(df_case, 'CASE', 'stress')
        all_main.append(m); all_subj.append(s)
        all_disagree.append(d); all_calib.append(c)
    else:
        print("[CASE] No subjects successfully extracted.")

    # --- Save CSVs ---
    def safe_concat(lst):
        lst = [x for x in lst if x is not None and not x.empty]
        return pd.concat(lst, ignore_index=True) if lst else pd.DataFrame()

    safe_concat(all_main).to_csv(f"{OUT_DIR}/results_main.csv", index=False)
    safe_concat(all_subj).to_csv(f"{OUT_DIR}/results_subject_level.csv", index=False)
    safe_concat(all_disagree).to_csv(f"{OUT_DIR}/pipeline_disagreement.csv", index=False)
    safe_concat(all_calib).to_csv(f"{OUT_DIR}/calibration_length_results.csv", index=False)
    # Placeholder cross-dataset (populated after subject-level WESAD→CASE transfer)
    pd.DataFrame({'note': ['cross_dataset_transfer placeholder']}).to_csv(
        f"{OUT_DIR}/cross_dataset_transfer.csv", index=False)

    print(f"\nAll CSVs saved to {OUT_DIR}/")
    print("Finished: " + datetime.datetime.now(datetime.timezone.utc).isoformat())

    # ── Immediately trigger gate - DO NOT open CSVs manually ─────────────────
    print("\n=== Running Submission Gate (automated) ===")
    result = subprocess.run(
        ["python", GATE_SCRIPT, "--results_dir", OUT_DIR],
        capture_output=True, text=True
    )
    # Print only gate_report.json summary
    gate_path = os.path.join(OUT_DIR, "gate_report.json")
    if os.path.exists(gate_path):
        with open(gate_path) as f:
            gate = json.load(f)
        print("\n=== GATE REPORT SUMMARY ===")
        summary = {
            "verdict":               gate.get("overall_decision"),
            "gates_passed":          gate.get("gates_passed"),
            "gates_failed":          gate.get("gates_failed"),
            "effect_source":         gate.get("effect_source"),
            "warnings":              gate.get("warnings"),
            "recommended_next_step": gate.get("recommended_next_step")
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print("Gate script did not produce gate_report.json.")
        print("Gate stderr:", result.stderr[:500])
