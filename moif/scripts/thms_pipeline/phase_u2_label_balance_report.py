import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import neurokit2 as nk

CASE_DIR = Path("C:/Projects/Meaning-Oriented invariance Framework/moif/data/case")
OUT_DIR = "phase_u_results"
os.makedirs(OUT_DIR, exist_ok=True)

def get_subject_labels(sub_id):
    a_path = CASE_DIR / 'data' / 'interpolated' / 'annotations' / f"{sub_id}.csv"
    p_path = CASE_DIR / 'data' / 'interpolated' / 'physiological' / f"{sub_id}.csv"
    if not a_path.exists(): return None
    df_a = pd.read_csv(a_path)
    df_p = pd.read_csv(p_path)
    
    # Standardize scale 1-9 to -1 to 1
    def rescale(v): return (v - 5.0) / 4.0
    
    # 1s windows
    ws = 20 # Annotations are 20Hz
    nw = len(df_a) // ws
    def win(a): return a[:nw*ws].reshape(nw, ws).mean(axis=1)
    
    arousal = win(rescale(df_a['arousal'].values))
    valence = win(rescale(df_a['valence'].values))
    video = win(df_a['video'].values).astype(int)
    
    return pd.DataFrame({
        'sub_id': sub_id,
        'Arousal': arousal,
        'Valence': valence,
        'video_id': video
    })

def run_balance_report():
    all_counts = []
    for i in range(1, 31):
        sid = f"sub_{i}"
        df = get_subject_labels(sid)
        if df is None: continue
        
        # Calculate thresholds (Subject-specific top/bottom 30%)
        # Exclude baseline (video=0)
        df_v = df[df['video_id'] != 0]
        if len(df_v) == 0: continue
        
        a_high_thresh = np.percentile(df_v['Arousal'], 70)
        a_low_thresh = np.percentile(df_v['Arousal'], 30)
        v_pos_thresh = np.percentile(df_v['Valence'], 70)
        v_neg_thresh = np.percentile(df_v['Valence'], 30)
        
        counts = {
            'sub_id': sid,
            'total_video_windows': len(df_v),
            'HA': len(df_v[df_v['Arousal'] > a_high_thresh]),
            'LA': len(df_v[df_v['Arousal'] < a_low_thresh]),
            'HA_Pos': len(df_v[(df_v['Arousal'] > a_high_thresh) & (df_v['Valence'] > v_pos_thresh)]),
            'HA_Neg': len(df_v[(df_v['Arousal'] > a_high_thresh) & (df_v['Valence'] < v_neg_thresh)])
        }
        all_counts.append(counts)
    
    report_df = pd.DataFrame(all_counts)
    report_df.to_csv(os.path.join(OUT_DIR, "phase_u2_label_balance_report.csv"), index=False)
    print("Phase U2 Label Balance Report generated.")
    
    # Check Termination Criteria
    valid_subs = report_df[(report_df['HA_Pos'] >= 20) & (report_df['HA_Neg'] >= 20)]
    print(f"Subjects with sufficient HA-Pos/Neg samples (N>=20): {len(valid_subs)}")
    if len(valid_subs) < 10:
        print("WARNING: Insufficient subjects for Task C. Phase U2 may be untestable.")

if __name__ == "__main__":
    run_balance_report()
