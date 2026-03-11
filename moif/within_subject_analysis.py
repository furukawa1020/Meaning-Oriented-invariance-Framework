import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import numpy as np

print("Loading 100Hz WESAD data...")
df = pd.read_csv('wesad_100hz_instantaneous_raw.csv')
print(f"Loaded: {len(df):,} rows")

results = []

for subject in df['subject_id'].unique():
    sub_df = df[df['subject_id'] == subject].copy()
    
    # Normalize to individual baseline (Z-score within subject)
    mean_tonic = sub_df['EDA_Tonic'].mean()
    std_tonic  = sub_df['EDA_Tonic'].std()
    if std_tonic == 0:
        continue
    sub_df['tonic_z'] = (sub_df['EDA_Tonic'] - mean_tonic) / std_tonic
    
    # Select moments of high SCL within this individual
    high_scl = sub_df[sub_df['tonic_z'] > 0.5]
    
    if len(high_scl) < 500:
        continue  # too few samples
    
    # Label distribution in the high-SCL state FOR THIS SAME PERSON
    dist = high_scl['label'].value_counts(normalize=True)
    n_labels = len(dist)
    entropy = -sum(p * np.log2(p) for p in dist if p > 0)
    
    results.append({
        'subject': subject,
        'n_high_scl_samples': len(high_scl),
        'n_distinct_labels': n_labels,
        'label_entropy': round(entropy, 4),
        'label_distribution': dict(dist.round(3))
    })

results_df = pd.DataFrame(results).sort_values('label_entropy', ascending=False)
print("\n=== Within-Subject: Label distribution during High SCL state ===")
print(results_df[['subject','n_high_scl_samples','n_distinct_labels','label_entropy']].to_string())

# Show full distribution for the most ambiguous subject
top = results_df.iloc[0]
print(f"\n>>> Most ambiguous subject: {top['subject']}")
print(f"    Label spread during their own high-SCL moments:")
print(f"    {top['label_distribution']}")

results_df.to_csv('within_subject_invariance.csv', index=False)
print("\nSaved: within_subject_invariance.csv")
