import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Loading 500MB dataset (S11 deep-dive)...")
df = pd.read_csv('wesad_100hz_instantaneous_raw.csv')

# Focus on Subject S11
sub_df = df[df['subject_id'] == 'S11'].copy()

# Baseline Anchoring: Use 'baseline' session data to define Z-score normalization
baseline_data = sub_df[sub_df['label'] == 'baseline']
if baseline_data.empty:
    print("Error: No baseline data found for S11.")
    exit(1)

mean_tonic = baseline_data['EDA_Tonic'].mean()
std_tonic  = baseline_data['EDA_Tonic'].std()
mean_lf = baseline_data['HRV_Inst_LF'].mean()
std_lf  = baseline_data['HRV_Inst_LF'].std()

sub_df['EDA_Tonic_Z'] = (sub_df['EDA_Tonic'] - mean_tonic) / std_tonic
sub_df['HRV_LF_Z'] = (sub_df['HRV_Inst_LF'] - mean_lf) / std_lf

# Filter out 'meditation' or other minor labels if needed for clarity
# Labels in WESAD: baseline, stress, amusement, meditation
sub_df = sub_df[sub_df['label'].isin(['baseline', 'stress', 'amusement'])]

sns.set_theme(style="white")
plt.figure(figsize=(10, 8))

# Use a jointplot to show both density and overlap
# Using scatter for clarity on the spread
g = sns.jointplot(
    data=sub_df, 
    x='EDA_Tonic_Z', 
    y='HRV_LF_Z', 
    hue='label', 
    palette={'baseline': '#2ecc71', 'stress': '#e74c3c', 'amusement': '#f39c12'},
    alpha=0.1,  # Many points, so keep transparency high
    s=5,
    kind='scatter'
)

# Highlight a specific "Invariance Breaking" region (e.g., High EDA, Low LF)
g.ax_joint.axvspan(0.5, 3.0, color='gray', alpha=0.1, label='High Physiological Arousal')
g.fig.suptitle('Subject S11: Overlap of Physiological States across Emotions\n(X: EDA Tonic Z, Y: HRV LF Z - Both anchored to S11 Baseline)', y=1.03, fontsize=14, fontweight='bold')

plt.savefig('S11_physiological_overlap.png', dpi=300)
print("Saved: S11_physiological_overlap.png")

# Calculate overlap percentage in the 2D space? (Bonus)
# Just a simple count for now in a specific region
high_arousal = sub_df[(sub_df['EDA_Tonic_Z'] > 0.5) & (sub_df['HRV_LF_Z'] > 0.5)]
print("\n=== Statistics for S11 in 'High Arousal' Region (EDA Z > 0.5, LF Z > 0.5) ===")
print(high_arousal['label'].value_counts(normalize=True))
print(f"Total points in this region: {len(high_arousal)}")
