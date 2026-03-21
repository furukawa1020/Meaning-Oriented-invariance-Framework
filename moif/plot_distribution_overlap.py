import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

print("Loading augmented dataset...")
df = pd.read_csv('wesad_100hz_instantaneous_augmented.csv')

features = ['EDA_Tonic', 'EDA_Phasic', 'HRV_Inst_LF', 'HRV_Inst_HF']
z_features = [f'{col}_Z' for col in features]

# 1. Plot S11 2D KDE Overlap
sub = 'S11'
s11 = df[df['subject_id'] == sub].copy()
base_s11 = s11[s11['label'] == 'baseline'].copy()
stress_s11 = s11[s11['label'] == 'stress'].copy()

# Z-score based on baseline
for col in features:
    mean_val = base_s11[col].mean()
    std_val = base_s11[col].std()
    base_s11.loc[:, f'{col}_Z'] = (base_s11[col] - mean_val) / std_val
    stress_s11.loc[:, f'{col}_Z'] = (stress_s11[col] - mean_val) / std_val

# Remove outliers for plot stability (> 3 std dev)
base_clean = base_s11[(base_s11[z_features].abs() < 5).all(axis=1)]
stress_clean = stress_s11[(stress_s11[z_features].abs() < 5).all(axis=1)]

plt.figure(figsize=(10, 8))
# Sample data for KDE to avoid memory crash
b_sample = base_clean.sample(min(5000, len(base_clean)), random_state=42)
s_sample = stress_clean.sample(min(5000, len(stress_clean)), random_state=42)

sns.kdeplot(x=b_sample['EDA_Tonic_Z'], y=b_sample['HRV_Inst_LF_Z'], cmap="Blues", fill=True, alpha=0.5)
sns.kdeplot(x=s_sample['EDA_Tonic_Z'], y=s_sample['HRV_Inst_LF_Z'], cmap="Reds", fill=True, alpha=0.5)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.5, label='Baseline Block (True Meaning: Valence 6, Arousal 4)'),
    Patch(facecolor='red', alpha=0.5, label='Stress Block (True Meaning: Valence 2, Arousal 6)')
]
plt.legend(handles=legend_elements, loc='upper right')
plt.title('Distribution-Level Invariance Breaking (S11)\nThe physiological spaces of Baseline and Stress overlap almost entirely,\nyet the subjective meanings attached to these blocks are opposites.', fontsize=13, fontweight='bold')
plt.xlabel('EDA Tonic (Z-Score from Baseline)')
plt.ylabel('HRV LF (Z-Score from Baseline)')
plt.tight_layout()
plt.savefig('distribution_overlap_S11.png', dpi=300)
print("Saved 2D Distribution Overlap for S11.")


# 2. Universal Overlap Calculation (4D Radius = 1.0 StdDev)
results = []
for sub_id in df['subject_id'].unique():
    sub_df = df[df['subject_id'] == sub_id].copy()
    b = sub_df[sub_df['label'] == 'baseline'].copy()
    s = sub_df[sub_df['label'] == 'stress'].copy()
    if b.empty or s.empty: continue
        
    for col in features:
        mean_val = b[col].mean()
        std_val = b[col].std()
        if std_val == 0: continue
        b.loc[:, f'{col}_Z'] = (b[col] - mean_val) / std_val
        s.loc[:, f'{col}_Z'] = (s[col] - mean_val) / std_val
        
    if f'{features[0]}_Z' not in b.columns: continue
    
    # How much of Stress physiology is INSIDE the Baseline physiology?
    # Fit NN on Baseline points
    nn = NearestNeighbors(radius=1.0)
    b_vals = b[z_features].values
    if len(b_vals) > 10000:
        np.random.seed(42)
        b_vals = b_vals[np.random.choice(len(b_vals), 10000, replace=False)]
    nn.fit(b_vals)

    s_vals = s[z_features].values
    if len(s_vals) > 10000:
        np.random.seed(42)
        s_vals = s_vals[np.random.choice(len(s_vals), 10000, replace=False)]
        
    ind = nn.radius_neighbors(s_vals, return_distance=False)
    overlap_pct = np.mean([len(n) > 0 for n in ind]) * 100
    
    results.append({
        'Subject': sub_id,
        'Overlap (%)': overlap_pct
    })

res_df = pd.DataFrame(results).sort_values('Overlap (%)', ascending=False)
print("\n=== Universal Distribution Overlap (Radius=1.0 StdDev in 4D) ===")
print(res_df.to_string())

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=res_df, x='Subject', y='Overlap (%)', color='#8e44ad')
plt.title('Universality of State Overlap without Data Interpolation\n% of "Stress" Physiology that is statistically identical (r<1.0 StdDev) to "Baseline" Physiology', fontsize=14, fontweight='bold')
plt.ylabel('Overlap Percentage (%)', fontsize=12)
plt.axhline(50, color='red', linestyle='--', label='50% Overlap')
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=9, color='black', xytext=(0, 10), textcoords='offset points')
plt.legend()
plt.tight_layout()
plt.savefig('universal_distribution_overlap.png', dpi=300)
print("Saved universal distribution overlap plot.")

