import pandas as pd
import numpy as np

# A quick scratch script to test how to define "identical" mathematically
df = pd.read_csv('wesad_100hz_instantaneous_augmented.csv')
s11 = df[df['subject_id'] == 'S11'].copy()
base = s11[s11['label'] == 'baseline']

# Normalize 4 distinct features
for col in ['EDA_Tonic', 'EDA_Phasic', 'HRV_Inst_LF', 'HRV_Inst_HF']:
    s11[f'{col}_Z'] = (s11[col] - base[col].mean()) / base[col].std()

features = ['EDA_Tonic_Z', 'EDA_Phasic_Z', 'HRV_Inst_LF_Z', 'HRV_Inst_HF_Z']

# Drop NaN subjective scores
s11_valid = s11.dropna(subset=['SAM_Valence'])

# Find the most dense point where Valence is split (brute force search for a hyper-specific coordinate)
print("Finding an exact 4D coordinate where emotions split...")
from sklearn.neighbors import KDTree

X = s11_valid[features].values
tree = KDTree(X)

# We want a very small radius, e.g., r = 0.1 standard deviations in all 4 dimensions
# This is physically indistinguishable
radius = 0.1

best_split = None
max_entropy = -1
best_center = None
best_n = 0

# Sample some points to test
np.random.seed(42)
b_idx = np.random.choice(len(X), size=2000, replace=False)

for i in b_idx:
    center = X[i].reshape(1, -1)
    ind = tree.query_radius(center, r=radius)[0]
    if len(ind) < 100: # Need enough points to be statistically relevant
        continue
    
    subset = s11_valid.iloc[ind]
    counts = subset.groupby(['SAM_Valence', 'SAM_Arousal'])['timestamp'].count()
    dist = counts / counts.sum()
    
    if len(dist) > 1:
        entropy = -sum(p * np.log2(p) for p in dist if p > 0)
        if entropy > max_entropy:
            max_entropy = entropy
            best_split = counts
            best_center = center
            best_n = len(ind)

print(f"\nMost ambiguous exact state found (Radius = {radius} standard deviations in 4D space!)")
print(f"Center Coordinate (Tonic_Z, Phasic_Z, LF_Z, HF_Z): \n{np.round(best_center[0], 3)}")
print(f"Number of moments in this exact micro-state: {best_n}")
print("\nTrue Subjective Split in this micro-state:")
if best_split is not None:
    print(best_split)
else:
    print("No split found at this tight radius.")
