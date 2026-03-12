import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KDTree

print("Loading augmented dataset...")
df = pd.read_csv('wesad_100hz_instantaneous_augmented.csv')

features = ['EDA_Tonic_Z', 'EDA_Phasic_Z', 'HRV_Inst_LF_Z', 'HRV_Inst_HF_Z']
radius = 0.1 # Ultra-strict mathematical equivalence
min_samples = 50 # Minimum physiological moments needed in a micro-state to calculate reliable emotion split

results = []
all_subject_entropies = []

for sub in df['subject_id'].unique():
    sub_df = df[df['subject_id'] == sub].copy()
    base = sub_df[sub_df['label'] == 'baseline']
    if base.empty: continue
        
    for col in ['EDA_Tonic', 'EDA_Phasic', 'HRV_Inst_LF', 'HRV_Inst_HF']:
        if base[col].std() == 0: continue
        sub_df[f'{col}_Z'] = (sub_df[col] - base[col].mean()) / base[col].std()

    sub_valid = sub_df.dropna(subset=['SAM_Valence']).copy()
    if len(sub_valid) < 1000: continue
    
    X = sub_valid[features].values
    tree = KDTree(X)
    
    # Query random 5000 points per subject to represent their entire physiological space
    sample_size = min(len(X), 5000)
    np.random.seed(42)
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    
    ind_list = tree.query_radius(X_sample, r=radius)
    
    entropies = []
    for neighbor_indices in ind_list:
        if len(neighbor_indices) >= min_samples:
            subset = sub_valid.iloc[neighbor_indices]
            counts = subset.groupby(['SAM_Valence', 'SAM_Arousal'])['timestamp'].count()
            dist = counts / counts.sum()
            if len(dist) > 0:
                entropy = -sum(p * np.log2(p) for p in dist if p > 0)
                entropies.append(entropy)
                all_subject_entropies.append({'Subject': sub, 'Entropy': entropy})
                
    if entropies:
        ambiguous_pct = np.mean(np.array(entropies) > 0.5) * 100
        results.append({
            'Subject': sub,
            'Average Microstate Entropy': np.mean(entropies),
            'Max Entropy': np.max(entropies),
            'Percent Ambiguous (Entropy>0.5)': ambiguous_pct,
            'Microstates Analyzed': len(entropies)
        })
        print(f"Processed {sub} | Avg Ent: {np.mean(entropies):.2f} | Ambiguous: {ambiguous_pct:.1f}% | Microstates: {len(entropies)}")

res_df = pd.DataFrame(results).sort_values('Percent Ambiguous (Entropy>0.5)', ascending=False)
print("\n=== Universal Microstate Analysis (r=0.1 StdDev in 4D) ===")
print(res_df.to_string())

# Plotting the percentage of ambiguous space for all subjects
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=res_df, x='Subject', y='Percent Ambiguous (Entropy>0.5)', color='#2c3e50')

plt.title('Universality of Meaning Fluctuation (All Subjects, Entire Space)\nPercentage of EXACT physiological states (r=0.1 StdDev) that have split emotions', fontsize=14, fontweight='bold')
plt.ylabel('Proportion of Physiologically\n"Identical" States that are Ambiguous (%)', fontsize=12)
plt.xlabel('Subjects (sorted by ambiguity)', fontsize=12)

# Annotate percentages
for i, p in enumerate(ax.patches):
    ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 10), textcoords='offset points')

plt.ylim(0, 105)
plt.tight_layout()
plt.savefig('universal_microstate_entropy.png', dpi=300)
print("\nSaved universal_microstate_entropy.png")
