import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Loading augmented 100Hz dataset with subjective scores...")
df = pd.read_csv('wesad_100hz_instantaneous_augmented.csv')

results = []
for sub in df['subject_id'].unique():
    sub_df = df[df['subject_id'] == sub].copy()
    base = sub_df[sub_df['label'] == 'baseline']
    if base.empty: 
        continue
        
    mean_t = base['EDA_Tonic'].mean()
    std_t = base['EDA_Tonic'].std()
    if std_t == 0: continue
    
    sub_df['EDA_Tonic_Z'] = (sub_df['EDA_Tonic'] - mean_t) / std_t
    
    # Filter for High SCL and valid SAM scores
    high_scl = sub_df[(sub_df['EDA_Tonic_Z'] > 0.5) & sub_df['SAM_Valence'].notnull()]
    if len(high_scl) < 500: 
        continue
        
    counts = high_scl.groupby(['SAM_Valence', 'SAM_Arousal'])['timestamp'].count()
    dist = counts / counts.sum()
    
    entropy_val = -sum(p * np.log2(p) for p in dist if p > 0)
    
    results.append({
        'subject': sub,
        'dist': dist,
        'entropy': entropy_val,
        'n': len(high_scl)
    })

# Sort by entropy (most ambiguous first)
results_sorted = sorted(results, key=lambda x: x['entropy'], reverse=True)

# Plotting
fig, ax = plt.subplots(figsize=(12, 7))

all_states = set()
for r in results_sorted:
    all_states.update(r['dist'].index)
states_list = sorted(list(all_states))

import matplotlib.cm as cm
cmap = plt.get_cmap('tab20')
colors = {state: cmap(i % 20) for i, state in enumerate(states_list)}

subjects_labels = [r['subject'] for r in results_sorted]
bottom = np.zeros(len(subjects_labels))

for state in states_list:
    vals = [r['dist'].get(state, 0) for r in results_sorted]
    # some states don't have any values in the plot if they are 0 everywhere, but we gathered from dists so it's fine
    if sum(vals) == 0: continue
    label_str = f"Valence {int(state[0])} / Arousal {int(state[1])}"
    ax.bar(subjects_labels, vals, bottom=bottom, label=label_str, color=colors[state], edgecolor='white', linewidth=0.5)
    bottom += np.array(vals)

ax.set_title('True Subjective Emotion Split Across ALL Subjects\n(Identical Physiological State -> Completely Different True Subjective Meanings)', fontsize=14, fontweight='bold')
ax.set_ylabel('Proportion of High SCL Moments', fontsize=12)
ax.set_xlabel('Subject (sorted by subjective ambiguity/entropy)', fontsize=12)

# Place legend outside
ax.legend(title='True SAM Score (Valence/Arousal)', bbox_to_anchor=(1.01, 1), loc='upper left')

# Annotate n sizes
for i, r in enumerate(results_sorted):
    ax.text(i, -0.05, f"n={r['n']//1000}k", ha='center', fontsize=9, color='gray')

plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig('all_subjects_true_emotion_split.png', dpi=300)
print(f"Computed for {len(results_sorted)} subjects.")
print("Saved all_subjects_true_emotion_split.png")

# Also print text summary
print("\n=== Summary of Invariance Breaking ===")
for r in results_sorted:
    print(f"{r['subject']} (entropy {r['entropy']:.2f}):")
    for state, p in r['dist'].items():
        print(f"  - Val {state[0]:.0f} / Aro {state[1]:.0f}: {p*100:.1f}%")
