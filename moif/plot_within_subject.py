import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df_raw = pd.read_csv('wesad_100hz_instantaneous_raw.csv')

# Get full results
results = []
for subject in df_raw['subject_id'].unique():
    sub = df_raw[df_raw['subject_id'] == subject].copy()
    mean_t = sub['EDA_Tonic'].mean()
    std_t  = sub['EDA_Tonic'].std()
    if std_t == 0: continue
    sub['tonic_z'] = (sub['EDA_Tonic'] - mean_t) / std_t
    high_scl = sub[sub['tonic_z'] > 0.5]
    if len(high_scl) < 500: continue
    dist = high_scl['label'].value_counts(normalize=True)
    results.append({'subject': subject, 'dist': dist, 'n': len(high_scl)})

# Sort by entropy (most ambiguous first)
def entropy(d):
    return -sum(p * np.log2(p) for p in d if p > 0)
results_sorted = sorted(results, key=lambda x: entropy(x['dist']), reverse=True)

# Plot: stacked bar for each subject showing label mix during high-SCL
fig, ax = plt.subplots(figsize=(14, 6))
colors = {'baseline': '#2ecc71', 'stress': '#e74c3c', 'amusement': '#f39c12', 'meditation': '#9b59b6'}
subjects = [r['subject'] for r in results_sorted]
bottom = np.zeros(len(subjects))

all_labels = set()
for r in results_sorted:
    all_labels.update(r['dist'].index)

for label in ['baseline', 'stress', 'amusement', 'meditation']:
    if label not in all_labels: continue
    vals = [r['dist'].get(label, 0) for r in results_sorted]
    ax.bar(subjects, vals, bottom=bottom, label=label, color=colors.get(label, '#888'), edgecolor='white', linewidth=0.5)
    bottom += np.array(vals)

ax.set_title('Within-Subject: Task Label Distribution During HIGH SCL State\n(Same person, same physiological threshold — which task were they in?)', fontsize=13, fontweight='bold')
ax.set_ylabel('Proportion of High-SCL Moments', fontsize=11)
ax.set_xlabel('Subject (sorted by label ambiguity, left = most ambiguous)', fontsize=11)
ax.legend(title='Session Label', loc='upper right')
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)

# Annotate n for each subject
for i, r in enumerate(results_sorted):
    ax.text(i, -0.06, f"n={r['n']//1000}k", ha='center', fontsize=8, color='gray')

plt.tight_layout()
plt.savefig('within_subject_label_ambiguity.png', dpi=300)
print("Saved: within_subject_label_ambiguity.png")
