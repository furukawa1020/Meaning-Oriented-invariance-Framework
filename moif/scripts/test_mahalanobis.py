"""
Test the Mahalanobis distance anchoring on WESAD Subject S11.
This script demonstrates how physiological state (Mahalanobis distance from baseline)
diverges from subjective emotion labels.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from moif.invariance.mahalanobis import calculate_mahalanobis_distance

print("Loading WESAD S11 data...")
# Read raw 100Hz WESAD data
df = pd.read_csv('results/wesad_100hz_instantaneous_raw.csv')
s11 = df[df['subject'] == 'S11'].copy()

# Ensure label mapping is standard
label_map = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}
s11['label_name'] = s11['label'].map(label_map)

# Define the physiological feature space
features = ['ECG_cwt_LF', 'ECG_cwt_HF', 'EDA_Phasic', 'EDA_Tonic']

print("Calculating Mahalanobis distance relative to Baseline...")
# Calculate Mahalanobis distance
s11['D_M'] = calculate_mahalanobis_distance(s11, features=features, baseline_label='baseline')

# Let's see the descriptive stats for Baseline vs Stress
b_dist = s11[s11['label_name'] == 'baseline']['D_M']
s_dist = s11[s11['label_name'] == 'stress']['D_M']

print("\n--- Mahalanobis Distance Stats ---")
print(f"Baseline Mean D_M: {b_dist.mean():.2f} (std: {b_dist.std():.2f})")
print(f"Stress   Mean D_M: {s_dist.mean():.2f} (std: {s_dist.std():.2f})")

# Visualize the distributions
plt.figure(figsize=(10, 6))
sns.kdeplot(b_dist.sample(min(10000, len(b_dist))), label='Baseline (True Neutral)', fill=True, color='blue', alpha=0.5)
sns.kdeplot(s_dist.sample(min(10000, len(s_dist))), label='Stress (Stressor Task)', fill=True, color='red', alpha=0.5)

plt.title('S11 Physiological Anchoring (Mahalanobis Distance from Baseline)', fontsize=14, fontweight='bold')
plt.xlabel('Mahalanobis Distance ($D_M$) - Overall Physiological Arousal')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('results/mahalanobis_S11.png', dpi=300)
print("\nSaved visualization to results/mahalanobis_S11.png")
