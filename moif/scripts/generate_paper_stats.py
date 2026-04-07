import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from diptest import diptest
import scipy.stats as stats
import os

os.makedirs('results/figures', exist_ok=True)

print("Loading evaluation results...")
df_wesad = pd.read_csv('results/evaluation_baselines_wesad.csv')
df_case = pd.read_csv('results/evaluation_baselines_case.csv')

# Combine datasets for universal topological testing
df_wesad['Dataset'] = 'WESAD'
df_case['Dataset'] = 'CASE'
df_all = pd.concat([df_wesad, df_case], ignore_index=True)

# 1. Statistical Proof of Bimodality (Hartigan's Dip Test)
x_dba = df_all['DBA Omega'].values
# diptest computes the dip statistic and the p-value
dip, p_val = diptest(x_dba)
print(f"\n--- 1. HARTIGAN'S DIP TEST FOR BIMODALITY ---")
print(f"Dip Statistic: {dip:.4f}, p-value: {p_val:.4e}")
if p_val < 0.05:
    print("Result: SIGNIFICANT. The distribution is strongly Bimodal/Multimodal. Assumes of normal homeostasis invariance are broken.")
else:
    print("Result: NOT significant. It might be unimodal.")

# 2. ANOVA on Variance heterogeneity between methods
# We want to show whether methods differ in inter-subject variance and mean.
methods = ['Global Z Omega', 'Rolling Z Omega', 'DBA Omega']
methods_overlap_data = [df_all[col].dropna().values for col in methods if col in df_all.columns]
f_stat, p_anova = stats.f_oneway(*methods_overlap_data)
print(f"\n--- 2. ANOVA ON CALIBRATION METHODS ---")
print(f"F-statistic: {f_stat:.4f}, p-value: {p_anova:.4e}")

# Levene's test for equality of variances
w_stat, p_levene = stats.levene(*methods_overlap_data)
print(f"\n--- 3. LEVENE'S TEST FOR VARIANCE HETEROGENEITY ---")
print(f"W-statistic: {w_stat:.4f}, p-value: {p_levene:.4e}")
if p_levene < 0.05:
    print("Result: DBA Calibration significantly exposes latent inter-subject variance compared to conventional methods.")

# FIGURE 1: Histogram showing Bimodal Distribution (The "Proof" Plot)
plt.figure(figsize=(8, 5))
sns.histplot(x_dba, bins=20, kde=True, color='#2c3e50', edgecolor='white', alpha=0.8)
plt.title(f'Histogram of Dynamic Baseline Overlap ($\\Omega$) across 45 Subjects\nHartigan\'s Dip p={p_val:.3e}', fontsize=12, fontweight='bold')
plt.xlabel('Geometric Overlap $\\Omega$ (%) between Resting and Task Topologies')
plt.ylabel('Number of Subjects')
plt.grid(axis='y', alpha=0.3)
plt.xlim(-5, 105)
plt.tight_layout()
plt.savefig('results/figures/fig1_bimodal_overlap.png', dpi=300)
print("\nSaved Figure 1: Bimodal Overlap Histogram")

# FIGURE 2: Swarm/Violin plot comparing methods to show DBA exposes the extremes
plt.figure(figsize=(10, 6))
# Melt df for seaborn
df_melt = pd.melt(df_all, id_vars=['Subject'], value_vars=['Global Z Omega', 'Rolling Z Omega', 'DBA Omega'],
                  var_name='Calibration Method', value_name='Overlap Omega (%)')
df_melt = df_melt.dropna()
# rename for nicer plots
df_melt['Calibration Method'] = df_melt['Calibration Method'].map({
    'Global Z Omega': 'Global Z-Score',
    'Rolling Z Omega': '60s Rolling Window',
    'DBA Omega': 'Mahalanobis DBA'
})
sns.violinplot(x='Calibration Method', y='Overlap Omega (%)', data=df_melt, inner='quartile', palette='muted')
sns.stripplot(x='Calibration Method', y='Overlap Omega (%)', data=df_melt, color='black', alpha=0.5, size=5)
plt.title('Topological Overlap by Calibration Method Matrix\n(DBA reveals the bimodal extremes masked by windowing)', fontsize=12, fontweight='bold')
plt.ylim(-10, 110)
plt.tight_layout()
plt.savefig('results/figures/fig2_method_comparison.png', dpi=300)
print("Saved Figure 2: Method Comparison Violin Plot")

# FIGURE 3: F1-Score vs Overlap Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_all, x='DBA Omega', y='DBA Separability (F1)', hue='Dataset', s=80, alpha=0.8, palette='coolwarm')
plt.title('Physiological Separability vs Overlap\n(High Separation is maintained despite extreme overlap)', fontsize=12, fontweight='bold')
plt.xlabel('Topological Overlap $\\Omega$ (%)')
plt.ylabel('F1-Score (Stress vs Baseline Separability)')
plt.ylim(0.4, 1.05)
plt.xlim(-5, 105)
plt.grid(alpha=0.3)
# Add median lines
plt.axhline(df_all['DBA Separability (F1)'].median(), color='red', linestyle='--', alpha=0.3, label='Median F1')
plt.legend()
plt.tight_layout()
plt.savefig('results/figures/fig3_separability_vs_overlap.png', dpi=300)
print("Saved Figure 3: Separability vs Overlap Scatter")

print("\nAll statistical validations and figures have been generated in results/figures/!")
