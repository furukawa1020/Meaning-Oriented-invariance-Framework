import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load a small slice of the massive 100Hz data (e.g., 30 seconds of S10's stress phase)
df = pd.read_csv('wesad_100hz_instantaneous_raw.csv')
slice_df = df[(df['subject_id'] == 'S10') & (df['label'] == 'stress')].iloc[5000:8000] # 30 seconds

sns.set_theme(style="whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: EDA Deconvolution (Bateman / cvxEDA)
ax1.plot(slice_df['timestamp'], slice_df['EDA_Phasic'], color='#e74c3c', label='Inst. Phasic (Sweat Bursts)', linewidth=2)
ax1.plot(slice_df['timestamp'], slice_df['EDA_Tonic'], color='#3498db', label='Inst. Tonic (Background Level)', linewidth=2)
ax1.set_title('High-Resolution (100Hz) EDA Deconvolution (Bateman Model)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.legend(loc='upper right')

# Plot 2: HRV Instantaneous Power (CWT / Hilbert)
ax2.plot(slice_df['timestamp'], slice_df['HRV_Inst_LF'], color='#f39c12', label='Inst. LF (Sympathetic)', linewidth=2)
ax2.plot(slice_df['timestamp'], slice_df['HRV_Inst_HF'], color='#2ecc71', label='Inst. HF (Parasympathetic/Vagal)', linewidth=2)
ax2.set_title('Instantaneous HRV Power (Analytic Hilbert Transform)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Power', fontsize=12)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('visualize_100hz_signals.png', dpi=300)
print("Graph saved as visualize_100hz_signals.png")
