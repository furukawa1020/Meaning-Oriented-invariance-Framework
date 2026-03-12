import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

df = pd.read_csv('wesad_100hz_instantaneous_augmented.csv')
s11 = df[df['subject_id'] == 'S11'].copy()
base = s11[s11['label'] == 'baseline']

for col in ['EDA_Tonic', 'EDA_Phasic', 'HRV_Inst_LF', 'HRV_Inst_HF']:
    s11[f'{col}_Z'] = (s11[col] - base[col].mean()) / base[col].std()

features = ['EDA_Tonic_Z', 'EDA_Phasic_Z', 'HRV_Inst_LF_Z', 'HRV_Inst_HF_Z']
s11_valid = s11.dropna(subset=['SAM_Valence']).copy()

# The previously found exact center
center_idx = np.array([-0.435, -0.23, -0.935, -1.057])
radius = 0.1

X = s11_valid[features].values
distances = np.linalg.norm(X - center_idx, axis=1)

# Points inside the microstate
mask_inside = distances <= radius
micro_state = s11_valid[mask_inside]

# Plotting: We will do a 3D scatter plot of Tonic, Phasic, LF, 
# and use a zoomed-in sphere to show how tight the cluster is, 
# colored by the True Subjective Emotion.

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# To provide context, plot some background points (outside the microstate) lightly
mask_bg = (distances > radius) & (distances < 1.0) # just a sample nearby
bg_sample = s11_valid[mask_bg].sample(n=min(sum(mask_bg), 1000), random_state=42)
ax.scatter(bg_sample['EDA_Tonic_Z'], bg_sample['EDA_Phasic_Z'], bg_sample['HRV_Inst_LF_Z'], 
           c='lightgray', alpha=0.1, s=10, label='Other States (Context)')

# Plot the microstate points, colored by SAM_Valence
val_2 = micro_state[micro_state['SAM_Valence'] == 2.0]
val_6 = micro_state[micro_state['SAM_Valence'] == 6.0]

ax.scatter(val_2['EDA_Tonic_Z'], val_2['EDA_Phasic_Z'], val_2['HRV_Inst_LF_Z'], 
           c='#e74c3c', s=60, alpha=0.8, edgecolor='black', label='Valence 2 (Negative)')

ax.scatter(val_6['EDA_Tonic_Z'], val_6['EDA_Phasic_Z'], val_6['HRV_Inst_LF_Z'], 
           c='#3498db', s=60, alpha=0.8, edgecolor='black', label='Valence 6 (Positive)')

# Draw a sphere around the center to visualize the radius
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = center_idx[0] + radius * np.cos(u)*np.sin(v)
y = center_idx[1] + radius * np.sin(u)*np.sin(v)
z = center_idx[2] + radius * np.cos(v)
ax.plot_wireframe(x, y, z, color='green', alpha=0.2, label=f'Radius: {radius} StdDev')

ax.set_xlabel('EDA Tonic (Z-Score)', fontsize=11)
ax.set_ylabel('EDA Phasic (Z-Score)', fontsize=11)
ax.set_zlabel('HRV LF (Z-Score)', fontsize=11)
ax.set_title('Fractal Invariance Breaking in a 4D Micro-State (S11)\n(Points physically indistinguishable within r=0.1 StdDev still split in emotion)', fontsize=14, fontweight='bold')

# Adjust view angle for best visibility
ax.view_init(elev=20, azim=45)

ax.legend(title='True Emotion', loc='upper right')

plt.tight_layout()
plt.savefig('micro_state_equivalence.png', dpi=300)
print(f"Saved micro_state_equivalence.png. Found {len(val_2)} Negative and {len(val_6)} Positive within r={radius}")

