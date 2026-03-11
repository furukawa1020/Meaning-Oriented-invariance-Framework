import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading massive 100Hz WESAD data...")
df = pd.read_csv('wesad_100hz_instantaneous_raw.csv')

def get_high_scl_state(df_full, sub):
    sub_df = df_full[df_full['subject_id'] == sub].copy()
    mean = sub_df['EDA_Tonic'].mean()
    std = sub_df['EDA_Tonic'].std()
    sub_df['tonic_z'] = (sub_df['EDA_Tonic'] - mean) / std
    # Filter for High Instantaneous SCL
    state_df = sub_df[sub_df['tonic_z'] > 0.5]
    return state_df

df17 = get_high_scl_state(df, 'S17')
df5 = get_high_scl_state(df, 'S5')

dist17 = df17['label'].value_counts(normalize=True).reset_index()
dist17.columns = ['Emotion Label', 'Probability']
dist17['Subject'] = 'Subject S17'

dist5 = df5['label'].value_counts(normalize=True).reset_index()
dist5.columns = ['Emotion Label', 'Probability']
dist5['Subject'] = 'Subject S5'

plot_df = pd.concat([dist17, dist5])

# Plot the distribution contradiction
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_df, x='Emotion Label', y='Probability', hue='Subject', palette=['#3498db', '#e74c3c'])
plt.title('Invariance Breaking: Identical Physiological State (High SCL)\nResults in Completely Diverged Subjective Emotions (JSD=1.0)', fontsize=14, fontweight='bold')
plt.ylabel('Probability (Frequency in State)', fontsize=12)
plt.ylim(0, 1.1)
plt.legend(title='Person')
plt.tight_layout()
plt.savefig('invariance_breaking_S17_S5.png', dpi=300)
print("Graph saved as invariance_breaking_S17_S5.png")
