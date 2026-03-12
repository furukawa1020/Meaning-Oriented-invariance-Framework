import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_quest = pd.read_csv('subjective_scores.csv')

label_map = {
    'baseline': 'Base',
    'stress': 'TSST',
    'amusement': 'Fun',
    'meditation': 'Medi 1'
}

df = pd.read_csv('wesad_100hz_instantaneous_raw.csv')
df['quest_cond'] = df['label'].map(label_map)
df = df.merge(df_quest, how='left', left_on=['subject_id', 'quest_cond'], right_on=['subject_id', 'condition'])

sub11 = df[df['subject_id'] == 'S11'].copy()
base = sub11[sub11['label'] == 'baseline']
sub11['EDA_Tonic_Z'] = (sub11['EDA_Tonic'] - base['EDA_Tonic'].mean()) / base['EDA_Tonic'].std()

high_scl = sub11[(sub11['EDA_Tonic_Z'] > 0.5) & sub11['SAM_Valence'].notnull()].copy()

# We want to plot a bar chart showing the split
counts = high_scl.groupby(['SAM_Valence', 'SAM_Arousal'])['timestamp'].count().reset_index()
counts['percentage'] = counts['timestamp'] / counts['timestamp'].sum() * 100

# Format the labels for the chart
counts['Emotion State'] = counts.apply(lambda x: f"Valence {int(x['SAM_Valence'])} / Arousal {int(x['SAM_Arousal'])}", axis=1)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 6))

ax = sns.barplot(data=counts, x='Emotion State', y='percentage', palette=['#3498db', '#e74c3c'])
plt.title('True Subjective Emotion during High Physiological Arousal (S11)\n(Identical Physiology -> Split True Subjective Questionnaires)', fontsize=14, fontweight='bold')
plt.ylabel('Proportion of Time in High SCL State (%)', fontsize=12)
plt.xlabel('Self-Reported Questionnaire Scores (SAM)', fontsize=12)

# Add percentages on top of bars
for i, p in enumerate(ax.patches):
    ax.annotate(f"{counts['percentage'].iloc[i]:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), textcoords='offset points')

plt.ylim(0, 100)
plt.tight_layout()
plt.savefig('S11_true_emotion_split.png', dpi=300)
print("Saved: S11_true_emotion_split.png")

df.to_csv('wesad_100hz_instantaneous_augmented.csv', index=False)
print("Saved augmented dataset.")
