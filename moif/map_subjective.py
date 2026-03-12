import pandas as pd
import numpy as np

# Load the subjective scores
df_quest = pd.read_csv('subjective_scores.csv')

# The WESAD task labels in our raw data are 'baseline', 'stress', 'amusement', 'meditation'
# But in the quest.csv they are 'Base', 'TSST', 'Fun', 'Medi 1', 'Medi 2'
label_map = {
    'baseline': 'Base',
    'stress': 'TSST',
    'amusement': 'Fun',
    'meditation': 'Medi 1' # Assuming Medi 1 for simplicity of this verification
}

# Create a dictionary for fast lookup: (subject, wesad_label) -> SAM_Arousal / SAM_Valence
quest_dict = {}
for _, row in df_quest.iterrows():
    sub = row['subject_id']
    cond = row['condition']
    quest_dict[(sub, cond)] = {
        'SAM_Arousal': row['SAM_Arousal'],
        'SAM_Valence': row['SAM_Valence']
    }

print("Loading 100Hz WESAD data...")
df = pd.read_csv('wesad_100hz_instantaneous_raw.csv')

def get_sam(row, key):
    sub = row['subject_id']
    label = row['label']
    quest_cond = label_map.get(label, None)
    if quest_cond and (sub, quest_cond) in quest_dict:
        return quest_dict[(sub, quest_cond)].get(key, np.nan)
    return np.nan

print("Mapping SAM Arousal and Valence to 100Hz data...")
# This mapping is slow using apply, let's use a merge
df['quest_cond'] = df['label'].map(label_map)
df = df.merge(df_quest, how='left', left_on=['subject_id', 'quest_cond'], right_on=['subject_id', 'condition'])

# Now let's calculate the real Invariance Breaking for S11 using true SAM scores
sub11 = df[df['subject_id'] == 'S11'].copy()
base = sub11[sub11['label'] == 'baseline']
sub11['EDA_Tonic_Z'] = (sub11['EDA_Tonic'] - base['EDA_Tonic'].mean()) / base['EDA_Tonic'].std()
sub11['HRV_LF_Z'] = (sub11['HRV_Inst_LF'] - base['HRV_Inst_LF'].mean()) / base['HRV_Inst_LF'].std()

# Look at High SCL state for S11
high_scl = sub11[sub11['EDA_Tonic_Z'] > 0.5]

print("\n=== S11: True Subjective JSD (Invariance Breaking) ===")
print("When S11 is in identical High SCL physiological state, what are their True Subjective Emotions?")

# Average SAM Arousal/Valence in this physiological state, split by the actual subjective context
breakdown = high_scl.groupby(['SAM_Valence', 'SAM_Arousal'])['timestamp'].count() / len(high_scl)
print(breakdown)

# Let's save the augmented dataset with subjective scores
# df.to_csv('wesad_100hz_with_subjective.csv', index=False)
print("Finished analysis.")
