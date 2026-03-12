import pandas as pd
import glob
import os
import numpy as np

def parse_quest(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        
    conditions = []
    data = {'PANAS': [], 'STAI': [], 'DIM': [], 'SSSQ': []}
    
    for line in lines:
        parts = line.strip().split(';')
        if not parts or parts[0] == '': continue
        
        row_type = parts[0].replace('# ', '').strip()
        
        if row_type == 'ORDER':
            conditions = [c.strip() for c in parts[1:] if c.strip() != '']
        elif row_type in ['PANAS', 'STAI', 'DIM', 'SSSQ']:
            values = []
            for v in parts[1:len(conditions)+1]:
                try:
                     values.append(int(v))
                except:
                     values.append(np.nan)
            data[row_type].append(values)
            
    res = {}
    for i, cond in enumerate(conditions):
        res[cond] = {}
        if data['STAI']:
            # Sum of STAI
            res[cond]['STAI'] = sum([row[i] for row in data['STAI'] if not np.isnan(row[i])])
        if data['DIM']:
            # DIM typically is Valence, Arousal
            if len(data['DIM']) >= 1:
                res[cond]['SAM_Valence'] = data['DIM'][0][i]
            if len(data['DIM']) >= 2:
                res[cond]['SAM_Arousal'] = data['DIM'][1][i]
    return res

results = {}
files = glob.glob(r'C:\Projects\Meaning-Oriented invariance Framework\moif\data\wesad\WESAD\*\*_quest.csv')
for f in files:
    sub = os.path.basename(f).split('_')[0]
    results[sub] = parse_quest(f)

print("Parsed subjective scores for subject S11:")
if 'S11' in results:
    import pprint
    pprint.pprint(results['S11'])
else:
    print("S11 not found.")

df_rows = []
for sub, conds in results.items():
    for cond, scores in conds.items():
        row = {'subject_id': sub, 'condition': cond}
        row.update(scores)
        df_rows.append(row)

df_quest = pd.DataFrame(df_rows)
# print(df_quest.head())
df_quest.to_csv('subjective_scores.csv', index=False)
print("Saved subjective_scores.csv")
