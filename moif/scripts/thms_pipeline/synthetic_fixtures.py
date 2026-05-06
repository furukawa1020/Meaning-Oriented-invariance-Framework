import numpy as np
import pandas as pd

def generate_synthetic_data(seed=42):
    """
    Generates a synthetic dataset for adversarial testing of normalizers.
    Strictly NO real WESAD or CASE data is used here.
    
    Structure:
    - 2 datasets ('SYNTH_WESAD', 'SYNTH_CASE')
    - 3 subjects per dataset ('S1', 'S2', 'S3')
    - 2 splits per subject ('train', 'test') -> chronologically ordered
    - 2 conditions per split ('baseline', 'active')
    - feature_dim = 5
    """
    np.random.seed(seed)
    
    datasets = ['SYNTH_WESAD', 'SYNTH_CASE']
    subjects = ['S1', 'S2', 'S3']
    splits = ['train', 'test']
    conditions = ['baseline', 'active']
    feature_dim = 5
    
    samples_per_segment = 100
    
    rows = []
    timestamp = 0.0
    
    for ds in datasets:
        for sub in subjects:
            for split in splits:
                for cond in conditions:
                    # Generate base data (mean 0, std 1)
                    features = np.random.randn(samples_per_segment, feature_dim)
                    
                    # Add distinct offsets to make it easier to track leakage
                    if sub == 'S1':
                        features += 10.0
                    elif sub == 'S2':
                        features += 20.0
                    else:
                        features += 30.0
                        
                    if cond == 'active':
                        features += 5.0 # baseline vs active difference
                        
                    for i in range(samples_per_segment):
                        row = {
                            'dataset': ds,
                            'subject_id': sub,
                            'timestamp': timestamp,
                            'split': split,
                            'condition': cond,
                            'label': 1 if cond == 'active' else 0
                        }
                        for d in range(feature_dim):
                            row[f'feature_{d+1}'] = features[i, d]
                        
                        rows.append(row)
                        timestamp += 0.1 # 10Hz sampling
                        
    df = pd.DataFrame(rows)
    return df

def inject_active_train_perturbation(df, subject_id, offset=100.0):
    """Injects a massive +100 offset specifically into the active train segment."""
    df_adv = df.copy()
    mask = (df_adv['subject_id'] == subject_id) & (df_adv['split'] == 'train') & (df_adv['condition'] == 'active')
    feature_cols = [c for c in df_adv.columns if c.startswith('feature_')]
    df_adv.loc[mask, feature_cols] += offset
    return df_adv

def inject_test_perturbation(df, subject_id, offset=1000.0):
    """Injects a massive +1000 offset specifically into the test segment."""
    df_adv = df.copy()
    mask = (df_adv['subject_id'] == subject_id) & (df_adv['split'] == 'test')
    feature_cols = [c for c in df_adv.columns if c.startswith('feature_')]
    df_adv.loc[mask, feature_cols] += offset
    return df_adv

def inject_future_spike(df, subject_id, spike_index=150, spike_value=9999.0):
    """Injects a single massive spike to test causal rolling normalizers."""
    df_adv = df.copy()
    mask = df_adv['subject_id'] == subject_id
    subject_indices = df_adv[mask].index
    if spike_index < len(subject_indices):
        target_idx = subject_indices[spike_index]
        feature_cols = [c for c in df_adv.columns if c.startswith('feature_')]
        df_adv.loc[target_idx, feature_cols] += spike_value
    return df_adv

def make_singular_covariance(df, subject_id):
    """Makes feature_1 perfectly correlated with feature_2 to induce a singular matrix."""
    df_adv = df.copy()
    mask = df_adv['subject_id'] == subject_id
    df_adv.loc[mask, 'feature_2'] = df_adv.loc[mask, 'feature_1']
    return df_adv

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Generated synthetic dataset with {len(df)} rows.")
    print("Columns:", df.columns.tolist())
    print("Subject breakdown:\n", df['subject_id'].value_counts())
