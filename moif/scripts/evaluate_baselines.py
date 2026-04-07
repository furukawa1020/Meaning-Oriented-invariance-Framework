import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.spatial.distance import mahalanobis
import warnings
import os
import sys
sys.path.append('.')
from moif.loaders.wesad import load_wesad

warnings.filterwarnings('ignore')

def calculate_overlap_omega(baseline_data, stress_data, radius_multiplier=1.0):
    """
    Calculates the distribution overlap metric Omega (percentage of stress data 
    that falls within the 1-sigma dense region of baseline data).
    """
    if len(baseline_data) < 10 or len(stress_data) < 10:
        return 0.0
    
    # Define the dense boundary of the baseline data (1 sigma)
    # Using NearestNeighbors to find if points are within R of any baseline point
    nbrs = NearestNeighbors(radius=radius_multiplier, algorithm='auto').fit(baseline_data)
    
    # For each point in stress_data, find neighbors in baseline_data within radius
    # radius_neighbors returns (distances, indices). We just need to know if len(indices) > 0.
    # To be fast, we can use kneighbors with k=1 and check if dist < radius
    distances, _ = nbrs.kneighbors(stress_data, n_neighbors=1)
    
    # Calculate percentage
    overlap_count = np.sum(distances <= radius_multiplier)
    omega = (overlap_count / len(stress_data)) * 100
    return omega

def calculate_separability(X, y):
    """
    Calculate how well Baseline and Stress can be separated using Logistic Regression.
    """
    y = np.array(y)
    if len(np.unique(y)) < 2:
        return 0.0
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return f1_score(y_test, preds, pos_label='stress')

def process_subject(subj_df, features):
    label_map = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation'}
    subj_df['label_name'] = subj_df['label'].map(label_map)
    
    # Filter only baseline and stress for comparison
    df_eval = subj_df[subj_df['label_name'].isin(['baseline', 'stress'])].copy()
    if df_eval.empty or len(df_eval['label_name'].unique()) < 2:
        return None
    
    # 1. Global Z-Score
    scaler = StandardScaler()
    X_global_z = scaler.fit_transform(df_eval[features])
    
    # 2. Rolling Z-Score (60 seconds = 6000 samples at 100Hz)
    rolling_window = 6000
    df_eval_rolling = df_eval[features].copy()
    for col in features:
        mean_r = df_eval[col].rolling(window=rolling_window, min_periods=1).mean()
        std_r = df_eval[col].rolling(window=rolling_window, min_periods=1).std().replace(0, 1)
        df_eval_rolling[col] = (df_eval[col] - mean_r) / std_r
    # Drop NaNs created by rolling std (first element)
    df_eval_rolling = df_eval_rolling.fillna(0)
    X_rolling_z = df_eval_rolling.values
    
    # 3. Mahalanobis DBA
    # Calculate covariance and mean of BASELINE only
    base_data = df_eval[df_eval['label_name'] == 'baseline'][features]
    if len(base_data) < 100:
        return None
        
    mu_base = base_data.mean().values
    cov_base = np.cov(base_data.values, rowvar=False)
    
    # Add small epsilon to diagonal to prevent singular matrix
    cov_base += np.eye(cov_base.shape[0]) * 1e-6
    try:
        inv_cov_base = np.linalg.inv(cov_base)
    except np.linalg.LinAlgError:
        return None
        
    X_dba = np.zeros((len(df_eval), 1))
    for i, row in enumerate(df_eval[features].values):
        X_dba[i] = mahalanobis(row, mu_base, inv_cov_base)
    
    # Replace outliers > 10 in DBA with 10 to stabilize (since it's a distance)
    # X_dba = np.clip(X_dba, 0, 10)
    
    # Now calculate Omega and Separability for each method
    methods = {
        'Global Z': X_global_z,
        'Rolling Z': X_rolling_z,
        'DBA': X_dba
    }
    
    y = df_eval['label_name'].values
    idx_base = (y == 'baseline')
    idx_stress = (y == 'stress')
    
    res = {'Subject': df_eval.iloc[0]['subject_id']}
    
    for m_name, X in methods.items():
        # Standardize the output space of the method so 1 sigma means the same thing geometrically
        if X.shape[1] > 1: # multi-dimensional
            X_std = StandardScaler().fit_transform(X)
        else:
            X_std = StandardScaler().fit_transform(X)
            
        b_data = X_std[idx_base]
        s_data = X_std[idx_stress]
        
        omega = calculate_overlap_omega(b_data, s_data, radius_multiplier=1.0)
        sep_f1 = calculate_separability(X_std, y)
        
        res[f'{m_name} Omega'] = omega
        res[f'{m_name} Separability (F1)'] = sep_f1
        
    return res

print("Loading WESAD raw features...")
df_all = pd.read_csv('results/wesad_100hz_instantaneous_raw.csv')
features = ['ECG_cwt_LF', 'ECG_cwt_HF', 'EDA_Phasic', 'EDA_Tonic']

results = []
subjects = df_all['subject_id'].unique()

print(f"Evaluating {len(subjects)} subjects for Overlap & Separability...")
for subj in subjects:
    print(f"Processing {subj}...")
    subj_df = df_all[df_all['subject_id'] == subj]
    res = process_subject(subj_df, features)
    if res:
        results.append(res)

df_res = pd.DataFrame(results)
print("\n--- RESULTS OVERVIEW ---")
print(df_res.mean(numeric_only=True).round(2).to_string())

df_res.to_csv('results/evaluation_baselines_wesad.csv', index=False)
print("\nSaved detailed results to results/evaluation_baselines_wesad.csv")
