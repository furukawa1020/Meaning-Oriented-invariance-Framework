import os
import pandas as pd
import subprocess

def create_mock_csvs(base_dir, main_diff, disagree_rate, kappa, subj_diff_pct, calib_diff):
    os.makedirs(base_dir, exist_ok=True)
    
    # 1. results_main.csv
    pd.DataFrame({
        'method': ['A', 'B'],
        'auroc_mean': [0.70, 0.70 + main_diff],
        'auprc_mean': [0.65, 0.65 + main_diff]
    }).to_csv(os.path.join(base_dir, "results_main.csv"), index=False)
    
    # 2. pipeline_disagreement.csv
    pd.DataFrame({
        'prediction_disagreement_rate': [disagree_rate, disagree_rate],
        'cohens_kappa': [kappa, kappa]
    }).to_csv(os.path.join(base_dir, "pipeline_disagreement.csv"), index=False)
    
    # 3. results_subject_level.csv
    # We need subjects where diff >= 0.05.
    n_subjects = 10
    n_het = int(n_subjects * subj_diff_pct)
    
    rows = []
    for i in range(n_subjects):
        rows.append({'subject_id': f"S{i}", 'method': 'A', 'auroc': 0.70})
        # If heterogeneous, difference is 0.06 (> 0.05). If not, difference is 0.01.
        diff = 0.06 if i < n_het else 0.01
        rows.append({'subject_id': f"S{i}", 'method': 'B', 'auroc': 0.70 + diff})
        
    pd.DataFrame(rows).to_csv(os.path.join(base_dir, "results_subject_level.csv"), index=False)
    
    # 4. calibration_length_results.csv
    pd.DataFrame({
        'duration': [300, 60],
        'auroc_mean': [0.75, 0.75 - calib_diff]
    }).to_csv(os.path.join(base_dir, "calibration_length_results.csv"), index=False)

def run_dry_run():
    # 1. Strong Results (Passes all 4 gates)
    create_mock_csvs("mock_strong_results", 
                     main_diff=0.04, 
                     disagree_rate=0.06, kappa=0.88, 
                     subj_diff_pct=0.30, 
                     calib_diff=0.03)
                     
    # 2. Moderate Results (Passes 2 gates)
    create_mock_csvs("mock_moderate_results", 
                     main_diff=0.01, # Fails
                     disagree_rate=0.06, kappa=0.88, # Passes
                     subj_diff_pct=0.10, # Fails
                     calib_diff=0.03) # Passes
                     
    # 3. Null Results (Passes 0 gates)
    create_mock_csvs("mock_null_results", 
                     main_diff=0.01, 
                     disagree_rate=0.01, kappa=0.95, 
                     subj_diff_pct=0.05, 
                     calib_diff=0.01)
                     
    script_path = r"C:\Projects\Meaning-Oriented invariance Framework\moif\scripts\thms_pipeline\run_submission_gate.py"
    for mock_dir in ["mock_strong_results", "mock_moderate_results", "mock_null_results"]:
        subprocess.run(["python", script_path, "--results_dir", mock_dir])

if __name__ == "__main__":
    run_dry_run()
