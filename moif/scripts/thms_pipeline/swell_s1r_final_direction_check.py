import pandas as pd
import numpy as np
from pathlib import Path
import json

def final_direction_check():
    swell_dir = Path(r"C:\Projects\Meaning-Oriented invariance Framework\moif\data\swell")
    table_path = swell_dir / "swell_joined_minute_block_table.csv"
    df = pd.read_csv(table_path)
    
    # Target variables
    raw = 'Performance'
    rec = 'Performance (recoded)'
    stress = 'Stress'
    effort = 'MentalEffort'
    
    # 1. Full Correlations
    corrs = df[[raw, rec, stress, effort]].corr()
    
    # 2. Condition Means
    means = df.groupby('Condition')[[raw, rec, stress, effort]].mean()
    
    results = {
        "correlations": corrs.to_dict(),
        "condition_means": means.to_dict(),
        "interpretation_logic": {
            "is_raw_performance_lower_in_stressful_conditions": bool(means.loc['T', raw] < means.loc['N', raw]) if 'T' in means.index and 'N' in means.index else None,
            "is_recoded_performance_higher_in_stressful_conditions": bool(means.loc['T', rec] > means.loc['N', rec]) if 'T' in means.index and 'N' in means.index else None,
            "corr_recoded_with_stress": corrs.loc[rec, stress],
            "corr_recoded_with_effort": corrs.loc[rec, effort]
        }
    }
    
    print(json.dumps(results, indent=2))
    with open(swell_dir / "swell_s1r_final_direction_report.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    final_direction_check()
