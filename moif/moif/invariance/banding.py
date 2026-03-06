import pandas as pd
import numpy as np

def apply_banding(df: pd.DataFrame, banding_cfg: dict) -> pd.DataFrame:
    """
    Apply banding logic based on config, returning a dataframe with an added 'in_band' column.
    
    Parameters:
        df: DataFrame with 'subject_id' and 'value' columns.
        banding_cfg: Dictionary with 'mode', 'abs', 'norm', etc.
    """
    mode = banding_cfg.get("mode")
    if mode not in ["abs", "norm"]:
        raise ValueError(f"Unknown banding mode: {mode}")
        
    df = df.copy()
    df['in_band'] = False
    
    if mode == "abs":
        params = banding_cfg.get("abs", {})
        low = params.get("low", -np.inf)
        high = params.get("high", np.inf)
        df['in_band'] = df['value'].between(low, high)
        
    elif mode == "norm":
        params = banding_cfg.get("norm", {})
        method = params.get("method")
        
        if method == "z":
            z_low = params.get("z_low", -np.inf)
            z_high = params.get("z_high", np.inf)
            
            # calculate z-score per subject
            def calc_z(group):
                mean = group.mean()
                std = group.std()
                if std == 0:
                    return np.zeros_like(group)
                return (group - mean) / std
                
            df['z_score'] = df.groupby('subject_id')['value'].transform(calc_z)
            df['in_band'] = df['z_score'].between(z_low, z_high)
            df.drop(columns=['z_score'], inplace=True)
            
        elif method == "quantile":
            q_low = params.get("q_low", 0.0)
            q_high = params.get("q_high", 1.0)
            
            def is_in_quantile(group):
                low_val = group.quantile(q_low)
                high_val = group.quantile(q_high)
                return group.between(low_val, high_val)
                
            df['in_band'] = df.groupby('subject_id')['value'].transform(is_in_quantile)
            
    return df
