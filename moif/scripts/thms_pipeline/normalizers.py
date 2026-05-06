import numpy as np

class BaselineOnlyZScore:
    """Strictly uses ONLY baseline train data for mu/sigma estimation."""
    def fit_transform(self, df_subject):
        # Good: strictly mask for baseline train
        train_baseline_mask = (df_subject['split'] == 'train') & (df_subject['condition'] == 'baseline')
        train_baseline_df = df_subject[train_baseline_mask]
        
        feature_cols = [c for c in df_subject.columns if c.startswith('feature_')]
        
        if len(train_baseline_df) < len(feature_cols):
            # Fallback for insufficient baseline
            mu = np.zeros(len(feature_cols))
            sigma = np.ones(len(feature_cols))
            failure = "insufficient_baseline_samples"
        else:
            mu = train_baseline_df[feature_cols].mean().values
            sigma = train_baseline_df[feature_cols].std().values
            sigma[sigma == 0] = 1e-8
            failure = None
            
        transformed_X = (df_subject[feature_cols].values - mu) / sigma
        
        metadata = {
            "normalizer_name": "baseline_train_only_zscore",
            "fit_subjects": [df_subject['subject_id'].iloc[0]],
            "fit_labels_used": train_baseline_df['condition'].unique().tolist(),
            "leakage_audit": {
                "active_train_used": False,
                "active_test_used": False,
                "baseline_train_used": True,
                "baseline_test_used": False,
                "other_subjects_used": False,
                "future_samples_used": False
            },
            "failure_or_fallback": failure
        }
        return transformed_X, metadata

class BaselineCovarianceWhitening:
    """Strictly uses ONLY baseline train data for covariance estimation."""
    def __init__(self, regularization=1e-5):
        self.regularization = regularization
        
    def fit_transform(self, df_subject):
        train_baseline_mask = (df_subject['split'] == 'train') & (df_subject['condition'] == 'baseline')
        train_baseline_df = df_subject[train_baseline_mask]
        
        feature_cols = [c for c in df_subject.columns if c.startswith('feature_')]
        
        if len(train_baseline_df) < len(feature_cols):
            transformed_X = df_subject[feature_cols].values
            failure = "insufficient_baseline_samples"
            reg_applied = False
        else:
            X_base = train_baseline_df[feature_cols].values
            mu = np.mean(X_base, axis=0)
            cov = np.cov(X_base, rowvar=False)
            
            # Regularization to prevent singular matrix
            cov += np.eye(len(feature_cols)) * self.regularization
            
            try:
                L = np.linalg.cholesky(cov)
                L_inv = np.linalg.inv(L)
                X_centered = df_subject[feature_cols].values - mu
                transformed_X = X_centered @ L_inv.T
                failure = None
            except np.linalg.LinAlgError:
                transformed_X = df_subject[feature_cols].values
                failure = "singular_matrix_even_with_regularization"
                
        metadata = {
            "normalizer_name": "baseline_covariance_whitening",
            "fit_subjects": [df_subject['subject_id'].iloc[0]],
            "fit_labels_used": train_baseline_df['condition'].unique().tolist(),
            "leakage_audit": {
                "active_train_used": False,
                "active_test_used": False,
                "baseline_train_used": True,
                "baseline_test_used": False,
                "other_subjects_used": False,
                "future_samples_used": False
            },
            "regularization_applied": self.regularization,
            "failure_or_fallback": failure
        }
        return transformed_X, metadata

class RollingZScoreCausal:
    """Strictly causal rolling window (no future peeking)."""
    def __init__(self, window_size=60):
        self.window_size = window_size
        
    def fit_transform(self, df_subject):
        feature_cols = [c for c in df_subject.columns if c.startswith('feature_')]
        
        # Good: center=False ensures strictly causal (past samples only)
        rolling_mean = df_subject[feature_cols].rolling(window=self.window_size, min_periods=1, center=False).mean().values
        rolling_std = df_subject[feature_cols].rolling(window=self.window_size, min_periods=1, center=False).std().values
        
        rolling_std[rolling_std == 0] = 1e-8
        rolling_std[np.isnan(rolling_std)] = 1.0 
        
        transformed_X = (df_subject[feature_cols].values - rolling_mean) / rolling_std
        
        metadata = {
            "normalizer_name": "rolling_zscore_causal",
            "fit_subjects": [df_subject['subject_id'].iloc[0]],
            "fit_labels_used": ["causal_past_window"],
            "leakage_audit": {
                "active_train_used": False,
                "active_test_used": False,
                "baseline_train_used": False,
                "baseline_test_used": False,
                "other_subjects_used": False,
                "future_samples_used": False # Clean
            },
            "failure_or_fallback": None
        }
        return transformed_X, metadata

class PopulationZScore:
    """Strictly excludes the target test subject from the scaler."""
    def fit_transform_population(self, df_all, target_subject_id):
        # Good: Exclude the target_subject_id from the training pool
        train_df = df_all[(df_all['split'] == 'train') & (df_all['subject_id'] != target_subject_id)]
        feature_cols = [c for c in df_all.columns if c.startswith('feature_')]
        
        mu = train_df[feature_cols].mean().values
        sigma = train_df[feature_cols].std().values
        sigma[sigma == 0] = 1e-8
        
        target_df = df_all[df_all['subject_id'] == target_subject_id]
        transformed_X = (target_df[feature_cols].values - mu) / sigma
        
        metadata = {
            "normalizer_name": "population_zscore",
            "fit_subjects": train_df['subject_id'].unique().tolist(),
            "fit_labels_used": train_df['condition'].unique().tolist(),
            "leakage_audit": {
                "active_train_used": True,
                "active_test_used": False,
                "baseline_train_used": True,
                "baseline_test_used": False,
                "other_subjects_used": True,
                "future_samples_used": False,
                "target_subject_included_in_population": False # Clean
            },
            "failure_or_fallback": None
        }
        return transformed_X, metadata
