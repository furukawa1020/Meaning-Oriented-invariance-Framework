import numpy as np

class BadBaselineZScoreUsesActiveTrain:
    """Intentionally leaky: uses active_train data when calculating baseline stats."""
    def fit_transform(self, df_subject):
        # BAD: Filtering only by split='train', ignoring condition='baseline'
        # Thus, active train stats bleed into the baseline scaler.
        train_df = df_subject[df_subject['split'] == 'train']
        feature_cols = [c for c in df_subject.columns if c.startswith('feature_')]
        
        mu = train_df[feature_cols].mean().values
        sigma = train_df[feature_cols].std().values
        sigma[sigma == 0] = 1e-8
        
        transformed_X = (df_subject[feature_cols].values - mu) / sigma
        
        # We intentionally log it honestly here so the metadata test catches it,
        # but even if the metadata lied, the mathematical adversarial test will catch it.
        metadata = {
            "normalizer_name": "bad_baseline_zscore_uses_active_train",
            "fit_subjects": [df_subject['subject_id'].iloc[0]],
            "fit_labels_used": train_df['condition'].unique().tolist(), # Will include 'active'
            "leakage_audit": {
                "active_train_used": True, # Leaky
                "active_test_used": False,
                "baseline_train_used": True,
                "baseline_test_used": False,
                "other_subjects_used": False,
                "future_samples_used": False
            }
        }
        return transformed_X, metadata

class BadBaselineZScoreUsesTest:
    """Intentionally leaky: uses the entire subject's data (train+test) for fitting."""
    def fit_transform(self, df_subject):
        feature_cols = [c for c in df_subject.columns if c.startswith('feature_')]
        
        # BAD: Using full dataset for mu and sigma
        mu = df_subject[feature_cols].mean().values
        sigma = df_subject[feature_cols].std().values
        sigma[sigma == 0] = 1e-8
        
        transformed_X = (df_subject[feature_cols].values - mu) / sigma
        
        metadata = {
            "normalizer_name": "bad_baseline_zscore_uses_test",
            "fit_subjects": [df_subject['subject_id'].iloc[0]],
            "fit_labels_used": df_subject['condition'].unique().tolist(),
            "leakage_audit": {
                "active_train_used": True,
                "active_test_used": True, # Leaky
                "baseline_train_used": True,
                "baseline_test_used": True, # Leaky
                "other_subjects_used": False,
                "future_samples_used": False
            }
        }
        return transformed_X, metadata

class BadRollingZScoreUsesCenteredWindow:
    """Intentionally leaky: non-causal rolling window (peeks into the future)."""
    def __init__(self, window_size=60):
        self.window_size = window_size
        
    def fit_transform(self, df_subject):
        feature_cols = [c for c in df_subject.columns if c.startswith('feature_')]
        
        # BAD: center=True means it uses future samples
        rolling_mean = df_subject[feature_cols].rolling(window=self.window_size, min_periods=1, center=True).mean().values
        rolling_std = df_subject[feature_cols].rolling(window=self.window_size, min_periods=1, center=True).std().values
        
        # Handle 0 std
        rolling_std[rolling_std == 0] = 1e-8
        rolling_std[np.isnan(rolling_std)] = 1.0 # fallback for first element
        
        transformed_X = (df_subject[feature_cols].values - rolling_mean) / rolling_std
        
        metadata = {
            "normalizer_name": "bad_rolling_zscore_uses_centered_window",
            "fit_subjects": [df_subject['subject_id'].iloc[0]],
            "fit_labels_used": ["all_past_and_future"],
            "leakage_audit": {
                "active_train_used": False,
                "active_test_used": False,
                "baseline_train_used": False,
                "baseline_test_used": False,
                "other_subjects_used": False,
                "future_samples_used": True # Leaky
            }
        }
        return transformed_X, metadata

class BadPopulationZScoreUsesTestSubject:
    """Intentionally leaky: uses the test subject in the population mean."""
    def fit_transform_population(self, df_all, target_subject_id):
        # BAD: We don't exclude target_subject_id
        train_df = df_all[df_all['split'] == 'train']
        feature_cols = [c for c in df_all.columns if c.startswith('feature_')]
        
        mu = train_df[feature_cols].mean().values
        sigma = train_df[feature_cols].std().values
        sigma[sigma == 0] = 1e-8
        
        target_df = df_all[df_all['subject_id'] == target_subject_id]
        transformed_X = (target_df[feature_cols].values - mu) / sigma
        
        metadata = {
            "normalizer_name": "bad_population_zscore_uses_test_subject",
            "fit_subjects": train_df['subject_id'].unique().tolist(), # Includes target
            "fit_labels_used": train_df['condition'].unique().tolist(),
            "leakage_audit": {
                "active_train_used": True,
                "active_test_used": False,
                "baseline_train_used": True,
                "baseline_test_used": False,
                "other_subjects_used": True,
                "future_samples_used": False,
                "target_subject_included_in_population": True # Leaky
            }
        }
        return transformed_X, metadata
