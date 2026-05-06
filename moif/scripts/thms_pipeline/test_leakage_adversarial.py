import unittest
import numpy as np

from synthetic_fixtures import (
    generate_synthetic_data,
    inject_active_train_perturbation,
    inject_test_perturbation,
    inject_future_spike,
    make_singular_covariance
)

from bad_normalizers import (
    BadBaselineZScoreUsesActiveTrain,
    BadBaselineZScoreUsesTest,
    BadRollingZScoreUsesCenteredWindow,
    BadPopulationZScoreUsesTestSubject
)

from normalizers import (
    BaselineOnlyZScore,
    BaselineCovarianceWhitening,
    RollingZScoreCausal,
    PopulationZScore
)

class TestAdversarialLeakage(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df_clean = generate_synthetic_data()
        cls.df_s1_clean = cls.df_clean[cls.df_clean['subject_id'] == 'S1'].copy()
        
    def test_01_active_train_perturbation_fails_bad_baseline(self):
        """Active train +100 should completely skew the BAD baseline normalizer."""
        df_adv = inject_active_train_perturbation(self.df_clean, 'S1', offset=100.0)
        df_s1_adv = df_adv[df_adv['subject_id'] == 'S1']
        
        normalizer = BadBaselineZScoreUsesActiveTrain()
        
        # Clean transform
        X_clean, _ = normalizer.fit_transform(self.df_s1_clean)
        # Adversarial transform
        X_adv, meta = normalizer.fit_transform(df_s1_adv)
        
        # The clean and adv transformations for the baseline condition should be different 
        # because the active train leaked into the mu/sigma!
        mask_baseline = self.df_s1_clean['condition'] == 'baseline'
        diff = np.abs(X_clean[mask_baseline] - X_adv[mask_baseline]).mean()
        
        # We EXPECT a large diff because it's leaky
        self.assertTrue(diff > 0.1, f"Expected massive leakage difference, got {diff}")
        self.assertTrue(meta['leakage_audit']['active_train_used'])
        
    def test_02_active_train_perturbation_passes_correct_baseline(self):
        """Active train +100 should NOT affect the correct baseline normalizer."""
        df_adv = inject_active_train_perturbation(self.df_clean, 'S1', offset=100.0)
        df_s1_adv = df_adv[df_adv['subject_id'] == 'S1']
        
        normalizer = BaselineOnlyZScore()
        
        X_clean, _ = normalizer.fit_transform(self.df_s1_clean)
        X_adv, meta = normalizer.fit_transform(df_s1_adv)
        
        mask_baseline = self.df_s1_clean['condition'] == 'baseline'
        diff = np.abs(X_clean[mask_baseline] - X_adv[mask_baseline]).mean()
        
        # We EXPECT 0.0 diff because it strictly isolated the baseline train
        self.assertAlmostEqual(diff, 0.0, places=5, msg="Leakage detected in correct normalizer!")
        self.assertFalse(meta['leakage_audit']['active_train_used'])
        
    def test_03_test_perturbation_fails_bad_baseline(self):
        """Test segment +1000 should completely skew the BAD baseline normalizer."""
        df_adv = inject_test_perturbation(self.df_clean, 'S1', offset=1000.0)
        df_s1_adv = df_adv[df_adv['subject_id'] == 'S1']
        
        normalizer = BadBaselineZScoreUsesTest()
        
        X_clean, _ = normalizer.fit_transform(self.df_s1_clean)
        X_adv, meta = normalizer.fit_transform(df_s1_adv)
        
        mask_train = self.df_s1_clean['split'] == 'train'
        diff = np.abs(X_clean[mask_train] - X_adv[mask_train]).mean()
        
        self.assertTrue(diff > 1.0, "Expected test data to leak back into train params.")
        self.assertTrue(meta['leakage_audit']['active_test_used'])

    def test_04_test_perturbation_passes_correct_baseline(self):
        df_adv = inject_test_perturbation(self.df_clean, 'S1', offset=1000.0)
        df_s1_adv = df_adv[df_adv['subject_id'] == 'S1']
        
        normalizer = BaselineOnlyZScore()
        
        X_clean, _ = normalizer.fit_transform(self.df_s1_clean)
        X_adv, meta = normalizer.fit_transform(df_s1_adv)
        
        mask_train = self.df_s1_clean['split'] == 'train'
        diff = np.abs(X_clean[mask_train] - X_adv[mask_train]).mean()
        
        self.assertAlmostEqual(diff, 0.0, places=5)
        
    def test_05_future_spike_fails_centered_rolling(self):
        """A spike in the future should change PAST values in a non-causal centered rolling window."""
        spike_idx = 150
        df_adv = inject_future_spike(self.df_clean, 'S1', spike_index=spike_idx, spike_value=9999.0)
        df_s1_adv = df_adv[df_adv['subject_id'] == 'S1'].copy()
        
        normalizer = BadRollingZScoreUsesCenteredWindow(window_size=10)
        X_clean, _ = normalizer.fit_transform(self.df_s1_clean)
        X_adv, meta = normalizer.fit_transform(df_s1_adv)
        
        # Check exactly 1 index BEFORE the spike (t=149). It should be identical if causal.
        # But centered window will look ahead and alter it.
        diff_pre_spike = np.abs(X_clean[spike_idx-1] - X_adv[spike_idx-1]).sum()
        
        self.assertTrue(diff_pre_spike > 0.1, "Expected future data to leak backwards in time.")
        self.assertTrue(meta['leakage_audit']['future_samples_used'])

    def test_06_future_spike_passes_causal_rolling(self):
        """A spike in the future MUST NOT change PAST values in a causal rolling window."""
        spike_idx = 150
        df_adv = inject_future_spike(self.df_clean, 'S1', spike_index=spike_idx, spike_value=9999.0)
        df_s1_adv = df_adv[df_adv['subject_id'] == 'S1'].copy()
        
        normalizer = RollingZScoreCausal(window_size=10)
        X_clean, _ = normalizer.fit_transform(self.df_s1_clean)
        X_adv, meta = normalizer.fit_transform(df_s1_adv)
        
        diff_pre_spike = np.abs(X_clean[spike_idx-1] - X_adv[spike_idx-1]).sum()
        
        self.assertAlmostEqual(diff_pre_spike, 0.0, places=5)
        self.assertFalse(meta['leakage_audit']['future_samples_used'])

    def test_07_heldout_subject_fails_bad_population(self):
        # We perturb S1 (held out). If the scaler uses S1, S1's transform will change.
        df_adv = inject_active_train_perturbation(self.df_clean, 'S1', offset=100.0)
        
        normalizer = BadPopulationZScoreUsesTestSubject()
        X_clean, _ = normalizer.fit_transform_population(self.df_clean, target_subject_id='S1')
        X_adv, meta = normalizer.fit_transform_population(df_adv, target_subject_id='S1')
        
        diff = np.abs(X_clean - X_adv).mean()
        self.assertTrue(diff > 0.5, "Expected target subject to illegally bleed into population stats.")
        self.assertTrue(meta['leakage_audit']['target_subject_included_in_population'])

    def test_08_heldout_subject_passes_correct_population(self):
        # We perturb S1 (held out). The scaler is strictly trained on S2 and S3.
        # Thus, the scaler parameters won't change, and S1's transform will perfectly reflect the perturbation
        # as a linear offset, but the scaler logic itself is completely shielded.
        # Specifically, the baseline condition of S1 should be identically transformed
        # because the scaler on S2+S3 didn't change!
        df_adv = inject_active_train_perturbation(self.df_clean, 'S1', offset=100.0)
        
        normalizer = PopulationZScore()
        X_clean, _ = normalizer.fit_transform_population(self.df_clean, target_subject_id='S1')
        X_adv, meta = normalizer.fit_transform_population(df_adv, target_subject_id='S1')
        
        df_s1_mask_baseline = (self.df_clean['subject_id'] == 'S1') & (self.df_clean['condition'] == 'baseline')
        baseline_indices = self.df_clean[df_s1_mask_baseline].index
        
        s1_indices = self.df_clean[self.df_clean['subject_id'] == 'S1'].index
        idx_mapping = {val: i for i, val in enumerate(s1_indices)}
        
        # Check baseline indices. They should be strictly identical because active perturbation
        # shouldn't affect the scaler, and the scaler doesn't look at S1 at all.
        diff = 0.0
        for idx in baseline_indices:
            local_i = idx_mapping[idx]
            diff += np.abs(X_clean[local_i] - X_adv[local_i]).sum()
            
        self.assertAlmostEqual(diff, 0.0, places=5)
        self.assertFalse(meta['leakage_audit']['target_subject_included_in_population'])
        
    def test_09_singular_covariance_requires_logged_regularization(self):
        df_adv = make_singular_covariance(self.df_clean, 'S1')
        df_s1_adv = df_adv[df_adv['subject_id'] == 'S1']
        
        normalizer = BaselineCovarianceWhitening(regularization=1e-5)
        X_adv, meta = normalizer.fit_transform(df_s1_adv)
        
        # Must not crash, must return valid array
        self.assertFalse(np.isnan(X_adv).any())
        self.assertEqual(meta['regularization_applied'], 1e-5)
        self.assertIsNone(meta['failure_or_fallback'])
        
    def test_10_short_baseline_failure_fallback(self):
        # Create a tiny baseline of just 1 sample
        df_s1_tiny = self.df_s1_clean.copy()
        mask = df_s1_tiny['condition'] == 'baseline'
        # Drop all but first row of baseline
        drop_idx = df_s1_tiny[mask].index[1:]
        df_s1_tiny = df_s1_tiny.drop(drop_idx)
        
        normalizer = BaselineOnlyZScore()
        X_adv, meta = normalizer.fit_transform(df_s1_tiny)
        
        # Must log fallback
        self.assertEqual(meta['failure_or_fallback'], "insufficient_baseline_samples")

if __name__ == '__main__':
    unittest.main(verbosity=2)
