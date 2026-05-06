import unittest
from jsonschema import validate
from jsonschema.exceptions import ValidationError

from synthetic_fixtures import generate_synthetic_data
from normalizers import BaselineOnlyZScore, RollingZScoreCausal

# The required schema structure according to normalization_audit_schema.md
NORMALIZATION_AUDIT_SCHEMA = {
    "type": "object",
    "required": ["normalizer_name", "fit_subjects", "fit_labels_used", "leakage_audit", "failure_or_fallback"],
    "properties": {
        "normalizer_name": {"type": "string"},
        "fit_subjects": {
            "type": "array",
            "items": {"type": "string"}
        },
        "fit_labels_used": {
            "type": "array",
            "items": {"type": "string"}
        },
        "leakage_audit": {
            "type": "object",
            "required": [
                "active_train_used",
                "active_test_used",
                "baseline_train_used",
                "baseline_test_used",
                "other_subjects_used",
                "future_samples_used"
            ],
            "properties": {
                "active_train_used": {"type": "boolean"},
                "active_test_used": {"type": "boolean"},
                "baseline_train_used": {"type": "boolean"},
                "baseline_test_used": {"type": "boolean"},
                "other_subjects_used": {"type": "boolean"},
                "future_samples_used": {"type": "boolean"}
            }
        },
        "failure_or_fallback": {
            "type": ["string", "null"]
        }
    }
}

class TestAuditSchema(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.df_clean = generate_synthetic_data()
        cls.df_s1 = cls.df_clean[cls.df_clean['subject_id'] == 'S1'].copy()
        
    def test_baseline_only_metadata_matches_schema(self):
        normalizer = BaselineOnlyZScore()
        _, metadata = normalizer.fit_transform(self.df_s1)
        
        try:
            validate(instance=metadata, schema=NORMALIZATION_AUDIT_SCHEMA)
        except ValidationError as e:
            self.fail(f"BaselineOnlyZScore metadata failed schema validation: {e}")
            
    def test_rolling_causal_metadata_matches_schema(self):
        normalizer = RollingZScoreCausal(window_size=10)
        _, metadata = normalizer.fit_transform(self.df_s1)
        
        try:
            validate(instance=metadata, schema=NORMALIZATION_AUDIT_SCHEMA)
        except ValidationError as e:
            self.fail(f"RollingZScoreCausal metadata failed schema validation: {e}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
