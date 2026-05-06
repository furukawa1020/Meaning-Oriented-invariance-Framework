import os
import hashlib
import datetime
import subprocess

def hash_file(filepath):
    """Calculate the SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_git_commit_hash(repo_dir):
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], 
            cwd=repo_dir, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "GIT_NOT_FOUND_OR_NOT_REPO"

def freeze_contract():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    thms_dir = os.path.join(base_dir, "thms_pipeline")
    
    # 1. Pre-registration documents
    docs_to_hash = {
        "analysis_config_hash": os.path.join(base_dir, "analysis_config.yaml"),
        "go_no_go_hash": os.path.join(base_dir, "07_submission_go_no_go.md"),
        "claim_strength_hash": os.path.join(base_dir, "08_claim_strength_ladder.md"),
        "reviewer_attacks_hash": os.path.join(base_dir, "09_known_reviewer_attacks.md"),
        "normalization_audit_schema_hash": os.path.join(base_dir, "normalization_audit_schema.md"),
        "feature_extraction_audit_schema_hash": os.path.join(base_dir, "feature_extraction_audit_schema.md"),
        "results_blinding_protocol_hash": os.path.join(base_dir, "results_blinding_protocol.md"),
        "test_fixture_design_hash": os.path.join(base_dir, "test_fixture_design.md")
    }
    
    # 2. Python Code and Tests
    code_to_hash = {
        "synthetic_fixtures_py_hash": os.path.join(thms_dir, "synthetic_fixtures.py"),
        "normalizers_py_hash": os.path.join(thms_dir, "normalizers.py"),
        "bad_normalizers_py_hash": os.path.join(thms_dir, "bad_normalizers.py"),
        "test_leakage_adversarial_py_hash": os.path.join(thms_dir, "test_leakage_adversarial.py"),
        "test_audit_schema_py_hash": os.path.join(thms_dir, "test_audit_schema.py"),
        "run_bad_implementations_py_hash": os.path.join(thms_dir, "run_bad_implementations_should_fail.py"),
        "freeze_contract_py_hash": os.path.join(thms_dir, "freeze_contract.py")
    }
    
    hashes = {}
    for key, path in {**docs_to_hash, **code_to_hash}.items():
        hashes[key] = hash_file(path)
        
    git_hash = get_git_commit_hash(base_dir)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    lock_content = f"""# Analysis Lock

*This file was automatically populated and timestamped by the blinding script.*

```yaml
git_commit_hash: "{git_hash}"
date_locked: "{timestamp}"

pre_registration_documents:
"""
    for key in docs_to_hash.keys():
        lock_content += f"  {key}: \"{hashes[key]}\"\n"

    lock_content += "\nphase1_code_and_tests:\n"
    for key in code_to_hash.keys():
        lock_content += f"  {key}: \"{hashes[key]}\"\n"

    lock_content += """
allowed_changes_after_lock:
  - bug fixes
  - documentation correction
  - computational failure fix
  - plot aesthetic adjustments

forbidden_changes_after_lock:
  - threshold adjustment
  - metric addition to fit desired claims
  - post-hoc method exclusion
  - post-hoc subject exclusion
  - figure replacement
  - primary claim alteration
```
"""
    lock_path = os.path.join(base_dir, "analysis_lock.md")
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(lock_content)
        
    print(f"Contract frozen at {timestamp}. Hashes written to {lock_path}")

if __name__ == "__main__":
    freeze_contract()
