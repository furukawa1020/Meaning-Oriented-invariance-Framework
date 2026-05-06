import os
import hashlib
import datetime

def hash_file(filepath):
    """Calculate the SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def freeze_contract():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    files_to_hash = {
        "analysis_config_hash": os.path.join(base_dir, "analysis_config.yaml"),
        "go_no_go_hash": os.path.join(base_dir, "07_submission_go_no_go.md"),
        "normalization_audit_schema_hash": os.path.join(base_dir, "normalization_audit_schema.md"),
        "feature_extraction_audit_schema_hash": os.path.join(base_dir, "feature_extraction_audit_schema.md")
    }
    
    hashes = {}
    for key, path in files_to_hash.items():
        hashes[key] = hash_file(path)
        
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    lock_content = f"""# Analysis Lock

*This file was automatically populated and timestamped by the blinding script.*

```yaml
analysis_config_hash: "{hashes['analysis_config_hash']}"
go_no_go_hash: "{hashes['go_no_go_hash']}"
normalization_audit_schema_hash: "{hashes['normalization_audit_schema_hash']}"
feature_extraction_audit_schema_hash: "{hashes['feature_extraction_audit_schema_hash']}"
date_locked: "{timestamp}"

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
