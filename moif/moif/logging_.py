from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return []

def create_run_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha1(ts.encode()).hexdigest()[:8]
    run_id = f"{ts}_{h}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "latest").write_text(str(run_dir), encoding="utf-8")
    return run_dir

def write_resolved_config(run_dir: Path, cfg: dict) -> None:
    import yaml
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

def capture_env(run_dir: Path) -> None:
    env = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": sys.platform,
        "git_commit": _git_commit(),
        "pip_freeze": _pip_freeze(),
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

def write_run_result_stub(run_dir: Path, findings_path: Path, report_path: Path) -> None:
    from datetime import datetime, timezone
    rr = {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": _git_commit()},
        "config_path": str(run_dir / "config.yaml"),
        "findings_path": str(findings_path),
        "report_path": str(report_path),
        "summary": {"n_findings": 0, "n_significant": 0},
    }
    (run_dir / "run_result.json").write_text(json.dumps(rr, indent=2), encoding="utf-8")

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def hash_dataset(root: Path, run_dir: Path) -> None:
    files = {}
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file():
                files[str(p.relative_to(root)).replace("\\", "/")] = _sha256_file(p)
    out = {
        "dataset_root": str(root).replace("\\", "/"),
        "files": files
    }
    (run_dir / "data_hash.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
