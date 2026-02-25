from __future__ import annotations

from pathlib import Path

from moif.config import load_and_validate_config
from moif.logging_ import (
    capture_env,
    create_run_dir,
    hash_dataset,
    write_resolved_config,
    write_run_result_stub,
)


def run_detect(config_path: Path, out_dir: Path) -> None:
    cfg = load_and_validate_config(config_path)
    run_dir = create_run_dir(out_dir)
    write_resolved_config(run_dir, cfg)
    capture_env(run_dir)

    if cfg.get("dataset", {}).get("type") == "wesad":
        hash_dataset(Path(cfg["dataset"]["path"]), run_dir)

    # v0.1 skeleton: create empty findings + run_result with schema-valid structure.
    findings_path = run_dir / "findings.jsonl"
    findings_path.write_text("", encoding="utf-8")

    report_path = run_dir / "report.md"
    report_path.write_text("# MOIF Report (placeholder)\n", encoding="utf-8")

    write_run_result_stub(run_dir, findings_path, report_path)

    print(str(run_dir))

def run_report(run_dir: Path) -> None:
    # placeholder: keep contract
    report_path = run_dir / "report.md"
    if not report_path.exists():
        report_path.write_text("# MOIF Report (generated)\n", encoding="utf-8")
    print(str(report_path))

def run_sced(run_dir: Path, sced_type: str) -> None:
    # placeholder: copy templates in v0.1 skeleton
    proto_dir = run_dir / "protocol"
    proto_dir.mkdir(parents=True, exist_ok=True)
    (proto_dir / "protocol.yaml").write_text(f"type: {sced_type}\n", encoding="utf-8")
    (proto_dir / "run_sheet.md").write_text(f"# Run sheet ({sced_type})\n", encoding="utf-8")
    print(str(proto_dir))
