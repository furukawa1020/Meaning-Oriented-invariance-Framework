from pathlib import Path

from moif.pipeline import run_detect


def test_run_creates_artifacts(tmp_path: Path):
    out = tmp_path / "runs"
    run_detect(Path("configs/demo_csv.yaml"), out)
    latest = Path((out / "latest").read_text(encoding="utf-8").strip())
    assert (latest / "config.yaml").exists()
    assert (latest / "env.json").exists()
    assert (latest / "findings.jsonl").exists()
    assert (latest / "run_result.json").exists()
    assert (latest / "report.md").exists()
