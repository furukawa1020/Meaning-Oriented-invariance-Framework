from pathlib import Path

from moif.config import load_and_validate_config


def test_config_validates():
    cfg = load_and_validate_config(Path("configs/wesad_hr_absband.yaml"))
    assert cfg["dataset"]["type"] == "wesad"
