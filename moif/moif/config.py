from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft7Validator


def load_schema(schema_path: Path) -> dict:
    return json.loads(schema_path.read_text(encoding="utf-8"))

def load_and_validate_config(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema = load_schema(Path(__file__).resolve().parent.parent / "schemas" / "config.schema.json")
    v = Draft7Validator(schema)
    errors = sorted(v.iter_errors(cfg), key=lambda e: e.path)
    if errors:
        msg = "\n".join([f"- {list(e.path)}: {e.message}" for e in errors])
        raise ValueError(f"Config validation failed:\n{msg}")
    return cfg
