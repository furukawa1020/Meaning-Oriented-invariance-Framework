from pathlib import Path


def test_schemas_exist():
    root = Path(__file__).resolve().parent.parent
    for name in [
        "config.schema.json",
        "finding.schema.json",
        "run_result.schema.json",
        "external_state.schema.json",
        "sced_protocol.schema.json",
    ]:
        assert (root / "schemas" / name).exists()
