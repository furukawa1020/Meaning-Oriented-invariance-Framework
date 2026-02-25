import argparse
from pathlib import Path

from moif.pipeline import run_detect, run_report, run_sced


def main():
    p = argparse.ArgumentParser(prog="moif")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_detect = sub.add_parser("detect", help="Run detection pipeline")
    p_detect.add_argument("--config", required=True)
    p_detect.add_argument("--out", required=True)

    p_report = sub.add_parser("report", help="Render report from a run directory")
    p_report.add_argument("--run", required=True)

    p_sced = sub.add_parser("sced", help="Generate SCED protocol templates from a run")
    p_sced.add_argument("--run", required=True)
    p_sced.add_argument("--type", required=True, choices=["abab", "randomized_switch"])

    args = p.parse_args()

    if args.cmd == "detect":
        run_detect(Path(args.config), Path(args.out))
    elif args.cmd == "report":
        run_report(Path(args.run))
    elif args.cmd == "sced":
        run_sced(Path(args.run), args.type)
