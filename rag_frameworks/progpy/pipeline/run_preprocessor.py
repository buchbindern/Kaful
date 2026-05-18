"""
pipeline/run_preprocessor.py

CLI runner for the data summary preprocessor.

Usage:
    python run_preprocessor.py \
        --machine coffee_machine \
        --events data/coffee_machine/events.csv \
        --maintenance data/coffee_machine/maintenance.csv \
        --output data/coffee_machine/summary.json
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from pipeline.preprocessor import build_data_summary, save_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build a DataSummary from raw CSVs")
    parser.add_argument("--machine",     required=True, help="Machine name e.g. coffee_machine")
    parser.add_argument("--events",      required=True, help="Path to events CSV")
    parser.add_argument("--maintenance", default=None,  help="Path to maintenance CSV (optional)")
    parser.add_argument("--output",      required=True, help="Output path for summary JSON")
    parser.add_argument("--type",        default=None,  help="Override machine type detection")
    return parser.parse_args()


def run():
    args = parse_args()

    summary = build_data_summary(
        machine_name=args.machine,
        events_csv=args.events,
        maintenance_csv=args.maintenance,
        machine_type=args.type,
    )

    save_summary(summary, args.output)
    print("\nDone. Next step: run pipeline/decompose.py with this summary.")


if __name__ == "__main__":
    run()