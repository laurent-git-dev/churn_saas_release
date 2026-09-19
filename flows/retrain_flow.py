"""Point d'entrée pour ``make flow`` — délègue à flows/retraining.py.

Usage :
    uv run python flows/retrain_flow.py [--forcer]
    make flow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flows.retraining import flow_retrainement  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow de réentraînement churn SaaS.")
    parser.add_argument(
        "--forcer", action="store_true", help="Force le recalcul de toutes les étapes."
    )
    args = parser.parse_args()
    sys.exit(flow_retrainement(forcer=args.forcer))
