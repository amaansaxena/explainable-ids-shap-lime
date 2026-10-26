"""Run the available pipeline stages in order."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SMOKE_DIR = Path(__file__).resolve().parent / "smoke_output"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help=f"quick check on tiny subsets; models and results go to {SMOKE_DIR.name}/, not models/ or results/")
    args = parser.parse_args()
    smoke = ["--smoke"] if args.smoke else []
    env = dict(os.environ)
    if args.smoke:
        env["IDS_OUTPUT_DIR"] = str(SMOKE_DIR)
    commands = [
        # Preprocessing writes the shared finalized data, so a smoke run only creates it if missing.
        [sys.executable, "-m", "src.preprocessing", "--dataset", "all", *(smoke or ["--force"])],
        [sys.executable, "-m", "src.train", "--dataset", "all", "--model", "all", "--force", *smoke],
        [sys.executable, "-m", "src.evaluate", "--experiments", "A", "B", "C", "D", "--force", *smoke],
        [sys.executable, "-m", "src.explain", "--experiments", "A", "B", "C", "D", "--models", "mlp_a", "mlp_b", "mlp_c", "rf", "dt", "--methods", "shap", "lime", "--force", *smoke],
        [sys.executable, "-m", "src.compare_xai", *smoke],
    ]
    for command in commands:
        subprocess.run(command, check=True, env=env)
    if args.smoke:
        print(f"Smoke run finished; outputs are in {SMOKE_DIR}")


if __name__ == "__main__":
    main()
