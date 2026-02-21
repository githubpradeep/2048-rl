from __future__ import annotations

import argparse
import subprocess
import sys


TRAIN_MODULES = {
    "2048": "src.train_dqn",
    "snake": "src.train_snake_dqn",
    "fruit": "src.train_fruit_dqn",
    "shooter": "src.train_shooter_dqn",
    "tetris": "src.train_tetris_dqn",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic game trainer dispatch")
    parser.add_argument("game", choices=sorted(TRAIN_MODULES))
    args, rest = parser.parse_known_args()

    module = TRAIN_MODULES[args.game]
    cmd = [sys.executable, "-m", module, *rest]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
