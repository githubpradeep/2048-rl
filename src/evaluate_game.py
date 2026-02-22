from __future__ import annotations

import argparse
import subprocess
import sys


EVAL_MODULES = {
    "2048": "src.evaluate",
    "breakout": "src.evaluate_breakout",
    "snake": "src.evaluate_snake",
    "fruit": "src.evaluate_fruit",
    "flappy": "src.evaluate_flappy",
    "shooter": "src.evaluate_shooter",
    "tetris": "src.evaluate_tetris",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic game evaluator dispatch")
    parser.add_argument("game", choices=sorted(EVAL_MODULES))
    args, rest = parser.parse_known_args()

    module = EVAL_MODULES[args.game]
    cmd = [sys.executable, "-m", module, *rest]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
