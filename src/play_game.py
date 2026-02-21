from __future__ import annotations

import argparse
import subprocess
import sys


PLAY_MODULES = {
    "2048": "src.play_agent",
    "snake": "src.play_snake_agent",
    "fruit": "src.play_fruit_agent",
    "shooter": "src.play_shooter_agent",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic game player dispatch")
    parser.add_argument("game", choices=sorted(PLAY_MODULES))
    args, rest = parser.parse_known_args()

    module = PLAY_MODULES[args.game]
    cmd = [sys.executable, "-m", module, *rest]
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
