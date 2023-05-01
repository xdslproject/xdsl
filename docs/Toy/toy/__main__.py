import argparse
from pathlib import Path

from frontend.parser import Parser

parser = argparse.ArgumentParser(description="Process Toy file")
parser.add_argument("source", type=Path, help="toy source file")
parser.add_argument(
    "--emit",
    dest="emit",
    choices=["ast"],
    default="ast",
    help="Action to perform on source file (default: interpret file)",
)


def main(path: Path, emit: str):
    path = args.source

    with open(path, "r") as f:
        parser = Parser(path, f.read())
        ast = parser.parseModule()

    if emit == "ast":
        print(ast.dump())
        return

    print(f"Unknown option {emit}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.source, args.emit)
