#!/usr/bin/env python3
"""Extract test MLIR files from the xdsl repository.

This is required as generic MLIR files generated using `mlir-opt` cannot be
created on-the-fly in CI, as `mlir-opt` is not available without an expensive
build operation on the runner.

This script allows a user to scrape the test cases from xDSL automatically as
they are updated as needed, but also helps pin the test data to make fair
comparisons.
"""

import subprocess
from pathlib import Path
from shutil import copy, rmtree

from tqdm import tqdm

RESOURCES_DIR = Path(__file__).parent
TEST_DIR = RESOURCES_DIR.parent.parent / "xdsl/tests"
GENERIC_DIR = RESOURCES_DIR / "generic_test_mlir/"
RAW_DIR = RESOURCES_DIR / "raw_test_mlir/"


def split_mlir_file(contents: str) -> list[str] | str:
    """Split the MLIR program into multiple ones.

    If `split-input-file` is enabled for the test then "// -----" is used as a
    seperator for the file.
    """
    if "split-input-file" in contents:
        return contents.split("// -----")
    return contents


def mlir_opt_contents(contents: str, mlir_path: str) -> str | None:
    """Run `mlir-opt` on the (optionally split) contents of a MLIR file."""
    result = subprocess.run(  # noqa: S603
        [
            mlir_path,
            "--allow-unregistered-dialect",
            "-mlir-print-op-generic",
            "-mlir-print-local-scope",
        ],
        input=contents,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def main(mlir_path: str = "mlir-opt") -> None:
    """The script to copy across and generate the raw and generic files."""
    rmtree(RAW_DIR)
    rmtree(GENERIC_DIR)
    RAW_DIR.mkdir()
    GENERIC_DIR.mkdir()

    print("Copying all test MLIR files.")
    num_files: int = 0
    for file in tqdm(TEST_DIR.rglob("*.mlir")):
        destination = RAW_DIR / "__".join(str(file.relative_to(TEST_DIR)).split("/"))
        if destination.exists():
            print(f"WARNING: overwriting {destination}")
        else:
            num_files += 1
        copy(file, destination)

    print("\nApplying `mlir-opt` to all test MLIR files.")
    for file in tqdm(RAW_DIR.iterdir(), total=num_files):
        contents = file.read_text()
        if isinstance((split := split_mlir_file(contents)), list):
            for i, split_contents in enumerate(split):
                output_file = GENERIC_DIR / f"{file.stem}__split_{i}.{file.suffix}"
                mlir_opt = mlir_opt_contents(split_contents, mlir_path)
                if mlir_opt is not None:
                    output_file.write_text(mlir_opt)
        else:
            output_file = GENERIC_DIR / file.name
            mlir_opt = mlir_opt_contents(split, mlir_path)
            if mlir_opt is not None:
                output_file.write_text(mlir_opt)


if __name__ == "__main__":
    main()
