"""
This script benchmarks the xDSL parser by parsing all files in the
given root directory.
It then prints the total time taken to parse all files.
Each file is first parsed by MLIR to check that it is valid, and in
the generic form.
Each file is also splitted according to `split-input-file` if it is
present.
"""

import argparse
import cProfile
import glob
import subprocess
import timeit
from collections.abc import Iterable

from xdsl.context import MLContext
from xdsl.parser import Parser


def parse_file(file: str, ctx: MLContext):
    """
    Parse the given file.
    """
    parser = Parser(ctx, file)
    parser.parse_op()


def split_mlir_file(contents: str) -> list[str]:
    """
    Split the MLIR program into multiple ones, separated by `// -----`,
    if `split-input-file` is enabled for the test.
    """
    if "split-input-file" in contents:
        separated_contents = contents.split("// -----")
        return separated_contents
    return [contents]


def run_on_files(file_names: Iterable[str], mlir_path: str, ctx: MLContext):
    """
    Run the parser on all the given files.
    """
    total_time = 0
    n_total_files = 0
    n_parsed_files = 0

    for file_name in file_names:
        contents = open(file_name).read()
        print("Parsing file: " + file_name)
        splitted_contents = split_mlir_file(contents)
        print(f"File has been split into {len(splitted_contents)} sub-files.")

        # Parse each sub-file separately.
        for sub_contents in splitted_contents:
            # First, parse the file with MLIR to check that it is valid, and
            # print it back in generic form.
            res = subprocess.run(
                [
                    mlir_path,
                    "--allow-unregistered-dialect",
                    "-mlir-print-op-generic",
                    "-mlir-print-local-scope",
                ],
                input=sub_contents,
                text=True,
                capture_output=True,
                timeout=60,
            )
            if res.returncode != 0:
                continue
            n_total_files += 1

            # Get the generic form of the file.
            generic_sub_contents = res.stdout

            # Time the parser on the generic form.
            try:
                file_time = timeit.timeit(
                    lambda: parse_file(generic_sub_contents, ctx),
                    number=args.num_iterations,
                )
                total_time += file_time / args.num_iterations
                print("Time taken: " + str(file_time))
                n_parsed_files += 1
            except Exception as e:
                print("Error while parsing file: " + file_name)
                print(e)

    print("Total number of files:", n_total_files)
    print("Number of parsed files:", n_parsed_files)
    print("Total time to parse all files:", total_time)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="xDSL parser benchmark")
    arg_parser.add_argument(
        "root_directory",
        type=str,
        help="Path to the root directory containing MLIR files.",
    )
    arg_parser.add_argument("--mlir-path", type=str, help="Path to mlir-opt.")
    arg_parser.add_argument(
        "--num_iterations",
        type=int,
        required=False,
        default=1,
        help="Number of times to parse each file.",
    )
    arg_parser.add_argument(
        "--profile", action="store_true", help="Enable profiling metrics."
    )
    arg_parser.add_argument(
        "--timeout",
        type=int,
        required=False,
        default=60,
        help="Timeout for processing each sub-program with MLIR. (in seconds)",
    )

    args = arg_parser.parse_args()

    file_names = list(glob.iglob(args.root_directory + "/**/*.mlir", recursive=True))
    print("Found " + str(len(file_names)) + " files to parse.")

    ctx = MLContext(allow_unregistered=True)

    if args.profile:
        cProfile.run("run_on_files(file_names, args.mlir_path, ctx)")
    else:
        run_on_files(file_names, args.mlir_path, ctx)
