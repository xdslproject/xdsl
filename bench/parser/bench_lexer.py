"""
This script benchmarks the xDSL lexer by parsing all files in the
given root directory.
It then prints the total time taken to parse all files.
"""

import argparse
import cProfile
import glob
import timeit
from collections.abc import Iterable

from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import MLIRLexer, MLIRTokenKind


def lex_file(file: Input):
    """
    Lex the given file
    """
    lexer = MLIRLexer(file)
    while lexer.lex().kind is not MLIRTokenKind.EOF:
        pass


def run_on_files(file_names: Iterable[str]):
    total_time = 0
    for file_name in file_names:
        print("Lexing file: " + file_name)
        try:
            contents = open(file_name).read()
            input = Input(contents, file_name)
            file_time = timeit.timeit(
                lambda: lex_file(input), number=args.num_iterations
            )
            total_time += file_time / args.num_iterations
            print("Time taken: " + str(file_time))
        except Exception as e:
            print("Error while lexing file: " + file_name)
            print(e)

    print(total_time)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="xDSL lexer benchmark")
    arg_parser.add_argument(
        "root_directory",
        type=str,
        help="Path to the root directory containing MLIR files.",
    )
    arg_parser.add_argument(
        "--num_iterations",
        type=int,
        required=False,
        default=1,
        help="Number of times to lex each file.",
    )
    arg_parser.add_argument(
        "--profile", action="store_true", help="Enable profiling metrics."
    )

    args = arg_parser.parse_args()

    file_names = list(glob.iglob(args.root_directory + "/**/*.mlir", recursive=True))
    print("Found " + str(len(file_names)) + " files to lex.")

    if args.profile:
        cProfile.run("run_on_files(file_names)")
    else:
        run_on_files(file_names)
