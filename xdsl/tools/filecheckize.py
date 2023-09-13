#!/usr/bin/env python3

import argparse
import re
import sys
from io import TextIOWrapper
from typing import cast


def main():
    parser = argparse.ArgumentParser(
        description="Generate the FileCheck lines that expects the given input."
    )
    parser.add_argument(
        "file",
        type=argparse.FileType("r"),
        default=sys.stdin,
        nargs="?",
        help="Input file to read. Defaults to standard input.",
    )
    parser.add_argument(
        "--mlir-anonymize",
        action=argparse.BooleanOptionalAction,
        help="Anonymize MLIR SSA value names and basic block names.",
    )
    parser.add_argument(
        "--check-empty-lines",
        action=argparse.BooleanOptionalAction,
        help="Check strictly for empty lines. Default behavior is to just skip them.",
    )
    parser.add_argument(
        "--strip-comments",
        action=argparse.BooleanOptionalAction,
        help="Strip comments from input rather than checking for them.",
    )
    parser.add_argument(
        "--comments-prefix",
        type=str,
        default="//",
        help="Set the prefix of a comment line. Defaults to '//' as in MLIR.",
    )
    parser.add_argument(
        "--check-prefix",
        type=str,
        default="CHECK",
        help="Set the Filecheck prefix to generate. defaults to 'CHECK' as Filecheck's default.",
    )

    args = parser.parse_args(sys.argv[1:])

    comment_line = re.compile(rf"^\s*{re.escape(args.comments_prefix)}.*$")
    ssa_value_name = re.compile(r"%([\d]+|[\w$._-][\w\d$._-]*)(:|#[\d]*)?")
    basic_block_name = re.compile(r"\^([\d]+|[\w$._-][\w\d$._-]*)")

    prefix = args.check_prefix

    next = False

    for line in cast(TextIOWrapper, args.file):
        line = line.rstrip()

        # Ignore whitespace-only lines
        # xDSL doesn't print spacing blank lines between ops or regions
        if not line:
            if args.check_empty_lines:
                print(f"// {prefix}-EMPTY:")
                next = True
            else:
                next = False
            continue

        # Ignore remaining comment lines
        if args.strip_comments:
            if re.match(comment_line, line):
                continue

        if args.mlir_anonymize:
            # Anonymize SSA value names
            line = re.sub(ssa_value_name, r"%{{.*}}", line)

            # Anonymize basic blocks names
            line = re.sub(basic_block_name, r"^{{.*}}", line)

        # Print the modified line
        if next:
            print(f"// {prefix}-NEXT:  ", end="")
        else:
            print(f"// {prefix}:       ", end="")
            next = True
        print(line)
