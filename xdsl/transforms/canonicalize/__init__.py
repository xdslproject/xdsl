# ruff: noqa
# import main module first
from xdsl.transforms.canonicalize.canonicalize_pass import (
    CanonicalizationPass,
    is_canonicalization,
)

# THEN import the other things so they resolve dependencies correctly:
import xdsl.transforms.canonicalize.canonicalize_riscv
