"""
Lower Data-Layout Trees into LLVM struct types (and other things)
"""
import hashlib
import re
from collections.abc import Iterable

from xdsl.dialects import arith, builtin, llvm, printf
from xdsl.ir import Attribute, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
