from __future__ import annotations

from xdsl.ir.core import ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition

"""
Implementation of the LTL dialect by CIRCT. Documentation: https://circt.llvm.org/docs/Dialects/LTL/
"""


@irdl_attr_definition
class propertytype(ParametrizedAttribute, TypeAttribute):
    """
    Explicitly represents a verifiable property built from linear temporal logic sequences and quantifiers.
    """
