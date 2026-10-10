from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.dialects import pdl
from xdsl.dialects.builtin import FunctionType
from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.utils.exceptions import CodeGenerationException
from xdsl.ir import Block, Region, SSAValue
from xdsl.utils.lexer import Location


def build_pdl_pattern(
    name: str, signature: FunctionType, body: Region, location: Location
) -> pdl.PatternOp:
    """
    Build a pattern whose Python body rewrites one unconstrained operation.

    Parameter annotations specify the IR handle type, not matching constraints.
    The matcher accepts any number of operands and results on the root operation.
    """
    if signature.inputs.data != (pdl.OperationType(),) or signature.outputs:
        raise CodeGenerationException(
            *location,
            "PDL rewrites require one !pdl.operation argument and no results.",
        )

    operands = pdl.OperandsOp(None)
    types = pdl.TypesOp()
    root = pdl.OperationOp(
        None, operand_values=[operands.value], type_values=[types.result]
    )
    arg = body.block.args[0]
    root.op.name_hint = arg.name_hint
    arg.replace_all_uses_with(root.op)
    body.block.erase_arg(arg)
    rewrite = pdl.RewriteOp(root.op, body)
    return pdl.PatternOp(1, name, Region(Block([operands, types, root, rewrite])))


def build_pdl_return(values: Sequence[SSAValue], location: Location) -> None:
    if values:
        raise CodeGenerationException(
            *location, "Return values are not supported in PDL rewrites."
        )


@dataclass
class PyPDLContext(PyASTContext):
    """Encapsulate the mapping between Python and IR types and operations."""

    def __init__(self):
        super().__init__()
        self.register_function_definition(build_pdl_pattern)
        self.register_return(build_pdl_return)
