from xdsl.dialects.builtin import (
    I64,
    DenseArrayBase,
    SymbolNameConstraint,
)
from xdsl.dialects.utils.dimension_list import DimensionList
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    prop_def,
    traits_def,
)
from xdsl.traits import SymbolOpInterface


@irdl_op_definition
class MeshOp(IRDLOperation):
    name = "mesh.mesh"

    sym_name = prop_def(SymbolNameConstraint())
    shape = prop_def(DenseArrayBase[I64])

    traits = traits_def(SymbolOpInterface())

    assembly_format = (
        "$sym_name `(` `shape` `=` custom<DimensionList>($shape) `)` attr-dict"
    )

    custom_directives = (DimensionList,)


Mesh = Dialect(
    "mesh",
    [
        MeshOp,
    ],
    [],
)
