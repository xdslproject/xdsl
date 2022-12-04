from dataclasses import dataclass, field
from xdsl.dialects import arith, tensor, unimplemented

from xdsl.dialects.builtin import IndexType, IntegerType, Signedness, TensorType, f32, i32, i1
from xdsl.frontend.codegen.inserter import OpInserter
from xdsl.ir import Attribute, Operation, SSAValue


@dataclass
class TypeManagerException(Exception):
    """
    Exception type if type manager encounter an error.
    """
    msg: str

    def __str__(self) -> str:
        return f"Conversion of type hint failed with: {self.msg}."


# TODO: This should be defined by the frontend program.
default_type_map = {
    bool: i1,
    int: i32,
    float: f32,
    str: str,
}

@dataclass
class TypeManager:
    """
    Class responsible for managing and iferring types in assignemnts, etc.
    This class is just a placeholder for more nuanced type inference class.
    """

    inserter: OpInserter
    """Inserter to fix types if necessary"""

    def match(self, lhs_ty: Attribute, rhs: SSAValue) -> SSAValue:
        # If both types match, return.
        if lhs_ty == rhs.typ:
            return rhs
        
        if isinstance(lhs_ty, IndexType):
            if isinstance(rhs.typ, IntegerType):
                cast_op = arith.IndexCast.get(rhs, lhs_ty)
            else:
                cast_op = unimplemented.Cast.get(rhs, lhs_ty)
        elif isinstance(lhs_ty, IntegerType):
            if isinstance(rhs.typ, IndexType):
                cast_op = arith.IndexCast.get(rhs, lhs_ty)
            elif isinstance(rhs.typ, IntegerType):
                lhs_width = lhs_ty.width.data
                rhs_width = rhs.typ.width.data
                if lhs_width < rhs_width:
                    cast_op = arith.TruncI.get(rhs, lhs_ty)
                else:
                    cast_op = arith.ExtSI.get(rhs, lhs_ty)
            else:
                # TODO: support type inference and more casts.
                cast_op = unimplemented.Cast.get(rhs, lhs_ty)
        elif isinstance(lhs_ty, TensorType) and isinstance(rhs.typ, TensorType) and lhs_ty.element_type == rhs.typ.element_type:
            cast_op = tensor.Cast.get(rhs, lhs_ty)
        elif isinstance(lhs_ty, TensorType) and isinstance(rhs.typ, TensorType):
            if isinstance(lhs_ty.element_type, IntegerType):
                if isinstance(rhs.typ.element_type, IndexType):
                    cast_op = arith.IndexCast.get(rhs, lhs_ty)
                elif isinstance(rhs.typ.element_type, IntegerType):
                    lhs_width = lhs_ty.element_type.width.data
                    rhs_width = rhs.typ.element_type.width.data
                    if lhs_width < rhs_width:
                        cast_op = arith.TruncI.get(rhs, lhs_ty)
                    else:
                        cast_op = arith.ExtSI.get(rhs, lhs_ty)
                else:
                    # TODO: support type inference and more casts.
                    cast_op = unimplemented.Cast.get(rhs, lhs_ty)
        else:
            # TODO: support type inference and more casts.
            cast_op = unimplemented.Cast.get(rhs, lhs_ty)
        self.inserter.insert_op(cast_op)
        return self.inserter.get_operand()
    
    def default_type(self, python_ty: type) -> str | Attribute:
        return default_type_map[python_ty]
