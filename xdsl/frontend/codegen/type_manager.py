from dataclasses import dataclass, field
from xdsl.dialects import unimplemented

from xdsl.dialects.builtin import f32, i32
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


default_type_map = {
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
        
        # TODO: support type inference and more casts.
        cast_op = unimplemented.Cast.get(rhs, lhs_ty)
        self.inserter.insert_op(cast_op)
        return self.inserter.get_operand()
    
    def default_type(self, python_ty: type) -> str | Attribute:
        return default_type_map[python_ty]
