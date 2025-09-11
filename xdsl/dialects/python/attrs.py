from xdsl.ir import (
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    irdl_attr_definition,
)


@irdl_attr_definition
class ObjectType(ParametrizedAttribute, TypeAttribute):
    """Python opaque type"""

    name = "python.object"
