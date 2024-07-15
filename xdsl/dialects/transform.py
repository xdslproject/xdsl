from xdsl.ir import (
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    irdl_attr_definition,
)


@irdl_attr_definition
class AffineMapType(ParametrizedAttribute, TypeAttribute):
    name = "transform.affine_map"


@irdl_attr_definition
class AnyOpType(ParametrizedAttribute, TypeAttribute):
    name = "transform.any_op"


@irdl_attr_definition
class AnyParamType(ParametrizedAttribute, TypeAttribute):
    name = "transform.any_param"


@irdl_attr_definition
class AnyValueType(ParametrizedAttribute, TypeAttribute):
    name = "transform.any_value"


Transform = Dialect(
    "transform",
    [],
    [
        # TYpes
        AffineMapType,
        AnyOpType,
        AnyParamType,
        AnyValueType,
    ],
)
