from xdsl.ir import ParametrizedAttribute
from xdsl.irdl import irdl_attr_definition


@irdl_attr_definition
class AttrA(ParametrizedAttribute):
    pass


@irdl_attr_definition
class AttrB(ParametrizedAttribute):
    pass


@irdl_attr_definition
class AttrC(ParametrizedAttribute):
    pass


def test_attr_constraint_get_unique_base():
    pass
