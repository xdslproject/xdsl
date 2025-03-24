# RUN: python %s | filecheck %s

from xdsl.irdl import OpDef, ParamAttrDef
from xdsl.utils.dialect_codegen import dump_dialect_pyfile

types = [
    ("Test_SingletonAType", ParamAttrDef(name="test.singleton_a", parameters=[])),
    ("Test_SingletonBType", ParamAttrDef(name="test.singleton_b", parameters=[])),
    ("Test_SingletonCType", ParamAttrDef(name="test.singleton_c", parameters=[])),
]

attrs = [("Test_TestAttr", ParamAttrDef(name="test.test", parameters=[]))]

ops = [
    (
        "Test_SimpleOp",
        OpDef(name="test.simple", assembly_format="attr-dict"),
    ),
]

dump_dialect_pyfile(
    "test",
    ops,
    attributes=attrs,
    types=types,
)

# CHECK:       from xdsl.dialects.builtin import *
# CHECK-NEXT:  from xdsl.ir import *
# CHECK-NEXT:  from xdsl.irdl import *

# CHECK:       # ruff: noqa: F403, F405

# CHECK:       @irdl_attr_definition
# CHECK-NEXT:  class Test_SingletonAType(ParametrizedAttribute, TypeAttribute):
# CHECK-NEXT:      name = "test.singleton_a"

# CHECK:       @irdl_attr_definition
# CHECK-NEXT:  class Test_SingletonBType(ParametrizedAttribute, TypeAttribute):
# CHECK-NEXT:      name = "test.singleton_b"

# CHECK:       @irdl_attr_definition
# CHECK-NEXT:  class Test_SingletonCType(ParametrizedAttribute, TypeAttribute):
# CHECK-NEXT:      name = "test.singleton_c"

# CHECK:       @irdl_attr_definition
# CHECK-NEXT:  class Test_TestAttr(ParametrizedAttribute):
# CHECK-NEXT:      name = "test.test"

# CHECK:       @irdl_op_definition
# CHECK-NEXT:  class Test_SimpleOp(IRDLOperation):
# CHECK-NEXT:      name = "test.simple"

# CHECK:       TestDialect = Dialect(
# CHECK-NEXT:      "test",
# CHECK-NEXT:      [Test_SimpleOp],
# CHECK-NEXT:      [Test_TestAttr, Test_SingletonAType, Test_SingletonBType, Test_SingletonCType],
# CHECK-NEXT:  )
