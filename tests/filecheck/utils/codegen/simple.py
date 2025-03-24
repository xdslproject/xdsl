# RUN: python %s | filecheck %s

from xdsl.dialects.builtin import (
    AnyTensorTypeConstr,
    ComplexType,
    Float32Type,
    IndexType,
    IntAttr,
    IntegerType,
    NoneType,
    Signedness,
)
from xdsl.irdl import (
    AllOf,
    AnyAttr,
    AnyOf,
    BaseAttr,
    EqAttrConstraint,
    OpDef,
    OperandDef,
    ParamAttrConstraint,
    ParamAttrDef,
    ResultDef,
    VarOperandDef,
)
from xdsl.utils.dialect_codegen import dump_dialect_pyfile, generate_dynamic_attr_class

types = [
    ("Test_SingletonAType", ParamAttrDef(name="test.singleton_a", parameters=[])),
    ("Test_SingletonBType", ParamAttrDef(name="test.singleton_b", parameters=[])),
    ("Test_SingletonCType", ParamAttrDef(name="test.singleton_c", parameters=[])),
]

SingletonAType = generate_dynamic_attr_class(types[0][0], types[0][1])
SingletonBType = generate_dynamic_attr_class(types[1][0], types[1][1])
SingletonCType = generate_dynamic_attr_class(types[2][0], types[2][1])

attrs = [("Test_TestAttr", ParamAttrDef(name="test.test", parameters=[]))]

ops = [
    (
        "Test_SimpleOp",
        OpDef(name="test.simple", assembly_format="attr-dict"),
    ),
    (
        "Test_AndOp",
        OpDef(
            name="test.and",
            operands=[
                (
                    "in_",
                    OperandDef(
                        AllOf(
                            (
                                AnyAttr(),
                                BaseAttr(SingletonAType),
                            )
                        )
                    ),
                )
            ],
        ),
    ),
    ("Test_AnyOp", OpDef(name="test.any", operands=[("in_", OperandDef(AnyAttr()))])),
    (
        "Test_Integers",
        OpDef(
            name="test.integers",
            operands=[
                (
                    "any_int",
                    OperandDef(
                        ParamAttrConstraint(
                            IntegerType,
                            (EqAttrConstraint(IntAttr(8)), AnyAttr()),
                        )
                    ),
                ),
                ("any_integer", OperandDef(BaseAttr(IntegerType))),
            ],
        ),
    ),
    (
        "Test_OrOp",
        OpDef(
            name="test.or",
            operands=[
                (
                    "in_",
                    OperandDef(
                        AnyOf(
                            (
                                BaseAttr(SingletonAType),
                                BaseAttr(SingletonBType),
                                BaseAttr(SingletonCType),
                            )
                        )
                    ),
                )
            ],
        ),
    ),
    (
        "Test_TypesOp",
        OpDef(
            name="test.types",
            operands=[
                ("a", OperandDef(EqAttrConstraint(IntegerType(32)))),
                ("b", OperandDef(EqAttrConstraint(IntegerType(64, Signedness.SIGNED)))),
                (
                    "c",
                    OperandDef(EqAttrConstraint(IntegerType(8, Signedness.UNSIGNED))),
                ),
                ("d", OperandDef(EqAttrConstraint(IndexType()))),
                ("e", OperandDef(EqAttrConstraint(Float32Type()))),
                ("f", OperandDef(EqAttrConstraint(NoneType()))),
                ("v1", OperandDef(ParamAttrConstraint(ComplexType, (AnyAttr(),)))),
            ],
        ),
    ),
    (
        "Test_SingleOp",
        OpDef(
            name="test.single",
            operands=[("arg", OperandDef(AnyTensorTypeConstr))],
            results=[("res", ResultDef(AnyTensorTypeConstr))],
            assembly_format="$arg attr-dict : type($arg) -> type($res)",
        ),
    ),
    (
        "Test_VariadicityOp",
        OpDef(
            name="test.variadic",
            operands=[
                ("variadic", VarOperandDef(BaseAttr(SingletonAType))),
                ("required", OperandDef(BaseAttr(SingletonCType))),
            ],
        ),
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

# CHECK:       @irdl_op_definition
# CHECK-NEXT:  class Test_AndOp(IRDLOperation):
# CHECK-NEXT:      name = "test.and"
# CHECK-NEXT:      in_ = operand_def(AllOf(attr_constrs=(AnyAttr(), BaseAttr(Test_SingletonAType))))

# CHECK:       @irdl_op_definition
# CHECK-NEXT:  class Test_AnyOp(IRDLOperation):
# CHECK-NEXT:      name = "test.any"
# CHECK-NEXT:      in_ = operand_def(AnyAttr())

# CHECK:       @irdl_op_definition
# CHECK-NEXT:  class Test_Integers(IRDLOperation):
# CHECK-NEXT:      name = "test.integers"
# CHECK-NEXT:      any_int = operand_def(
# CHECK-NEXT:          ParamAttrConstraint(
# CHECK-NEXT:              IntegerType, (EqAttrConstraint(attr=IntAttr(data=8)), AnyAttr())
# CHECK-NEXT:          )
# CHECK-NEXT:      )
# CHECK-NEXT:      any_integer = operand_def(BaseAttr(IntegerType))

# CHECK:       @irdl_op_definition
# CHECK-NEXT:  class Test_OrOp(IRDLOperation):
# CHECK-NEXT:      name = "test.or"
# CHECK-NEXT:      in_ = operand_def(
# CHECK-NEXT:          AnyOf(
# CHECK-NEXT:              attr_constrs=(
# CHECK-NEXT:                  BaseAttr(Test_SingletonAType),
# CHECK-NEXT:                  BaseAttr(Test_SingletonBType),
# CHECK-NEXT:                  BaseAttr(Test_SingletonCType),
# CHECK-NEXT:              )
# CHECK-NEXT:          )
# CHECK-NEXT:      )

# CHECK:       @irdl_op_definition
# CHECK-NEXT:  class Test_TypesOp(IRDLOperation):
# CHECK-NEXT:      name = "test.types"
# CHECK-NEXT:      a = operand_def(EqAttrConstraint(attr=IntegerType(32)))
# CHECK-NEXT:      b = operand_def(EqAttrConstraint(attr=IntegerType(64, Signedness.SIGNED)))
# CHECK-NEXT:      c = operand_def(EqAttrConstraint(attr=IntegerType(8, Signedness.UNSIGNED)))
# CHECK-NEXT:      d = operand_def(EqAttrConstraint(attr=IndexType(parameters=())))
# CHECK-NEXT:      e = operand_def(EqAttrConstraint(attr=Float32Type(parameters=())))
# CHECK-NEXT:      f = operand_def(EqAttrConstraint(attr=NoneType(parameters=())))
# CHECK-NEXT:      v1 = operand_def(ParamAttrConstraint(ComplexType, (AnyAttr(),)))

# CHECK:       @irdl_op_definition
# CHECK-NEXT:  class Test_SingleOp(IRDLOperation):
# CHECK-NEXT:      name = "test.single"
# CHECK-NEXT:      arg = operand_def(BaseAttr(TensorType))
# CHECK-NEXT:      res = result_def(BaseAttr(TensorType))
# CHECK-EMPTY:
# CHECK-NEXT:      assembly_format = "$arg attr-dict : type($arg) -> type($res)"


# CHECK:       @irdl_op_definition
# CHECK-NEXT:  class Test_VariadicityOp(IRDLOperation):
# CHECK-NEXT:      name = "test.variadic"
# CHECK-NEXT:      variadic = var_operand_def(BaseAttr(Test_SingletonAType))
# CHECK-NEXT:      required = operand_def(BaseAttr(Test_SingletonCType))

# CHECK:       TestDialect = Dialect(
# CHECK-NEXT:      "test",
# CHECK-NEXT:      [
# CHECK-NEXT:          Test_SimpleOp,
# CHECK-NEXT:          Test_AndOp,
# CHECK-NEXT:          Test_AnyOp,
# CHECK-NEXT:          Test_Integers,
# CHECK-NEXT:          Test_OrOp,
# CHECK-NEXT:          Test_TypesOp,
# CHECK-NEXT:          Test_SingleOp,
# CHECK-NEXT:          Test_VariadicityOp,
# CHECK-NEXT:      ],
# CHECK-NEXT:      [Test_TestAttr, Test_SingletonAType, Test_SingletonBType, Test_SingletonCType],
# CHECK-NEXT:  )
