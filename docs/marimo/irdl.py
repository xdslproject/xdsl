import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    # xDSL should be available in the environment
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.context import Context
    from xdsl.printer import Printer

    from xdsl.dialects.builtin import i32
    from xdsl.utils.exceptions import VerifyException

    # Context, containing information about the registered dialects
    context = Context()

    # Some useful dialects
    context.load_dialect(Arith)
    context.load_dialect(Builtin)

    # Printer used to pretty-print MLIR data structures
    printer = Printer()
    return Printer, VerifyException, i32, mo, printer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # IRDL tutorial

    ## An Intermediate Representation Definition Language (IRDL) for SSA Compilers
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Introduction

    xDSL is an extensible compiler, meaning that new operations, attributes, and types can be added. xDSL provides an embedded DSL, IRDL, to define new dialects.
    This tutorial aims to show the different features IRDL has, and presents examples on how to use them.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Attribute constraints

    Attribute constraints represent invariants over attributes, and are an important concept for defining new attributes and operations. In practice, an attribute constraint is a child class of `AttrConstraint` that implements a `verify` method. The method takes an attribute to verify as parameter, and a dictionary associating constraint variables to attributes. `verify` does not return anything, but raises an exception if the invariant is not respected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Any Constraint

    An `Any` constraint will never trigger an exception, and will always pass:
    """
    )
    return


@app.cell
def _():
    from xdsl.dialects.builtin import IndexType, IntegerType, StringAttr, i64
    from xdsl.irdl import AnyAttr, ConstraintContext

    # Construct the constraint
    any_constraint = AnyAttr()

    # This will pass without triggering an exception
    any_constraint.verify(i64, ConstraintContext())
    any_constraint.verify(StringAttr("ga"), ConstraintContext())
    return ConstraintContext, IndexType, IntegerType, StringAttr, i64


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Equality Constraint

    An equality constraint ensures that the attribute is equal to one provided to the constraint:
    """
    )
    return


@app.cell
def _(ConstraintContext, i32, i64):
    from xdsl.irdl import EqAttrConstraint

    # Construct the constraint
    eq_constraint = EqAttrConstraint(i64)

    # This will pass without triggering an exception
    eq_constraint.verify(i64, ConstraintContext())

    # This will trigger an exception
    try:
        eq_constraint.verify(i32, ConstraintContext())
    except Exception as e:
        print(e)
    return (EqAttrConstraint,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Base Attribute Constraint

    A base attribute constraint ensures that the attribute base type is equal to an expected attribute base type:
    """
    )
    return


@app.cell
def _(ConstraintContext, StringAttr, VerifyException, i32):
    from xdsl.dialects.builtin import IntAttr
    from xdsl.irdl import BaseAttr

    # Construct the constraint
    base_constraint = BaseAttr(StringAttr)

    # This will pass without triggering an exception
    base_constraint.verify(StringAttr("ga"), ConstraintContext())
    base_constraint.verify(StringAttr("bu"), ConstraintContext())

    # This will trigger an exception
    try:
        base_constraint.verify(i32, ConstraintContext())
    except VerifyException as e:
        print(e)
    return BaseAttr, IntAttr, base_constraint


@app.cell
def _(ConstraintContext, IntAttr, VerifyException, base_constraint):
    # This too
    try:
        base_constraint.verify(IntAttr(3), ConstraintContext())
    except VerifyException as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Attribute Constraint Coercion

    To simplify the definitions of constraints, constraint constructors expecting an attribute constraints will coerce `Attribute` to an equality attribute constraint, and will coerce an `Attribute` type to a base attribute constraint. this is done using the `irdl_to_attr_constraint` function:
    """
    )
    return


@app.cell
def _(BaseAttr, EqAttrConstraint, StringAttr, i32):
    from xdsl.irdl import AnyOf, irdl_to_attr_constraint

    assert irdl_to_attr_constraint(i32) == EqAttrConstraint(i32)
    assert irdl_to_attr_constraint(StringAttr) == BaseAttr(StringAttr)
    assert AnyOf([i32, StringAttr]) == AnyOf([EqAttrConstraint(i32), BaseAttr(StringAttr)])
    return AnyOf, irdl_to_attr_constraint


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Or Constraint

    An `or` constraint ensures that one of the given attribute constraints is satisfied by the attribute:
    """
    )
    return


@app.cell
def _(AnyOf, ConstraintContext, StringAttr, i32, i64):
    # Construct the constraint. Note that we are using the coercion defined previously.
    or_constraint = AnyOf([i32, StringAttr])

    # This will pass without triggering an exception, since the first constraint is satisfied
    or_constraint.verify(i32, ConstraintContext())

    # This will pass without triggering an exception, since the second constraint is satisfied
    or_constraint.verify(StringAttr("ga"), ConstraintContext())
    or_constraint.verify(StringAttr("bu"), ConstraintContext())

    # This will trigger an exception, since none of the constraints are satisfied
    try:
        or_constraint.verify(i64, ConstraintContext())
    except Exception as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Parametric Attribute Constraint

    A parametric attribute constraint is satisfied by parametric attributes of a certain base type. In addition, parametric attribute constraints specify constraints for each of the parameters of the attribute:
    """
    )
    return


@app.cell
def _(ConstraintContext, IntAttr, VerifyException, i32, i64):
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.irdl import ParamAttrConstraint

    # Construct the constraint. Note that we are using the coercion defined previously.
    param_constraint = ParamAttrConstraint(IntegerAttr, [IntAttr, i32])

    # This will pass without triggering an exception.
    param_constraint.verify(IntegerAttr(IntAttr(42), i32), ConstraintContext())
    param_constraint.verify(IntegerAttr(IntAttr(23), i32), ConstraintContext())

    # This will trigger an exception since the attribute type is not the expected one
    try:
        param_constraint.verify(i64, ConstraintContext())
    except VerifyException as e:
        print(e)
    return IntegerAttr, param_constraint


@app.cell
def _(
    ConstraintContext,
    IntAttr,
    IntegerAttr,
    VerifyException,
    i64,
    param_constraint,
):
    # This will trigger an exception since the second parameter constraint is not satisfied

    try:
        param_constraint.verify(IntegerAttr(IntAttr(42), i64), ConstraintContext())
    except VerifyException as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Constraint Variables

    Constraint variables are used to specify equality between attributes in operation and attribute definitions. They also contain a constraint that must be satisfied.
    The first time a constraint variable is used, it will check that the constraint is satisfied. If it is satisfied, it sets the variable to the given attribute. If a constraint variable is already set, it will check that the given attribute is equal to the one already set. Two constraint variables with the same name are considered equal, and are expected to carry the same constraint.
    """
    )
    return


@app.cell
def _(BaseAttr, ConstraintContext, IndexType, IntegerType, VerifyException):
    from xdsl.irdl import VarConstraint

    var_constraint = VarConstraint("T", BaseAttr(IntegerType))
    constraint_context = ConstraintContext()

    # The underlying constraint is not satisfied by the given attribute
    try:
        var_constraint.verify(IndexType(), constraint_context)
    except VerifyException as e:
        print(e)
    return VarConstraint, constraint_context, var_constraint


@app.cell
def _(constraint_context, i32, var_constraint):
    # The constraint sets the constraint variable if the attribute verifies
    var_constraint.verify(i32, constraint_context)
    print(constraint_context.get_variable("T"))
    return


@app.cell
def _(VerifyException, constraint_context, i64, var_constraint):
    # Since the variable is set, the constraint can now only be satisfied by the same attribute
    try:
        var_constraint.verify(i64, constraint_context)
    except VerifyException as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Custom Constraints

    Users can define their own constraints for their own types. For instance, here is the definition of the `ArrayOfConstraint` constraint, which expects an `ArrayAttr` where all elements of the array satisfy a specific constraint:
    """
    )
    return


@app.cell
def _(ConstraintContext, IntAttr, VerifyException, irdl_to_attr_constraint):
    from dataclasses import dataclass

    from xdsl.dialects.builtin import ArrayAttr
    from xdsl.ir import Attribute
    from xdsl.irdl import AttrConstraint
    from xdsl.utils.hints import isa
    from typing_extensions import TypeVar

    @dataclass(frozen=True)
    class ArrayOfConstraint(AttrConstraint):
        # The constraint that needs to be satisfied by all elements of the array
        elem_constr: AttrConstraint

        # The custom init applies the attribute constraint coercion
        def __init__(self, constr: Attribute | type[Attribute] | AttrConstraint):
            object.__setattr__(self, "elem_constr", irdl_to_attr_constraint(constr))

        # Check that an attribute satisfies the constraints
        def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
            # We first check that the attribute is an ArrayAttr
            if not isa(attr, ArrayAttr[Attribute]):
                raise VerifyException(f"expected attribute ArrayData but got {attr}")

            # We check the constraint for all elements in the array
            for e in attr.data:
                self.elem_constr.verify(e, constraint_context)

        def mapping_type_vars(self, type_var_mapping: dict[TypeVar, AttrConstraint]):
            return ArrayOfConstraint(self.elem_constr.mapping_type_vars(type_var_mapping))


    array_constraint = ArrayOfConstraint(IntAttr)

    # This will pass without triggering an exception
    array_constraint.verify(ArrayAttr([IntAttr(42)]), ConstraintContext())
    array_constraint.verify(ArrayAttr([IntAttr(3), IntAttr(7)]), ConstraintContext())
    return ArrayAttr, array_constraint


@app.cell
def _(ConstraintContext, array_constraint, i32):
    # This will trigger an exception, since the attribute is not an array
    try:
        array_constraint.verify(i32, ConstraintContext())
    except Exception as e:
        print(e)
    return


@app.cell
def _(ArrayAttr, ConstraintContext, IntAttr, StringAttr, array_constraint):
    # This will trigger an exception, since the array contains attribute that do not satisfies the constraint
    try:
        array_constraint.verify(
            ArrayAttr([IntAttr(42), StringAttr("ga")]), ConstraintContext()
        )
    except Exception as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Attribute Definition""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Data Attributes

    `Data` attribute types are defined by inheriting the python `Data` class. Each data attribute definition should define a `name`, and two methods for conversion to a string representation. Here is for example the definition of `IntAttr`:
    """
    )
    return


@app.cell
def _(Printer, printer):
    from xdsl.ir import Data
    from xdsl.irdl import irdl_attr_definition
    from xdsl.parser import AttrParser


    @irdl_attr_definition
    class MyIntAttr(Data[int]):
        name = "test.my_int"

        @classmethod
        def parse_parameter(cls, parser: AttrParser) -> int:
            data = parser.parse_integer()
            return data

        def print_parameter(self, printer: Printer) -> None:
            printer.print_string(f"{self.data}")


    MyIntAttr(3).print_parameter(printer)
    return (irdl_attr_definition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Parametrized Attributes

    `ParametrizedAttribute` attribute types are defined using the `irdl_attr_definition` decorator on a class. Such class should contain a `name` field specifying the attribute name. Parameters are added to attribute definitions by defining fields with a type, and optionally with a `param_def`. The field names correspond to the parameter names, and `param_def` contains an optional constraint that should be respected by this parameter. The order of the fields correspond to the order of the parameters when using the attribute. Upon construction of an attribute, all constraints will be checked, and an exception will be raised if the invariants are not satisfied.

    Here is an example of an integer type definition:
    """
    )
    return


@app.cell
def _(EqAttrConstraint, IntAttr, StringAttr, irdl_attr_definition):
    from xdsl.ir import ParametrizedAttribute
    from xdsl.irdl import param_def


    # Represent an integer type with a given bitwidth
    @irdl_attr_definition
    class MyIntegerType(ParametrizedAttribute):
        # Name of the type. This is used for printing and parsing.
        name = "test.integer_type"

        # Only parameter of the type, with an `EqAttrConstraint` constraint.
        width: IntAttr = param_def(EqAttrConstraint(IntAttr(32)))


    my_i32 = MyIntegerType(IntAttr(32))

    # This will trigger an exception, since the attribute is not an IntAttr
    try:
        MyIntegerType(StringAttr("not an int"))
    except Exception as e:
        print(e)

    # This will trigger an exception, since the attribute is not equal to the expected value
    try:
        MyIntegerType(IntAttr(64))
    except Exception as e:
        print(e)

    print(MyIntegerType(IntAttr(32)))
    return (my_i32,)


@app.cell
def _(mo):
    mo.md(r"""Each parameter can be accessed using the `parameters` field.""")
    return


@app.cell
def _(my_i32):
    my_i32.parameters
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our attribute definition also defines accessors for each parameter based on the name given in the `param_def` field:""")
    return


@app.cell
def _(my_i32):
    my_i32.width
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Operation Definition

    Operations are defined similarly to `ParametrizedAttribute`, by using the `irdl_op_definition` decorator. The decorator allows the definition of expected operands, results, attributes, and regions. Each definition should contain a `name` static field, which is used for parsing and printing:
    """
    )
    return


@app.cell
def _(printer):
    from xdsl.irdl import IRDLOperation, irdl_op_definition


    @irdl_op_definition
    class MyEmptyOp(IRDLOperation):
        name = "my_dialect.my_op"


    my_op = MyEmptyOp.build()
    printer.print_op(my_op)
    return IRDLOperation, irdl_op_definition


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Operands and Results

    Operands and results are added using fields containing `Operand` and `OpResult`, which each contain an attribute constraint. The order correspond to the operand and result order, and the constraint applies on the SSA variable type.

    Here is an example of an operation defining operands and a result:
    """
    )
    return


@app.cell
def _(IRDLOperation, IntegerAttr, i32, irdl_op_definition, printer):
    from typing import Annotated

    from xdsl.dialects.arith import ConstantOp
    from xdsl.ir import OpResult
    from xdsl.irdl import Operand, operand_def, result_def


    @irdl_op_definition
    class Addi32Op(IRDLOperation):
        name = "arith.addi32"

        # Definition of operands and results.
        input1 = operand_def(i32)
        input2 = operand_def(i32)
        output = result_def(i32)


    i32_ssa_var = ConstantOp(IntegerAttr.from_int_and_width(62, 32), i32)
    my_addi32 = Addi32Op.build(
        operands=[i32_ssa_var.result, i32_ssa_var.result], result_types=[i32]
    )
    printer.print_op(i32_ssa_var)
    print()
    printer.print_op(my_addi32)
    return (
        Addi32Op,
        ConstantOp,
        i32_ssa_var,
        my_addi32,
        operand_def,
        result_def,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The operation expects exactly the number of operands and results defined and checks that operands and results satisfy their invariants using the `verify` method.""")
    return


@app.cell
def _(my_addi32):
    # This will pass since the invariants are satisfied
    my_addi32.verify()
    return


@app.cell
def _(Addi32, i32, i32_ssa_var):
    # Wrong number of operands
    try:
        bad_addi32_a = Addi32.build(operands=[i32_ssa_var], result_types=[i32])
    except Exception as e:
        print(e)
    return


@app.cell
def _(Addi32Op, i32, i32_ssa_var):
    # Wrong number of results
    try:
        bad_addi32_b = Addi32Op.build(
            operands=[i32_ssa_var, i32_ssa_var], result_types=[i32, i32]
        )
    except Exception as e:
        print(e)
    return


@app.cell
def _(Addi32Op, i32_ssa_var, i64):
    # Wrong result type
    bad_addi32_c = Addi32Op.build(operands=[i32_ssa_var, i32_ssa_var], result_types=[i64])

    try:
        bad_addi32_c.verify()
    except Exception as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""As with `ParametrizedAttribute`, each operand and result definition defines accessors to easily access operands and results:""")
    return


@app.cell
def _(my_addi32):
    assert my_addi32.input1 == my_addi32.operands[0]
    assert my_addi32.input2 == my_addi32.operands[1]
    assert my_addi32.output == my_addi32.results[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Constraint Variables

    Constraint variables can directly be used in Operation and Attribute definitions by using a `TypeAlias` annotated with a `ConstraintVar`.
    """
    )
    return


@app.cell
def _(
    IRDLOperation,
    IntegerType,
    VarConstraint,
    i32,
    i32_ssa_var,
    i64,
    irdl_op_definition,
    operand_def,
    result_def,
):
    from typing import ClassVar
    from xdsl.irdl import base


    @irdl_op_definition
    class BinaryOp(IRDLOperation):
        name = "test.binary_op"

        T: ClassVar = VarConstraint("T", base(IntegerType))

        lhs = operand_def(T)
        rhs = operand_def(T)
        result = result_def(T)


    op = BinaryOp.build(operands=[i32_ssa_var, i32_ssa_var], result_types=[i32])
    op.verify()

    op_incorrect = BinaryOp.build(operands=[i32_ssa_var, i32_ssa_var], result_types=[i64])
    try:
        op_incorrect.verify()
    except Exception as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Variadic Operands and Results

    Operand and result definitions can be defined variadic, meaning that their definition can have different numbers of operands or results. Variadic definitions are defined with `var_operand_def` and `var_result_def`.
    """
    )
    return


@app.cell
def _(
    ConstantOp,
    IRDLOperation,
    IntegerAttr,
    i32,
    irdl_op_definition,
    printer,
    result_def,
):
    from xdsl.irdl import var_operand_def


    @irdl_op_definition
    class AddVariadicOp(IRDLOperation):
        name = "test.add_variadic"
        ops = var_operand_def(i32)
        res = result_def(i32)


    i32_ssa_var_b = ConstantOp(IntegerAttr.from_int_and_width(62, 32), i32)
    add_op = AddVariadicOp.build(operands=[[i32_ssa_var_b] * 3], result_types=[i32])
    printer.print_op(i32_ssa_var_b)
    print()
    printer.print_op(add_op)
    return add_op, var_operand_def


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Generated accessors return a list of SSA variables instead of a single variable:""")
    return


@app.cell
def _(add_op):
    print(len(add_op.ops))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Since it may be ambiguous, it is not possible to simply define two variadic operands, or two variadic results. To do so, the operation definition requires the `AttrSizedOperandSegments` or `AttrSizedResultSegments` IRDL option, which require the operation to contain a `operandSegmentSizes` or `resultSegmentSizes` attribute, containing the size of the variadic operands, and variadic results.""")
    return


@app.cell
def _(
    ConstantOp,
    IRDLOperation,
    IntegerAttr,
    i32,
    irdl_op_definition,
    result_def,
    var_operand_def,
):
    from xdsl.dialects.builtin import VectorType
    from xdsl.irdl import AttrSizedOperandSegments


    @irdl_op_definition
    class AddVariadic2Op(IRDLOperation):
        name = "test.add_variadic"
        ops1 = var_operand_def(i32)
        ops2 = var_operand_def(i32)
        res = result_def(i32)

        irdl_options = (AttrSizedOperandSegments(),)


    i32_ssa_var_c = ConstantOp(IntegerAttr.from_int_and_width(62, 32), i32)
    add_op2 = AddVariadic2Op.build(
        operands=[[i32_ssa_var_c] * 2, [i32_ssa_var_c]],
        result_types=[i32],
        attributes={"operandSegmentSizes": VectorType(i32, [2, 1])},
    )
    print("Length of add_op2.ops1:", len(add_op2.ops1))
    print("Length of add_op2.ops2:", len(add_op2.ops2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In addition to variadic operands and results, IRDL also allows the definition of optional operands and results. Optional operands and results are essentially variadic operands and regions that are constrained to have either 0 or 1 elements. They are defined using `OptOperand` and `OptResultDef`, and define accessors returning optional SSA variables.""")
    return


@app.cell
def _(
    ConstantOp,
    IRDLOperation,
    IntegerAttr,
    i32,
    irdl_op_definition,
    operand_def,
    result_def,
):
    from xdsl.irdl import OptOperand, opt_operand_def


    @irdl_op_definition
    class AddVariadic2Op2(IRDLOperation):
        name = "test.add_optional"
        ops1 = operand_def(i32)
        ops2 = opt_operand_def(i32)
        res = result_def(i32)


    i32_ssa_var_d = ConstantOp(IntegerAttr.from_int_and_width(62, 32), i32)
    add_op3 = AddVariadic2Op2.build(
        operands=[i32_ssa_var_d, [i32_ssa_var_d]], result_types=[i32]
    )
    print(add_op3.ops2)

    add_op4 = AddVariadic2Op2.build(operands=[i32_ssa_var_d, []], result_types=[i32])
    print(add_op4.ops2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Attributes Definition

    Attribute definitions are defined using `OpAttr`. The field name correspond to the expected attribute name.
    """
    )
    return


@app.cell
def _(IRDLOperation, StringAttr, irdl_op_definition, printer):
    from xdsl.irdl import attr_def


    @irdl_op_definition
    class StringAttrOp(IRDLOperation):
        name = "test.string_attr_op"
        value = attr_def(StringAttr)


    my_attr_op = StringAttrOp.build(attributes={"value": StringAttr("ga")})
    my_attr_op.verify()
    printer.print_op(my_attr_op)
    return (StringAttrOp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The defined verifier ensures that the attribute is present:""")
    return


@app.cell
def _(StringAttrOp):
    my_attr_op2 = StringAttrOp.build()
    try:
        my_attr_op2.verify()
    except Exception as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And expects that the attribute respect the constraint:""")
    return


@app.cell
def _(IntAttr, StringAttrOp):
    try:
        my_attr_op3 = StringAttrOp.build(attributes={"value": IntAttr(42)})
    except Exception as e:
        print(e)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Note that contrary to operands and results, other attributes may appear in an operation, even though they are not defined by the operation:""")
    return


@app.cell
def _(IntAttr, StringAttr, StringAttrOp, printer):
    my_attr_op4 = StringAttrOp.build(
        attributes={"value": StringAttr("ga"), "other_attr": IntAttr(42)}
    )
    my_attr_op4.verify()
    printer.print_op(my_attr_op4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Regions

    Regions definitions are defined using `Region` or `SingleBlockRegion` annotations. The second definition constrains the region to contain a single block, and both definitions allows to further constraint the region by giving a constraint for the entry basic block parameters.
    """
    )
    return


@app.cell
def _(IRDLOperation, i32, irdl_op_definition, printer):
    from xdsl.irdl import Block, Region, region_def, traits_def
    from xdsl.traits import NoTerminator


    @irdl_op_definition
    class WhileOp(IRDLOperation):
        name = "test.while_op"
        value = region_def()
        traits = traits_def(NoTerminator())


    region = Region(Block(arg_types=[i32]))
    region_op = WhileOp.build(regions=[region])
    region_op.verify()
    printer.print_op(region_op)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Additional Verifiers

    `irdl_op_definition` is not expressive enough to define arbitrary constraints, especially for constraints spanning over multiple operand and result definitions. To circumvent that, definitions may define a `verify_` method that will be called in the generated verifier:
    """
    )
    return


@app.cell
def _(
    ConstantOp,
    IRDLOperation,
    IntegerAttr,
    IntegerType,
    i32,
    i32_ssa_var,
    i64,
    irdl_op_definition,
    operand_def,
    result_def,
):
    from xdsl.dialects.arith import AddiOp


    @irdl_op_definition
    class MyAddiOp(IRDLOperation):
        name = "test.addi"
        input1 = operand_def(IntegerType)
        input2 = operand_def(IntegerType)
        output = result_def(IntegerType)

        # Ensure that the inputs and outputs have the same type:
        def verify_(self) -> None:
            if self.input1.type != self.input2.type or self.input2.type != self.output.type:
                raise Exception("expect all input and output types to be equal")


    i32_ssa_var_e = ConstantOp(IntegerAttr.from_int_and_width(62, 32), i32)
    add_op5 = AddiOp.build(operands=[i32_ssa_var_e, i32_ssa_var_e], result_types=[i32])
    # This will pass, since all operands and results have the same type
    add_op5.verify()

    # This will raise an error, since the result has a different type than the operands
    bad_add_op = AddiOp.build(operands=[i32_ssa_var, i32_ssa_var], result_types=[i64])
    try:
        bad_add_op.verify()
    except Exception as e:
        print(e)
    return


if __name__ == "__main__":
    app.run()
