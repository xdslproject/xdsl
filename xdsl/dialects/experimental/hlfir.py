"""
The High-Level Fortran IR (HLFIR) dialect that is used by Flang.

This is mixed with FIR, and provides a higher level view of Fortran
variables and some expressions. This means that temporaries need not
be materialised and there is a higher level of information about
the programmer's Fortran code, compared to FIR, available for
optimisation and lowering.

See external [documentation](https://flang.llvm.org/docs/HighLevelFIR.html).
"""

from __future__ import annotations

from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import (
    ArrayAttr,
    Attribute,
    BoolAttr,
    DenseArrayBase,
    IntAttr,
    IntegerAttr,
    ParametrizedAttribute,
    StringAttr,
    UnitAttr,
)
from xdsl.dialects.experimental.fir import (
    DeferredAttr,
    FortranVariableFlagsAttr,
    NoneType,
)
from xdsl.ir import Dialect, TypeAttribute
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator


@irdl_attr_definition
class ExprType(ParametrizedAttribute, TypeAttribute):
    """
    The type of an array, character, or derived type Fortran expression.

    Abstract value type for Fortran arrays, characters and derived types.
    The rank cannot be assumed, and empty shape means that the expression is a scalar.
    When the element type is a derived type, the polymorphic flag may be set to true
    to indicate that the expression dynamic type can differ from its static type.
    """

    name = "hlfir.expr"
    shape: ArrayAttr[IntegerAttr | DeferredAttr | NoneType]
    elementType: Attribute

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            for s in self.shape.data:
                if isinstance(s, DeferredAttr):
                    printer.print_string("?")
                elif isinstance(s, NoneType):
                    raise Exception(
                        "Can not have none type as part of sequence shape with only one type"
                    )
                else:
                    printer.print_int(s.value.data)
                printer.print_string("x")
            printer.print_attribute(self.elementType)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        # We need extra work here as the builtin tuple is not being supported
        # yet, therefore handle this here
        def parse_interval() -> IntegerAttr | DeferredAttr:
            if parser.parse_optional_punctuation("?"):
                return DeferredAttr()
            s = parser.parse_integer(allow_boolean=False)
            return IntegerAttr(s, 32)

        shape: list[IntegerAttr | DeferredAttr] = []
        parser.parse_characters("<")
        elementType = parser.parse_optional_type()
        while elementType is None:
            shape.append(parse_interval())
            parser.parse_shape_delimiter()
            elementType = parser.parse_optional_type()
        parser.parse_characters(">")
        return [ArrayAttr(shape), elementType]


@irdl_op_definition
class DeclareOp(IRDLOperation):
    """
    Declare a variable and produce an SSA value that can be used as a variable in HLFIR operations.

    Tie the properties of a Fortran variable to an address. The properties
    include bounds, length parameters, and Fortran attributes.

    The arguments are the same as for fir.declare.

    The main difference with fir.declare is that hlfir.declare returns two
    values:
      - the first one is an SSA value that allows retrieving the variable
        address, bounds, and type parameters at any point without requiring
        access to the defining operation. This may be:
        - for scalar numerical, logical, or derived type without length
          parameters: a fir.ref<T> (e.g. fir.ref<i32>)
        - for scalar characters: a fir.boxchar<kind> or fir.ref<fir.char<kind,
          cst_len>>
        - for arrays of types without length parameters, without lower bounds,
          that are not polymorphic and with a constant shape:
          fir.ref<fir.array<cst_shapexT>>
        - for all non pointer/non allocatable entities: fir.box<T>, and
          fir.class<T> for polymorphic entities.
        - for all pointers/allocatables:
          fir.ref<fir.box<fir.ptr<T>>>/fir.ref<fir.box<fir.heap<T>>>
      - the second value has the same type as the input memref, and is the
        same. If it is a fir.box or fir.class, it may not contain accurate
        local lower bound values. It is intended to be used when generating FIR
        from HLFIR in order to avoid descriptor creation for simple entities.

    Example:
    CHARACTER(n) :: c(10:n, 20:n)

    Can be represented as:
    func.func @foo(%arg0: !fir.ref<!fir.array<?x?x!fir.char<1,?>>>, %arg1: !fir.ref<i64>) {
      %c10 = arith.constant 10 : index
      %c20 = arith.constant 20 : index
      %1 = fir.load %ag1 : fir.ref<i64>
      %2 = fir.shape_shift %c10, %1, %c20, %1 : (index, index, index, index) -> !fir.shapeshift<2>
      %3 = hfir.declare %arg0(%2) typeparams %1 {uniq_name = "c"} (fir.ref<!fir.array<?x?x!fir.char<1,?>>>, fir.shapeshift<2>, index) -> (fir.box<!fir.array<?x?x!fir.char<1,?>>>, fir.ref<!fir.array<?x?x!fir.char<1,?>>>)
      // ... uses %3#0 as "c"
    }
    """  # noqa: E501

    name = "hlfir.declare"
    memref = operand_def()
    shape = opt_operand_def()
    typeparams = var_operand_def()
    dummy_scope = opt_operand_def()
    uniq_name = opt_prop_def(StringAttr)
    fortran_attrs = opt_prop_def(FortranVariableFlagsAttr)
    result = result_def()
    result2 = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class DesignateOp(IRDLOperation):
    """
    Designate a Fortran variable.

    This operation represents a Fortran "part-ref", except that it can embed a
    substring or or complex part directly, and that vector subscripts cannot be
    used. It returns a Fortran variable that is a part of the input variable.

    The operands are as follow:
      - memref is the variable being designated.
      - component may be provided if the memref is a derived type to
        represent a reference to a component. It must be the name of a
        component of memref derived type.
      - component_shape represents the shape of the component and must be
        provided if and only if both component and indices appear.
      - indices can be provided to index arrays. The indices may be simple
        indices or triplets.
        If indices are provided and there is a component, the component must be
        an array component and the indices index the array component.
        If memref is an array, and component is provided and is an array
        component, indices must be provided and must not be triplets. This
        ensures hlfir.designate does not create arrays of arrays (which is not
        possible in Fortran).
      - substring may contain two values to represent a substring lower and
        upper bounds.
      - complex_part may be provided to represent a complex part (true
        represents the imaginary part, and false the real part).
      - shape represents the shape of the result and must be provided if the
        result is an array that is not a box address.
      - typeparams represents the length parameters of the result and must be
        provided if the result type has length parameters and is not a box
        address.
    """

    name = "hlfir.designate"
    memref = operand_def()
    component_shape = opt_operand_def()
    indices = var_operand_def()
    substring = var_operand_def()
    shape = opt_operand_def()
    typeparams = var_operand_def()
    component = opt_prop_def(StringAttr)
    complex_part = opt_prop_def(BoolAttr)
    is_triplet = prop_def(DenseArrayBase)
    fortran_attrs = opt_prop_def(FortranVariableFlagsAttr)
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class AssignOp(IRDLOperation):
    """
    Assign an expression or variable value to a Fortran variable.

    Assign rhs to lhs following Fortran intrinsic assignments rules.
    The operation deals with inserting a temporary if the lhs and rhs
    may overlap.

    The optional "realloc" flag allows indicating that this assignment
    has the Fortran 95 semantics for assignments to a whole allocatable.
    In such case, the left hand side must be an allocatable that may be
    unallocated or allocated with a different type and shape than the right
    hand side. It will be allocated or re-allocated as needed during the
    assignment.

    When "realloc" is set and this is a character assignment, the optional
    flag "keep_lhs_length_if_realloc" indicates that the character
    left hand side should retain its length after the assignment. If the
    right hand side has a different length, truncation and padding will
    occur. This covers the case of explicit and assumed length character
    allocatables.

    Otherwise, the left hand side will be allocated or reallocated to match the
    right hand side length if they differ. This covers the case of deferred
    length character allocatables.

    The optional "temporary_lhs" flag indicates that the LHS is a compiler
    generated temporary. In this case the temporary is initialized if needed
    (e.g. the LHS is of derived type with allocatable/pointer components),
    and the assignment is done without LHS (or its subobjects) finalization
    and with automatic allocation.

    If "temporary_lhs" and "keep_lhs_length_if_realloc" are both set,
    this assign operation denotes special case of character allocatable
    LHS with explicit length. The LHS that must preserve its length
    during the assignment regardless of the the RHS's length or/and
    allocation status. This assign operation will be lowered into a call
    to AssignExplicitLengthCharacter().
    """

    name = "hlfir.assign"
    lhs = operand_def()
    rhs = operand_def()
    realloc = opt_prop_def(UnitAttr)
    keep_lhs_length_if_realloc = opt_prop_def(UnitAttr)
    temporary_lhs = opt_prop_def(UnitAttr)


@irdl_op_definition
class ParentComponentOp(IRDLOperation):
    """
    Designate the parent component of a variable.

    This operation represents a Fortran component reference where the
    component name is a parent type of the variable's derived type.
    These component references cannot be represented with an hlfir.designate
    because the parent type names are not embedded in fir.type<> types
    as opposed to the actual component names.

    The operands are as follow:
      - memref is a derived type variable whose parent component is being
        designated.
      - shape is the shape of memref and the result and must be provided if
        memref is an array. Parent component reference lower bounds are ones,
        so the provided shape must be a fir.shape.
      - typeparams are the type parameters of the parent component type if any.
        It is a subset of memref type parameters.
    The parent component type and name is reflected in the result type.
    """

    name = "hlfir.parent_comp"
    memref = operand_def()
    shape = opt_operand_def()
    typeparams = var_operand_def()
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class ConcatOp(IRDLOperation):
    """
    Concatenate characters.

    Concatenate two or more character strings of a same character kind.
    """

    name = "hlfir.concat"
    strings = var_operand_def()
    length = operand_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class AllOp(IRDLOperation):
    """
    ALL transformational intrinsic.

    Takes a logical array MASK as argument, optionally along a particular dimension,
    and returns true if all elements of MASK are true.
    """

    name = "hlfir.all"
    mask = operand_def()
    dim = opt_operand_def()
    result = result_def()


@irdl_op_definition
class AnyOp(IRDLOperation):
    """
    ANY transformational intrinsic.

    Takes a logical array MASK as argument, optionally along a particular dimension,
    and returns true if any element of MASK is true.
    """

    name = "hlfir.any"
    mask = operand_def()
    dim = opt_operand_def()
    result = result_def()


@irdl_op_definition
class CountOp(IRDLOperation):
    """
    COUNT transformational intrinsic.

    Takes a logical and counts the number of true values.
    """

    name = "hlfir.count"
    mask = operand_def()
    dim = opt_operand_def()
    result = result_def()


@irdl_op_definition
class MaxvalOp(IRDLOperation):
    """
    MAXVAL transformational intrinsic.

    Maximum value(s) of an array.
    If DIM is absent, the result is a scalar.
    If DIM is present, the result is an array of rank n-1, where n is the rank of ARRAY.
    """

    name = "hlfir.maxval"
    array = operand_def()
    dim = opt_operand_def()
    mask = opt_operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class MinvalOp(IRDLOperation):
    """
    MINVAL transformational intrinsic.

    Minimum value(s) of an array.
    If DIM is absent, the result is a scalar.
    If DIM is present, the result is an array of rank n-1, where n is the rank of ARRAY.
    """

    name = "hlfir.minval"
    array = operand_def()
    dim = opt_operand_def()
    mask = opt_operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class ProductOp(IRDLOperation):
    """
    PRODUCT transformational intrinsic.

    Multiplies the elements of an array, optionally along a particular dimension,
    optionally if a mask is true.
    """

    name = "hlfir.product"
    array = operand_def()
    dim = opt_operand_def()
    mask = opt_operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class SetLengthOp(IRDLOperation):
    """
    Change the length of a character entity.

    This trims or pads the
    character argument according to the new length.
    """

    name = "hlfir.set_length"
    string = operand_def()
    length = operand_def()
    result = result_def()


@irdl_op_definition
class GetLengthOp(IRDLOperation):
    """
    Get the length of a character entity.

    Get the length of character entity represented as hlfir.expr.
    """

    name = "hlfir.get_length"
    expr = operand_def()
    result = result_def()


@irdl_op_definition
class SumOp(IRDLOperation):
    """
    SUM transformational intrinsic.

    Sums the elements of an array, optionally along a particular dimension,
    optionally if a mask is true.
    """

    name = "hlfir.sum"
    array = operand_def()
    dim = opt_operand_def()
    mask = opt_operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class DotProductOp(IRDLOperation):
    """
    DOT_PRODUCT transformational intrinsic,

    Dot product of two vectors,
    """

    name = "hlfir.dot_product"
    lhs = operand_def()
    rhs = operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()


@irdl_op_definition
class MatmulOp(IRDLOperation):
    """
    MATMUL transformational intrinsic.

    Matrix multiplication.
    """

    name = "hlfir.matmul"
    lhs = operand_def()
    rhs = operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()


@irdl_op_definition
class TransposeOp(IRDLOperation):
    """
    TRANSPOSE transformational intrinsic.

    Transpose a rank 2 array.
    """

    name = "hlfir.transpose"
    array = operand_def()
    result = result_def()


@irdl_op_definition
class MatmulTransposeOp(IRDLOperation):
    """
    Optimized MATMUL(TRANSPOSE(...), ...).

    Matrix multiplication where the left hand side is transposed.
    """

    name = "hlfir.matmul_transpose"
    lhs = operand_def()
    rhs = operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()


@irdl_op_definition
class AssociateOp(IRDLOperation):
    """
    Create a variable from an expression value.

    For expressions, this operation is an incentive to re-use the expression
    storage, if any, after the bufferization pass when possible (if the
    expression is not used afterwards).
    """

    name = "hlfir.associate"
    source = operand_def()
    shape = opt_operand_def()
    typeparams = var_operand_def()
    uniq_name = opt_prop_def(StringAttr)
    fortran_attrs = opt_prop_def(FortranVariableFlagsAttr)
    result = var_result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class EndAssociateOp(IRDLOperation):
    """
    Mark the end of life of a variable associated to an expression.

    If the expression has a derived type that may contain allocatable
    components, the variable operand must be a Fortran entity.
    """

    name = "hlfir.end_associate"
    var = operand_def()
    must_free = operand_def()


@irdl_op_definition
class AsExprOp(IRDLOperation):
    """
    Take the value of an array, character or derived variable.

    In general, this operation will lead to a copy of the variable
    in the bufferization pass if it was not transformed.

    However, if it is known that the variable storage will not be used anymore
    afterwards, the variable storage ownership can be passed to the hlfir.expr
    by providing the $must_free argument that is a boolean that indicates if
    the storage must be freed (when it was allocated on the heap).
    This allows Fortran lowering to build some expression value in memory when
    there is no adequate hlfir operation, and to promote the result to an
    hlfir.expr value without paying the price of introducing a copy.
    """

    name = "hlfir.as_expr"
    var = operand_def()
    must_free = opt_operand_def()
    result = result_def()


@irdl_op_definition
class NoReassocOp(IRDLOperation):
    """
    Synthetic op to prevent reassociation.

    Same as fir.reassoc, except it accepts hlfir.expr arguments.
    """

    name = "hlfir.no_reassoc"
    val = operand_def()
    result = result_def()


@irdl_op_definition
class ElementalOp(IRDLOperation):
    """
    Elemental expression.

    Represent an elemental expression as a function of the indices.
    This operation contain a region whose block arguments are one
    based indices iterating over the elemental expression shape.
    Given these indices, the element value for the given iteration
    can be computed in the region and yielded with the hlfir.yield_element
    operation.

    The shape and typeparams operands represent the extents and type
    parameters of the resulting array value.

    The optional mold is an entity carrying the information about
    the dynamic type of the polymorphic result. Note that the shape
    of the mold does not necessarily match the shape of the result,
    for example, the result of `merge(poly_scalar1, poly_scalar2, mask_array)`
    will have the shape of `mask_array` and the dynamic type of `poly_scalar*`.

    The unordered attribute can be set to allow out of order processing
    of the indices. This is safe only if the operations in the body
    of the elemental do not have side effects.


    Example: Y + X,  with Integer :: X(10, 20), Y(10,20)

    %0 = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
    %5 = hlfir.elemental %0 : (!fir.shape<2>) -> !hlfir.expr<10x20xi32> {
    ^bb0(%i: index, %j: index):
      %6 = hlfir.designate %x (%i, %j)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
      %7 = hlfir.designate %y (%i, %j)  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
      %8 = fir.load %6 : !fir.ref<i32>
      %9 = fir.load %7 : !fir.ref<i32>
      %10 = arith.addi %8, %9 : i32
      hlfir.yield_element %10 : i32
    }
    """

    name = "hlfir.elemental"
    shape = operand_def()
    mold = opt_operand_def()
    typeparams = var_operand_def()
    unordered = opt_prop_def(UnitAttr)
    regs = var_region_def()
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class YieldElementOp(IRDLOperation):
    """
    Yield the elemental value in an ElementalOp.

    Yield the element value of the current elemental expression iteration
    in an hlfir.elemental region. See hlfir.elemental description for an
    example.
    """

    name = "hlfir.yield_element"
    element_value = operand_def()

    traits = traits_def(IsTerminator())


@irdl_op_definition
class ApplyOp(IRDLOperation):
    """
    Get the element value of an expression.

    Given an hlfir.expr array value, hlfir.apply allow retrieving
    the value for an element given one based indices.

    When hlfir.apply is used on an hlfir.elemental, and if the hlfir.elemental
    operation evaluation can be moved to the location of the hlfir.apply, it is
    as if the hlfir.elemental body was evaluated given the hlfir.apply indices.
    Therefore, apply operations on hlfir.elemental expressions should be located
    such that evaluating the hlfir.elemental at the position of the hlfir.apply
    operation produces the same result as evaluating the hlfir.elemental at its
    location in the instruction stream. Attention should be paid to
    hlfir.elemental memory side effects (in practice these are unlikely).
    "10.1.4 Evaluation of operations" says that expression evaluation shall not
    impact/be impacted by other expression evaluation in the statement.
    """

    name = "hlfir.apply"
    expr = operand_def()
    indices = var_operand_def()
    typeparams = var_operand_def()
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class NullOp(IRDLOperation):
    """
    Create a NULL() address.

    So far is not intended to represent NULL(MOLD).
    """

    name = "hlfir.null"
    result = result_def()


@irdl_op_definition
class DestroyOp(IRDLOperation):
    """
    Mark the last use of an hlfir.expr.

    This will be the point at which the
    buffer of an hlfir.expr, if any, will be deallocated if it was heap
    allocated.
    If "finalize" attribute is set, the hlfir.expr value will be finalized
    before the deallocation. Note that this implies that the hlfir.expr
    is placed into a memory buffer, so that the library runtime
    can be called on it. The element type of the hlfir.expr must be
    derived type in this case.
    It is not required to create an hlfir.destroy operation for and hlfir.expr
    created inside an hlfir.elemental and returned in the hlfir.yield_element.
    The last use of such expression is implicit and an hlfir.destroy could
    not be emitted after the hlfir.yield_element since it is a terminator.

    Note that hlfir.destroy are currently generated by Fortran lowering that
    has a good view of the expression use contexts, but this will need to be
    revisited if any motion of hlfir.expr is done (like CSE) since
    transformations should not introduce any hlfir.expr usages after an
    hlfir.destroy.
    The future will probably be to identify the last use points automatically
    in bufferization instead.
    """

    name = "hlfir.destroy"
    expr = operand_def()
    finalize = opt_prop_def(UnitAttr)


@irdl_op_definition
class CopyInOp(IRDLOperation):
    """
    Copy a variable into a contiguous temporary if it is not contiguous.

    Copy a variable into a contiguous temporary if the variable is not
    an absent optional and is not contiguous at runtime. When a copy is made this
    operation returns the temporary as first result, otherwise, it returns the
    potentially absent variable storage. The second result indicates if a copy
    was made.

    This operation is meant to be used in combination with the hlfir.copy_out
    operation that deletes the temporary if it was created and copies the data
    back if needed.
    This operation allows passing non contiguous arrays to contiguous dummy
    arguments, which is possible in Fortran procedure references.

    To deal with the optional case, an extra boolean value can be pass to the
    operation. In such cases, the copy-in will only be done if "var_is_present"
    is true and, when it is false, the original value will be returned instead.
    """

    name = "hlfir.copy_in"
    var = operand_def()
    tempBox = operand_def()
    var_is_present = opt_operand_def()
    result = var_result_def()


@irdl_op_definition
class CopyOutOp(IRDLOperation):
    """
    Copy out a variable after a copy in.

    If the variable was copied in a temporary in the related hlfir.copy_in,
    optionally copy back the temporary value to it (that may have been
    modified between the hlfir.copy_in and hlfir.copy_out). Then deallocate
    the temporary.
    The copy back is done if $var is provided and $was_copied is true.
    The deallocation of $temp is done if $was_copied is true.
    """

    name = "hlfir.copy_out"
    temp = operand_def()
    was_copied = operand_def()
    var = opt_operand_def()


@irdl_op_definition
class ShapeOfOp(IRDLOperation):
    """
    Get the shape of a hlfir.expr.

    Gets the runtime shape of a hlfir.expr. In lowering to FIR, the
    hlfir.shape_of operation will be replaced by an fir.shape.
    It is not valid to request the shape of a hlfir.expr which has no shape.
    """

    name = "hlfir.shape_of"
    expr = operand_def()
    result = result_def()


@irdl_op_definition
class GetExtentOp(IRDLOperation):
    """
    Gets an extent value from a fir.shape.

    The dimension argument uses C style
    indexing and so should be between 0 and 1 less than the rank of the shape
    """

    name = "hlfir.get_extent"
    shape = operand_def()
    dim = prop_def(IntAttr)
    result = result_def()


@irdl_op_definition
class RegionAssignOp(IRDLOperation):
    """
    Represent a Fortran assignment using regions for the LHS and RHS evaluation.

    This operation can represent Forall and Where assignment when inside an
    hlfir.forall or hlfir.where "ordered assignment tree". It can
    also represent user defined assignments and assignment to vector
    subscripted entities without requiring the materialization of the
    right-hand side temporary copy that may be needed to implement Fortran
    assignment semantic.

    The right-hand side and left-hand side evaluations are held in their
    own regions terminated with hlfir.yield operations (or hlfir.elemental_addr
    for a left-hand side with vector subscript).

    An optional region may be added to implement user defined assignment.
    This region provides two block arguments with the same type as the
    yielded rhs and lhs entities (in that order), or the element type if this
    is an elemental user defined assignment.

    If this optional region is not provided, intrinsic assignment is performed.

    Example: "X = Y",  where "=" is a user defined elemental assignment "foo"
    taking Y by value.

    hlfir.region_assign {
      hlfir.yield %y : !fir.box<!fir.array<?x!f32>>
    } to {
      hlfir.yield %x : !fir.box<!fir.array<?x!fir.type<t>>>
    } user_defined_assignment (%rhs_elt: !fir.ref<f32>) to (%lhs_elt: !fir.ref<!fir.type<t>>) {
      %0 = fir.load %rhs_elt : !fir.ref<f32>
      fir.call @foo(%lhs_elt, %0) : (!fir.ref<!fir.type<t>>, f32) -> ()
    }

    TODO: add optional "realloc" semantics like for hlfir.assign.
    """

    name = "hlfir.region_assign"
    rhs_region = region_def()
    lhs_region = region_def()
    user_defined_assignment = region_def()


@irdl_op_definition
class YieldOp(IRDLOperation):
    """
    Yield a value or variable inside a forall, where or region assignment.

    Terminator operation that yields an HLFIR value or variable that was computed in
    a region and hold the yielded entity cleanup, if any, into its own region.
    This allows representing any Fortran expression evaluation in its own region so
    that the evaluation can easily be scheduled/moved around in a pass.

    Example: "foo(x)" where foo returns an allocatable array.
    {
      // In some region.
      %0 = fir.call @foo(x) (!fir.ref<f32>) -> !fir.box<fir.heap<!fir.array<?xf32>>>
      hlfir.yield %0 : !fir.box<!fir.heap<!fir.array<?xf32>>> cleanup {
        %1 = fir.box_addr %0 : !fir.box<!fir.heap<!fir.array<?xf32>>> -> !fir.heap<!fir.array<?xf32>>
        %fir.freemem %1 : !fir.heap<!fir.array<?xf32>>
      }
    }
    """

    name = "hlfir.yield"
    entity = operand_def()
    cleanup = region_def()

    traits = traits_def(IsTerminator())


@irdl_op_definition
class ElementalAddrOp(IRDLOperation):
    """
    Yield the address of a vector subscripted variable inside an hlfir.region_assign.

    Special terminator node for the left-hand side region of an hlfir.region_assign
    to a vector subscripted entity.

    It represents how the address of an element of such entity is computed given
    one based indices.

    It is very similar to hlfir.elemental, except that it does not produce an SSA
    value because there is no hlfir type to describe a vector subscripted entity
    (the codegen of such type would be problematic). Hence, it is tightly linked
    to an hlfir.region_assign by its terminator property.

    An optional cleanup region may be provided if any of the subscript expressions
    of the designator require a cleanup.
    This allows documenting cleanups that cannot be generated after the vector
    subscripted designator usage (that has not been materizaled yet). The cleanups
    will be evaluated after the assignment once the related
    hlfir.region_assign is lowered.

    Example: "X(VECTOR) = Y"

    hlfir.region_assign {
      hlfir.yield %y : !fir.ref<!fir.array<20xf32>>
    } to {
      hlfir.elemental_addr %vector_shape  : !fir.shape<1> {
        ^bb0(%i: index):
        %0 = hlfir.designate %vector (%i)  : (!fir.ref<!fir.array<20xi32>>, index) -> !fir.ref<i32>
        %1 = fir.load %0 : !fir.ref<i32>
        %x_element_addr = hlfir.designate %x (%1)  : (!fir.ref<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
        hlfir.yield %x_element_addr : !fir.ref<f32>
      }
    }
    """

    name = "hlfir.elemental_addr"
    shape = operand_def()
    mold = opt_operand_def()
    typeparams = var_operand_def()
    unordered = opt_prop_def(UnitAttr)
    body = region_def()
    cleanup = region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class ForallOp(IRDLOperation):
    """
    Represent a Fortran forall.

    This operation allows representing Fortran forall. It computes
    a set of "index-name" values based on lower bound, upper bound,
    and step values whose evaluations are represented in their own
    regions.

    Operations nested in its body region are evaluated in order.
    As opposed to a regular loop, each nested operation is
    fully evaluated for all the values in the "active set of
    index-name" before the next nested operation. In practice, the
    nested operation evaluation may be fused if it is proven that
    they do not have data dependency.

    The "index-name" value is represented as the argument of the
    body region.

    The lower, upper, and step region (if provided), must be terminated
    by hlfir.yield that yields scalar integers.

    The body region must only contain other OrderedAssignmentTreeOpInterface
    operations (like hlfir.region_assign, or other hlfir.forall).

    A Fortran forall with several indices is represented as a nest
    of hlfir.forall.

    All the regions contained in the hlfir.forall must only contain
    code that is pure from a Fortran point of view, except for the
    assignment effect of the hlfir.region_assign.
    This matches Fortran constraint C1037, but requires the outer
    controls to be evaluated outside of the hlfir.forall (these
    controls may have side effects as per Fortran 2018 10.1.4 section).

    Example: FORALL(I=1:10) X(I) = FOO(I)
    hlfir.forall lb {
      hlfir.yield %c1 : index
    } ub {
      hlfir.yield %c10 : index
    } (%i : index) {
      hlfir.region_assign {
        %res = fir.call @foo(%i) : (index) -> f32
        hlfir.yield %res : f32
      } to {
        %xi = hlfir.designate %x(%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
        hlfir.yield %xi : !fir.ref<f32>
      }
    }
    """

    name = "hlfir.forall"
    lb_region = region_def()
    ub_region = region_def()
    step_region = region_def()
    body = region_def()


@irdl_op_definition
class ForallMaskOp(IRDLOperation):
    """
    Fortran Forall can have a scalar mask expression that depends on the
    Forall index-name value.
    hlfir.forall_mask allows representing this mask. The expression
    evaluation is held in the mask region that must yield an i1 scalar
    value.
    An hlfir.forall_mask must be directly nested in the body region of
    an hlfir.forall. It is a separate operation so that it can use the
    index SSA value defined by the hlfir.forall body region.

    Example: "FORALL(I=1:10, SOME_CONDITION(I)) X(I) = FOO(I)"
    hlfir.forall lb {
      hlfir.yield %c1 : index
    } ub {
      hlfir.yield %c10 : index
    } (%i : index) {
      hlfir.forall_mask {
        %mask = fir.call @some_condition(%i) : (index) -> i1
        hlfir.yield %mask : i1
      } do {
        hlfir.region_assign {
          %res = fir.call @foo(%i) : (index) -> f32
          hlfir.yield %res : f32
        } to {
          %xi = hlfir.designate %x(%i) : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          hlfir.yield %xi : !fir.ref<f32>
        }
      }
    }
    """

    name = "hlfir.forall_mask"
    mask_region = region_def()
    body = region_def()


@irdl_op_definition
class AssignmentMaskOp(IRDLOperation):
    """
    Represent Fortran "where" construct or statement. The mask
    expression evaluation is held in the mask region that must yield
    logical array that has the same shape as all the nested
    hlfir.region_assign left-hand sides, and all the nested hlfir.where
    or hlfir.elsewhere masks.

    The values of the where and elsewhere masks form a control mask that
    controls all the nested hlfir.region_assign: only the array element for
    which the related control mask value is true are assigned. Any right-hand
    side elemental expression is only evaluated for elements where the control
    mask is true. See Fortran standard 2018 section 10.2.3 for more detailed
    about the control mask semantic.

    An hlfir.where must not contain any hlfir.forall but it may be contained
    in such operation. This matches Fortran rules.
    """

    name = "hlfir.where"
    mask_region = region_def()
    body = region_def()


@irdl_op_definition
class ElseWhereOp(IRDLOperation):
    """
    Represent Fortran "elsewhere" construct or statement.

    It has an optional mask region to hold the evaluation of Fortran
    optional elsewhere mask expressions. If this region is provided,
    it must satisfy the same constraints as hlfir.where mask region.

    An hlfir.elsewhere must be the last operation of an hlfir.where or,
    hlfir.elsewhere body, which is enforced by its terminator property.

    Like in Fortran, an hlfir.elsewhere negate the current control mask,
    and if provided, adds the mask the resulting control mask (with a logical
    AND).
    """

    name = "hlfir.elsewhere"
    mask_region = region_def()
    body = region_def()


@irdl_op_definition
class ForallIndexOp(IRDLOperation):
    """
    Represent a Fortran forall index declaration.

    This operation allows placing an hlfir.forall index in memory with
    the related Fortran index-value name and type.

    So far, lowering needs to manipulate symbols as memory entities.
    This operation allows fulfilling this requirements without allowing
    bare alloca/declare/store inside the body of hlfir.forall, which would
    make their analysis more complex.

    Given Forall index-value cannot be modified it also allows defining
    a canonicalization of all its loads into a fir.convert of the
    hlfir.forall index, which helps simplifying the data dependency analysis
    of hlfir.forall.
    """

    name = "hlfir.forall_index"
    index = operand_def()
    indexname = prop_def(StringAttr)
    result = result_def()


@irdl_op_definition
class CharExtremumOp(IRDLOperation):
    """
    Find max/min from given character strings.

    Find the lexicographical minimum or maximum of two or more character
    strings of the same character kind and return the string with the lexicographical
    minimum or maximum number of characters. Example:

    %0 = hlfir.char_extremum min, %arg0, %arg1 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>) -> !hlfir.expr<!fir.char<1,10>>
    """  # noqa E501

    name = "hlfir.char_extremum"
    predicate = operand_def()
    strings = var_operand_def()
    result = result_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


HLFIR = Dialect(
    "hlfir",
    [
        AllOp,
        AnyOp,
        ApplyOp,
        AsExprOp,
        AssignmentMaskOp,
        AssignOp,
        AssociateOp,
        CharExtremumOp,
        ConcatOp,
        CopyInOp,
        CopyOutOp,
        CountOp,
        DeclareOp,
        DesignateOp,
        DestroyOp,
        DotProductOp,
        ElementalAddrOp,
        ElementalOp,
        ElseWhereOp,
        EndAssociateOp,
        ForallIndexOp,
        ForallMaskOp,
        ForallOp,
        GetExtentOp,
        GetLengthOp,
        MatmulOp,
        MatmulTransposeOp,
        MaxvalOp,
        MinvalOp,
        NoReassocOp,
        NullOp,
        ParentComponentOp,
        ProductOp,
        RegionAssignOp,
        SetLengthOp,
        ShapeOfOp,
        SumOp,
        TransposeOp,
        YieldElementOp,
        YieldOp,
    ],
    [
        ExprType,
    ],
)
