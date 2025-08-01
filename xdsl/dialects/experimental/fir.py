"""
The Fortran IR (FIR) dialect that is used by Flang.

This is used in two ways, firstly it is mixed with HLFIR, and this
FIR+HLFIR is the first MLIR representation of a Fortran code in the
compilation pipeline. Secondly, the HLFIR+FIR is then lowered to FIR
only, before this is then lowered to LLVM IR.

See external [documentation](https://flang.llvm.org/docs/FortranIR.html).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from xdsl.dialects.arith import FastMathFlagsAttr
from xdsl.dialects.builtin import (
    AnyFloat,
    ArrayAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    SymbolNameConstraint,
    SymbolRefAttr,
    UnitAttr,
)
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator, SymbolOpInterface


class FortranVariableFlags(Enum):
    NOATTRIBUTES = (
        "None"  # First character is meant to be capitalised unlike the others
    )
    ALLOCATABLE = "allocatable"
    ASYNCHRONOUS = "asynchronous"
    BIND_C = "bind_c"
    CONTIGUOUS = "contiguous"
    INTENT_IN = "intent_in"
    INTENT_INOUT = "intent_inout"
    INTENT_OUT = "intent_out"
    OPTIONAL = "optional"
    PARAMETER = "parameter"
    POINTER = "pointer"
    TARGET = "target"
    VALUE = "value"
    VOLATILE = "volatile"
    HOSTASSOC = "host_assoc"
    INTERNALASSOC = "internal_assoc"

    @staticmethod
    def try_parse(parser: AttrParser) -> set[FortranVariableFlags] | None:
        for option in FortranVariableFlags:
            if parser.parse_optional_characters(option.value) is not None:
                return {option}

        return None


@dataclass(frozen=True)
class FortranVariableFlagsAttrBase(Data[tuple[FortranVariableFlags, ...]]):
    @property
    def flags(self) -> set[FortranVariableFlags]:
        """
        Returns a copy of the Fortran variable flags.
        """
        return set(self.data)

    def __init__(self, flags: Sequence[FortranVariableFlags]):
        flags_: set[FortranVariableFlags] = set(flags)

        super().__init__(tuple(flags_))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> tuple[FortranVariableFlags, ...]:
        with parser.in_angle_brackets():
            flags = FortranVariableFlags.try_parse(parser)
            if flags is None:
                return tuple()

            while parser.parse_optional_punctuation(",") is not None:
                flag = parser.expect(
                    lambda: FortranVariableFlags.try_parse(parser),
                    "fortran variable flag expected",
                )
                flags.update(flag)

            return tuple(flags)

    def print_parameter(self, printer: Printer):
        with printer.in_angle_brackets():
            flags = self.data
            # make sure we emit flags in a consistent order
            printer.print_list(
                tuple(flag.value for flag in FortranVariableFlags if flag in flags),
                printer.print_string,
                ",",
            )


@irdl_attr_definition
class FortranVariableFlagsAttr(FortranVariableFlagsAttrBase):
    name = "fir.var_attrs"


@irdl_attr_definition
class ReferenceType(ParametrizedAttribute, TypeAttribute):
    """
    The type of a reference to an entity in memory.
    """

    name = "fir.ref"
    type: Attribute


@irdl_attr_definition
class DeferredAttr(ParametrizedAttribute, TypeAttribute):
    """
    A deferred size which is represented with a question mark
    """

    name = "fir.deferred"

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("?")


@irdl_attr_definition
class DummyScopeType(ParametrizedAttribute, TypeAttribute):
    """
    fir.dscope is a type returned by fir.dummy_scope operation.
    It defines a unique identifier for a runtime instance of a subroutine
    that is used by the [hl]fir.declare operations representing
    the dummy arguments' declarations.
    """

    name = "fir.dscope"


@irdl_attr_definition
class LLVMPointerType(ParametrizedAttribute, TypeAttribute):
    """
    A pointer type that does not have any of the constraints and semantics
    of other FIR pointer types and that translates to llvm pointer types.
    It is meant to implement indirection that cannot be expressed directly
    in Fortran, but are needed to implement some Fortran features (e.g,
    double indirections).
    """

    name = "fir.llvm_ptr"

    type: Attribute


@irdl_attr_definition
class PointerType(ParametrizedAttribute, TypeAttribute):
    """
    The type of entities with the POINTER attribute.  These pointers are
    explicitly distinguished to disallow the composition of multiple levels of
    indirection. For example, an ALLOCATABLE POINTER is invalid.
    """

    name = "fir.ptr"

    type: Attribute


@irdl_attr_definition
class NoneType(ParametrizedAttribute, TypeAttribute):
    """
    This isn't part of the FIR MLIR dialect, and is only represented internally with
    xDSL, but is useful to denote when there is a none or empty attribute
    """

    name = "fir.none"


@irdl_attr_definition
class SequenceType(ParametrizedAttribute, TypeAttribute):
    """
    A sequence type is a multi-dimensional array of values. The sequence type
    may have an unknown number of dimensions or the extent of dimensions may be
    unknown. A sequence type models a Fortran array entity, giving it a type in
    FIR. A sequence type is assumed to be stored in a column-major order, which
    differs from LLVM IR and other dialects of MLIR.
    """

    name = "fir.array"
    shape: ArrayAttr[IntegerAttr | DeferredAttr | NoneType]
    type: Attribute
    type2: Attribute

    def __init__(
        self,
        type1: IntegerType | AnyFloat | ReferenceType,
        shape: list[int | IntegerAttr[IndexType] | DeferredAttr] | None = None,
        type2: IntegerType | AnyFloat | ReferenceType | None = None,
    ):
        if type2 is not None:
            super().__init__(ArrayAttr([NoneType()]), type1, type2)
        else:
            if shape is None:
                shape = [1]
            shape_array_attr = ArrayAttr(
                [(IntegerAttr(d, 32) if isinstance(d, int) else d) for d in shape]
            )
            super().__init__(
                shape_array_attr,
                type1,
                NoneType(),
            )

    def print_parameters(self, printer: Printer) -> None:
        # We need extra work here as the builtin tuple is not being supported
        # yet, therefore handle this here
        with printer.in_angle_brackets():
            if isinstance(self.type2, NoneType):
                for s in self.shape.data:
                    if isinstance(s, DeferredAttr):
                        printer.print_string("?")
                    elif isinstance(s, NoneType):
                        raise Exception(
                            "Can not have none type as part of sequence shape with only one type"
                        )
                    else:
                        s.print_without_type(printer)
                    printer.print_string("x")
                printer.print_attribute(self.type)
            else:
                printer.print_string("0xtuple")
                with printer.in_angle_brackets():
                    printer.print_attribute(self.type)
                    printer.print_string(", ")
                    printer.print_attribute(self.type2)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        # We need extra work here as the builtin tuple is not being supported
        # yet, therefore handle this here
        def parse_interval() -> IntegerAttr[IntegerType] | DeferredAttr:
            if parser.parse_optional_punctuation("?"):
                return DeferredAttr()
            s = parser.parse_integer(allow_boolean=False)
            return IntegerAttr(s, 32)

        shape: list[IntegerAttr[IntegerType] | DeferredAttr] = []
        type2 = NoneType()
        parser.parse_characters("<")
        has_tuple = parser.parse_optional_characters("0")
        if has_tuple is not None:
            parser.parse_characters("xtuple")
            parser.parse_characters("<")
            type1 = parser.parse_type()
            parser.parse_characters(",")
            type2 = parser.parse_type()
            parser.parse_characters(">")
            shape.append(IntegerAttr(1, 32))
        else:
            type1 = parser.parse_optional_type()
            while type1 is None:
                shape.append(parse_interval())
                parser.parse_shape_delimiter()
                type1 = parser.parse_optional_type()
        parser.parse_characters(">")
        return [ArrayAttr(shape), type1, type2]

    def hasDeferredShape(self):
        for s in self.shape.data:
            if isinstance(s, DeferredAttr):
                return True
        return False

    def getNumberDims(self):
        return len(self.shape.data)


@irdl_attr_definition
class CharacterType(ParametrizedAttribute, TypeAttribute):
    """
    Model of the Fortran CHARACTER intrinsic type, including the KIND type
    parameter. The model optionally includes a LEN type parameter. A
    CharacterType is thus the type of both a single character value and a
    character with a LEN parameter.
    """

    name = "fir.char"

    from_index: IntAttr | DeferredAttr
    to_index: IntAttr | DeferredAttr

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            if isinstance(self.from_index, DeferredAttr):
                printer.print_string("?")
            else:
                printer.print_int(self.from_index.data)

            printer.print_string(",")

            if isinstance(self.to_index, DeferredAttr):
                printer.print_string("?")
            else:
                printer.print_int(self.to_index.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        def parse_value():
            if parser.parse_optional_punctuation("?"):
                return DeferredAttr()
            else:
                return IntAttr(parser.parse_integer(allow_boolean=False))

        parser.parse_characters("<")
        lower = parse_value()
        has_upper = parser.parse_optional_characters(",")
        if has_upper:
            upper = parse_value()
        else:
            upper = IntAttr(1)
        parser.parse_characters(">")
        return [lower, upper]


@irdl_attr_definition
class LogicalType(ParametrizedAttribute, TypeAttribute):
    """
    Model of a Fortran LOGICAL intrinsic type, including the KIND type
    parameter
    """

    name = "fir.logical"

    size: IntAttr

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.size.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        s = parser.parse_integer(allow_boolean=False)
        parser.parse_characters(">")
        return [IntAttr(s)]


@irdl_attr_definition
class ComplexType(ParametrizedAttribute, TypeAttribute):
    """
    Model of a Fortran COMPLEX intrinsic type, including the KIND type
    parameter. COMPLEX is a floating point type with a real and imaginary
    member.
    """

    name = "fir.complex"

    width: IntAttr

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.width.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        s = parser.parse_integer(allow_boolean=False)
        parser.parse_characters(">")
        return [IntAttr(s)]


@irdl_attr_definition
class ShiftType(ParametrizedAttribute, TypeAttribute):
    """
    Type of a vector of runtime values that define the lower bounds of a
    multidimensional array object. The vector is the lower bounds of each array
    dimension. The rank of a ShiftType must be at least 1.
    """

    name = "fir.shift"

    indexes: IntAttr

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.indexes.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        s = parser.parse_integer(allow_boolean=False)
        parser.parse_characters(">")
        return [IntAttr(s)]


@irdl_attr_definition
class ShapeType(ParametrizedAttribute, TypeAttribute):
    """
    Type of a vector of runtime values that define the shape of a
    multidimensional array object. The vector is the extents of each array
    dimension. The rank of a ShapeType must be at least 1.
    """

    name = "fir.shape"

    indexes: IntAttr

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.indexes.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        s = parser.parse_integer(allow_boolean=False)
        parser.parse_characters(">")
        return [IntAttr(s)]


@irdl_attr_definition
class ShapeShiftType(ParametrizedAttribute, TypeAttribute):
    """
    Type of a vector of runtime values that define the shape and the origin of a
    multidimensional array object. The vector is of pairs, origin offset and
    extent, of each array dimension. The rank of a ShapeShiftType must be at
    least 1.
    """

    name = "fir.shapeshift"

    indexes: IntAttr

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.indexes.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        s = parser.parse_integer(allow_boolean=False)
        parser.parse_characters(">")
        return [IntAttr(s)]


@irdl_attr_definition
class HeapType(ParametrizedAttribute, TypeAttribute):
    """
    The type of a heap pointer. Fortran entities with the ALLOCATABLE attribute
    may be allocated on the heap at runtime. These pointers are explicitly
    distinguished to disallow the composition of multiple levels of
    indirection. For example, an ALLOCATABLE POINTER is invalid.
    """

    name = "fir.heap"

    type: SequenceType | CharacterType


@irdl_attr_definition
class BoxType(ParametrizedAttribute, TypeAttribute):
    """
    Descriptors are tuples of information that describe an entity being passed
    from a calling context. This information might include (but is not limited
    to) whether the entity is an array, its size, or what type it has.
    """

    name = "fir.box"

    type: Attribute


@irdl_attr_definition
class BoxCharType(ParametrizedAttribute, TypeAttribute):
    """
    The type of a pair that describes a CHARACTER variable. Specifically, a
    CHARACTER consists of a reference to a buffer (the string value) and a LEN
    type parameter (the runtime length of the buffer).
    """

    name = "fir.boxchar"

    kind: IntAttr

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_int(self.kind.data)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        s = parser.parse_integer(allow_boolean=False)
        parser.parse_characters(">")
        return [IntAttr(s)]


@irdl_op_definition
class AbsentOp(IRDLOperation):
    """
    Given the type of a function argument, create a value that will signal that
    an optional argument is absent in the call. On the caller side, fir.is_present
    can be used to query if the value of an optional argument was created with
    a fir.absent operation.

    It is undefined to use a value that was created by a fir.absent op in any other
    operation than fir.call and fir.is_present.

    %1 = fir.absent fir.box<fir.array<?xf32>>
    fir.call @_QPfoo(%1) : (fir.box<fir.array<?xf32>>) -> ()
    """

    name = "fir.absent"
    intype = result_def()
    regs = var_region_def()


@irdl_op_definition
class AddcOp(IRDLOperation):
    name = "fir.addc"
    lhs = operand_def()
    rhs = operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()
    regs = var_region_def()


@irdl_op_definition
class AddressOfOp(IRDLOperation):
    """
    Convert a symbol (a function or global reference) to an SSA-value to be
    used in other operations. References to Fortran symbols are distinguished
    via this operation from other arbitrary constant values.

    %p = fir.address_of(@symbol) : !fir.ref<f64>
    """

    name = "fir.address_of"
    symbol = prop_def(SymbolRefAttr)
    resTy = result_def()
    regs = var_region_def()


@irdl_op_definition
class AllocmemOp(IRDLOperation):
    """
    Creates a heap memory reference suitable for storing a value of the
    given type, T.  The heap refernce returned has type `!fir.heap<T>`.
    The memory object is in an undefined state.  `allocmem` operations must
    be paired with `freemem` operations to avoid memory leaks.

    %0 = fir.allocmem !fir.array<10 x f32>
    fir.freemem %0 : !fir.heap<!fir.array<10 x f32>>
    """

    name = "fir.allocmem"
    in_type = prop_def()
    uniq_name = opt_prop_def(StringAttr)
    bindc_name = opt_prop_def(StringAttr)
    typeparams = var_operand_def()
    shape = var_operand_def()

    result_0 = result_def()
    regs = var_region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class AllocaOp(IRDLOperation):
    """
    This primitive operation is used to allocate an object on the stack.  A
    reference to the object of type '!fir.ref<T>' is returned.  The returned
    object has an undefined/uninitialized state.  The allocation can be given
    an optional name.  The allocation may have a dynamic repetition count
    for allocating a sequence of locations for the specified type.

    ```
    %c = ... : i64
    %x = fir.alloca i32
    %y = fir.alloca !fir.array<8 x i64>
    %z = fir.alloca f32, %c

    %i = ... : i16
    %j = ... : i32
    %w = fir.alloca !fir.type<PT(len1:i16, len2:i32)> (%i, %j : i16, i32)
    ```

    Note that in the case of '%z', a contiguous block of memory is allocated
    and its size is a runtime multiple of a 32-bit REAL value.

    In the case of '%w', the arguments '%i' and '%j' are LEN parameters
    ('len1', 'len2') to the type 'PT'.

    Finally, the operation is undefined if the ssa-value '%c' is negative.

    Fortran Semantics:
    There is no language mechanism in Fortran to allocate space on the stack
    like C's 'alloca()' function. Therefore fir.alloca is not control-flow
    dependent. However, the lifetime of a stack allocation is often limited to
    a small region and a legal implementation may reuse stack storage in other
    regions when there is no conflict. For example, take the following code
    fragment.

    CALL foo(1)
    CALL foo(2)
    CALL foo(3)

    A legal implementation can allocate a stack slot and initialize it with the
    constant '1', then pass that by reference to foo. Likewise for the second
    and third calls to foo, each stack slot being initialized accordingly. It is
    also a conforming implementation to reuse the same stack slot for all three
    calls, just initializing each in turn. This is possible as the lifetime of
    the copy of each constant need not exceed that of the CALL statement.
    Indeed, a user would likely expect a good Fortran compiler to perform such
    an optimization.

    Until Fortran 2018, procedures defaulted to non-recursive. A legal
    implementation could therefore convert stack allocations to global
    allocations. Such a conversion effectively adds the SAVE attribute to all
    variables.

    Some temporary entities (large arrays) probably should not be stack
    allocated as stack space can often be limited. A legal implementation can
    convert these large stack allocations to heap allocations regardless of
    whether the procedure is recursive or not.

    The pinned attribute is used to flag fir.alloca operation in a specific
    region and avoid them being hoisted in an alloca hoisting pass.
    """

    name = "fir.alloca"
    in_type = prop_def()
    uniq_name = opt_prop_def(StringAttr)
    bindc_name = opt_prop_def(StringAttr)
    typeparams = var_operand_def()
    shape = var_operand_def()
    result_0 = result_def()
    regs = var_region_def()
    valuebyref = opt_prop_def(UnitAttr)
    pinned = opt_prop_def(UnitAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class ArrayAccessOp(IRDLOperation):
    """
    The 'array_access' provides a reference to a single element from an array
    value. This is not a view in the immutable array, otherwise it couldn't
    be stored to. It can be see as a logical copy of the element and its
    position in the array. This reference can be written to and modified without
    changing the original array.

    The 'array_access' operation is used to fetch the memory reference of an
    element in an array value.

    real :: a(n,m)
    ...
    ... a ...
    ... a(r,s+1) ...

    One can use 'fir.array_access' to recover the implied memory reference to
    the element 'a(i,j)' in an array expression 'a' as shown above. It can also
    be used to recover the reference element 'a(r,s+1)' in the second
    expression.

    %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
    // load the entire array 'a'
    %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
    // fetch the value of one of the array value's elements
    %1 = fir.array_access %v, %i, %j : (!fir.array<?x?xf32>, index, index) -> !fir.ref<f32>

    It is only possible to use 'array_access' on an 'array_load' result value or
    a value that can be trace back transitively to an 'array_load' as the
    dominating source. Other array operation such as 'array_amend' can be in
    between.

    TODO: The above restriction is not enforced. The design of the operation
    might need to be revisited to avoid such restrictions.

    More information about 'array_access' and other array operations can be
    found in Flang documentation at flang/docs/FIRArrayOperations.md.
    """

    name = "fir.array_access"
    sequence = operand_def()
    indices = operand_def()
    typeparams = operand_def()
    element = result_def()
    regs = var_region_def()


@irdl_op_definition
class ArrayAmendOp(IRDLOperation):
    """
    The 'array_amend' operation marks an array value as having been changed via
    a reference obtained by an `array_access`. It acts as a logical transaction
    log that is used to merge the final result back with an `array_merge_store`
    operation.

    // fetch the value of one of the array value's elements
    %1 = fir.array_access %v, %i, %j : (!fir.array<?x?xT>, index, index) -> !fir.ref<T>
    // modify the element by storing data using %1 as a reference
    %2 = ... %1 ...
    // mark the array value
    %new_v = fir.array_amend %v, %2 : (!fir.array<?x?xT>, !fir.ref<T>) -> !fir.array<?x?xT>

    More information about `array_amend` and other array operations can be
    found in Flang documentation at flang/docs/FIRArrayOperations.md.
    """

    name = "fir.array_amend"
    sequence = operand_def()
    memref = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class ArrayCoorOp(IRDLOperation):
    """
    Compute the location of an element in an array when the shape of the
    array is only known at runtime.

    This operation is intended to capture all the runtime values needed to
    compute the address of an array reference in a single high-level op. Given
    the following Fortran input:

    real :: a(n,m)
    ...
    ... a(i,j) ...

    One can use 'fir.array_coor' to determine the address of 'a(i,j)'.

    %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
    %1 = fir.array_coor %a(%s) %i, %j : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
    """

    name = "fir.array_coor"
    memref = operand_def()
    shape = operand_def()
    slice = operand_def()
    indices = operand_def()
    typeparams = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class ArrayFetchOp(IRDLOperation):
    """
    Fetch the value of an element in an array value.

    real :: a(n,m)
    ...
    ... a ...
    ... a(r,s+1) ...

    One can use 'fir.array_fetch' to fetch the (implied) value of 'a(i,j)' in
    an array expression as shown above. It can also be used to extract the
    element 'a(r,s+1)' in the second expression.

    %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
    // load the entire array 'a'
    %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
    // fetch the value of one of the array value's elements
    %1 = fir.array_fetch %v, %i, %j : (!fir.array<?x?xf32>, index, index) -> f32

    It is only possible to use 'array_fetch' on an 'array_load' result value.
    """

    name = "fir.array_fetch"
    sequence = operand_def()
    indices = operand_def()
    typeparams = operand_def()
    element = result_def()
    regs = var_region_def()


@irdl_op_definition
class ArrayLoadOp(IRDLOperation):
    """
    This operation taken with array_merge_store captures Fortran's
    copy-in/copy-out semantics. One way to think of this is that array_load
    creates a snapshot copy of the entire array. This copy can then be used
    as the "original value" of the array while the array's new value is
    computed. The array_merge_store operation is the copy-out semantics, which
    merge the updates with the original array value to produce the final array
    result. This abstracts the copy operations as opposed to always creating
    copies or requiring dependence analysis be performed on the syntax trees
    and before lowering to the IR.

    Load an entire array as a single SSA value.

    real :: a(o:n,p:m)
    ...
    ... = ... a ...

    One can use 'fir.array_load' to produce an ssa-value that captures an
    immutable value of the entire array `a`, as in the Fortran array expression
    shown above. Subsequent changes to the memory containing the array do not
    alter its composite value. This operation lets one load an array as a
    value while applying a runtime shape, shift, or slice to the memory
    reference, and its semantics guarantee immutability.

    %s = fir.shape_shift %o, %n, %p, %m : (index, index, index, index) -> !fir.shapeshift<2>
    // load the entire array 'a'
    %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
    // a fir.store here into array %a does not change %v
    """

    name = "fir.array_load"
    memref = operand_def()
    shape = operand_def()
    slice = operand_def()
    typeparams = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class ArrayMergeStoreOp(IRDLOperation):
    """
    Store a merged array value to memory.

    real :: a(n,m)
    ...
    a = ...

    One can use 'fir.array_merge_store' to merge/copy the value of 'a' in an
    array expression as shown above.

    %v = fir.array_load %a(%shape) : ...
    %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
    fir.array_merge_store %v, %r to %a : !fir.ref<!fir.array<?x?xf32>>

    This operation merges the original loaded array value, '%v', with the
    chained updates, '%r', and stores the result to the array at address, '%a'.
    """

    name = "fir.array_merge_store"
    original = operand_def()
    sequence = operand_def()
    memref = operand_def()
    slice = operand_def()
    typeparams = operand_def()
    regs = var_region_def()


@irdl_op_definition
class ArrayModifyOp(IRDLOperation):
    """
    Modify the value of an element in an array value through actions done
    on the returned address. A new array value is also
    returned where all element values of the input array are identical except
    for the selected element which is the value after the modification done
    on the element address.

    real :: a(n)
    ...
    ! Elemental user defined assignment from type(SomeType) to real.
    a = value_of_some_type

    One can use 'fir.array_modify' to update the (implied) value of 'a(i)'
    in an array expression as shown above.

    %s = fir.shape %n : (index) -> !fir.shape<1>
    // Load the entire array 'a'.
    %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
    // Update the value of one of the array value's elements with a user
    // defined assignment from %rhs.
    %new = fir.do_loop %i = ... (%inner = %v) {
      %rhs = ...
      %addr, %r = fir.array_modify %inner, %i : (!fir.array<?xf32>, index) -> (fir.ref<f32>, !fir.array<?xf32>)
      fir.call @user_def_assign(%addr, %rhs) (fir.ref<f32>, fir.ref<!fir.type<SomeType>>) -> ()
      fir.result %r : !fir.ref<!fir.array<?xf32>>
    }
    fir.array_merge_store %v, %new to %a : !fir.ref<!fir.array<?xf32>>
    """

    name = "fir.array_modify"
    sequence = operand_def()
    indices = operand_def()
    typeparams = operand_def()
    result_0 = result_def()
    result_1 = result_def()
    regs = var_region_def()


@irdl_op_definition
class ArrayUpdateOp(IRDLOperation):
    """
     Updates the value of an element in an array value. A new array value is
    returned where all element values of the input array are identical except
    for the selected element which is the value passed in the update.

    real :: a(n,m)
    ...
    a = ...

    One can use 'fir.array_update' to update the (implied) value of 'a(i,j)'
    in an array expression as shown above.

    %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
    // load the entire array 'a'
    %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
    // update the value of one of the array value's elements
    // %r_{ij} = %f  if (i,j) = (%i,%j),   %v_{ij} otherwise
    %r = fir.array_update %v, %f, %i, %j : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
    fir.array_merge_store %v, %r to %a : !fir.ref<!fir.array<?x?xf32>>

    An array value update behaves as if a mapping function from the indices
    to the new value has been added, replacing the previous mapping. These
    mappings can be added to the ssa-value, but will not be materialized in
    memory until the 'fir.array_merge_store' is performed.
    """

    name = "fir.array_update"
    sequence = operand_def()
    merge = operand_def()
    indices = operand_def()
    typeparams = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxAddrOp(IRDLOperation):
    """
    This operator is overloaded to work with values of type 'box',
    'boxchar', and 'boxproc'.  The result for each of these
    cases, respectively, is the address of the data, the address of the
    'CHARACTER' data, and the address of the procedure.

    %51 = fir.box_addr %box : (!fir.box<f64>) -> !fir.ref<f64>
    %52 = fir.box_addr %boxchar : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1>>
    %53 = fir.box_addr %boxproc : (!fir.boxproc<!P>) -> !P
    """

    name = "fir.box_addr"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxcharLenOp(IRDLOperation):
    """
    Extracts the LEN type parameter from a 'boxchar' value.

    %45 = ... : !boxchar<1>  // CHARACTER(20)
    %59 = fir.boxchar_len %45 : (!fir.boxchar<1>) -> i64  // len=20
    """

    name = "fir.boxchar_len"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxDimsOp(IRDLOperation):
    """
    Returns the triple of lower bound, extent, and stride for 'dim' dimension
    of 'val', which must have a 'box' type.  The dimensions are enumerated from
    left to right from 0 to rank-1. This operation has undefined behavior if
    'dim' is out of bounds.

    %c1   = arith.constant 0 : i32
    %52:3 = fir.box_dims %40, %c1 : (!fir.box<!fir.array<*:f64>>, i32) -> (index, index, index)

    The above is a request to return the left most row (at index 0) triple from
    the box. The triple will be the lower bound, extent, and byte-stride, which
    are the values encoded in a standard descriptor.
    """

    name = "fir.box_dims"
    val = operand_def()
    dim = operand_def()
    result_0 = result_def()
    result_1 = result_def()
    result_2 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxElesizeOp(IRDLOperation):
    """
    Returns the size of an element in an entity of 'box' type.  This size may
    not be known until runtime.

    %53 = fir.box_elesize %40 : (!fir.box<f32>) -> i32  // size=4
    %54 = fir.box_elesize %40 : (!fir.box<!fir.array<*:f32>>) -> i32

    In the above example, '%53' may box an array of REAL values while '%54'
    must box an array of REAL values (with dynamic rank and extent).
    """

    name = "fir.box_elesize"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxIsallocOp(IRDLOperation):
    """
    Determine if the boxed value was from an ALLOCATABLE entity. This will
    return true if the originating box value was from a 'fir.embox' op
    with a mem-ref value that had the type !fir.heap<T>.

    %r = ... : !fir.heap<i64>
    %b = fir.embox %r : (!fir.heap<i64>) -> !fir.box<i64>
    %a = fir.box_isalloc %b : (!fir.box<i64>) -> i1  // true

    The canonical descriptor implementation will carry a flag to record if the
    variable is an 'ALLOCATABLE'.
    """

    name = "fir.box_isalloc"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxIsarrayOp(IRDLOperation):
    """
    Determine if the boxed value has a positive (> 0) rank. This will return
    true if the originating box value was from a fir.embox with a memory
    reference value that had the type !fir.array<T> and/or a shape argument.

    %r = ... : !fir.ref<i64>
    %c_100 = arith.constant 100 : index
    %d = fir.shape %c_100 : (index) -> !fir.shape<1>
    %b = fir.embox %r(%d) : (!fir.ref<i64>, !fir.shape<1>) -> !fir.box<i64>
    %a = fir.box_isarray %b : (!fir.box<i64>) -> i1  // true
    """

    name = "fir.box_isarray"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxIsptrOp(IRDLOperation):
    """
    Determine if the boxed value was from a POINTER entity.

    %p = ... : !fir.ptr<i64>
    %b = fir.embox %p : (!fir.ptr<i64>) -> !fir.box<i64>
    %a = fir.box_isptr %b : (!fir.box<i64>) -> i1  // true
    """

    name = "fir.box_isptr"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxOffsetOp(IRDLOperation):
    """
    Given the address of a fir.box, compute the address of a field inside
    the fir.box.
    This allows keeping the actual runtime descriptor layout abstract in
    FIR while providing access to the pointer addresses in the runtime
    descriptor for OpenMP/OpenACC target mapping.

    To avoid requiring too much information about the fields that the runtime
    descriptor implementation must have, only the base_addr and derived_type
    descriptor fields can be addressed.

    %addr = fir.box_offset %box base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
    %tdesc = fir.box_offset %box derived_type : (!fir.ref<!fir.box<!fir.type<t>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<t>>>
    """

    name = "fir.box_offset"
    field = prop_def()
    val = operand_def()
    result_0 = result_def()


@irdl_op_definition
class BoxprocHostOp(IRDLOperation):
    """
    Extract the host context pointer from a boxproc value.

    %8 = ... : !fir.boxproc<(!fir.ref<!fir.type<T>>) -> i32>
    %9 = fir.boxproc_host %8 : (!fir.boxproc<(!fir.ref<!fir.type<T>>) -> i32>) -> !fir.ref<tuple<i32, i32>>

    In the example, the reference to the closure over the host procedure's
    variables is returned. This allows an internal procedure to access the
    host's variables. It is up to lowering to determine the contract between
    the host and the internal procedure.
    """

    name = "fir.boxproc_host"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxRankOp(IRDLOperation):
    """
    Return the rank of a value of 'box' type.  If the value is scalar, the
    rank is 0.

    %57 = fir.box_rank %40 : (!fir.box<!fir.array<*:f64>>) -> i32
    %58 = fir.box_rank %41 : (!fir.box<f64>) -> i32

    The example '%57' shows how one would determine the rank of an array that
    has deferred rank at runtime. This rank should be at least 1. In %58, the
    descriptor may be either an array or a scalar, so the value is nonnegative.
    """

    name = "fir.box_rank"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class BoxTdescOp(IRDLOperation):
    """
    Return the opaque type descriptor of a value of 'box' type. A type
    descriptor is an implementation defined value that fully describes a type
    to the Fortran runtime.

    %7 = fir.box_tdesc %41 : (!fir.box<f64>) -> !fir.tdesc<f64>
    """

    name = "fir.box_tdesc"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class CallOp(IRDLOperation):
    """
    Call the specified function or function reference.

    %a = fir.call %funcref(%arg0) : (!fir.ref<f32>) -> f32
    %b = fir.call @function(%arg1, %arg2) : (!fir.ref<f32>, !fir.ref<f32>) -> f32
    """

    name = "fir.call"
    callee = prop_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result_0 = opt_result_def()
    args = var_operand_def()
    regs = var_region_def()


@irdl_op_definition
class CharConvertOp(IRDLOperation):
    """
    Copy a CHARACTER (must be in memory) of KIND _k1_ to a CHARACTER (also must
    be in memory) of KIND _k2_ where _k1_ != _k2_ and the buffers do not
    overlap. This latter restriction is unchecked, as the Fortran language
    definition eliminates the overlapping in memory case.

    The number of code points copied is specified explicitly as the second
    argument. The length of the !fir.char type is ignored.

    fir.char_convert %1 for %2 to %3 : !fir.ref<!fir.char<1,?>>, i32,
        !fir.ref<!fir.char<2,20>>

    Should future support for encodings other than ASCII be supported, codegen
    can generate a call to a runtime helper routine which will map the code
    points from UTF-8 to UCS-2, for example. Such remappings may not always
    be possible as they may involve the creation of more code points than the
    'count' limit. These details are left as future to-dos.
    """

    name = "fir.char_convert"
    _from = operand_def()
    count = operand_def()
    to = operand_def()
    regs = var_region_def()


@irdl_op_definition
class CmpcOp(IRDLOperation):
    """
    A complex comparison to handle complex types found in FIR.
    """

    name = "fir.cmpc"
    lhs = operand_def()
    rhs = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class ConstcOp(IRDLOperation):
    """
    A complex constant. Similar to the standard dialect complex type, but this
    extension allows constants with APFloat values that are not supported in
    the standard dialect.
    """

    name = "fir.constc"
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class ConvertOp(IRDLOperation):
    """
    Generalized type conversion. Convert the ssa-value from type T to type U.
    Not all pairs of types have conversions. When types T and U are the same
    type, this instruction is a NOP and may be folded away. This also supports
    integer to pointer conversion and pointer to integer conversion.

    %v = ... : i64
    %w = fir.convert %v : (i64) -> i32

    The example truncates the value '%v' from an i64 to an i32.
    """

    name = "fir.convert"
    value = operand_def()
    res = result_def()
    regs = var_region_def()


@irdl_op_definition
class CoordinateOfOp(IRDLOperation):
    """
    Compute the internal coordinate address starting from a boxed value or
    unboxed memory reference. Returns a memory reference. When computing the
    coordinate of an array element, the rank of the array must be known and
    the number of indexing expressions must not exceed the rank of the array.

    This operation will apply the access map from a boxed value implicitly.

    Unlike LLVM's GEP instruction, one cannot stride over the outermost
    reference; therefore, the leading 0 index must be omitted.

    %i = ... : index
    %h = ... : !fir.heap<!fir.array<100 x f32>>
    %p = fir.coordinate_of %h, %i : (!fir.heap<!fir.array<100 x f32>>, index) -> !fir.ref<f32>

    In the example, '%p' will be a pointer to the '%i'-th f32 value in the
    array '%h'.
    """

    name = "fir.coordinate_of"
    baseType = prop_def()
    ref = operand_def()
    coor = var_operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class DeclareOp(IRDLOperation):
    """
    Tie the properties of a Fortran variable to an address. The properties
    include bounds, length parameters, and Fortran attributes.

    The memref argument describes the storage of the variable. It may be a
    raw address (fir.ref<T>), or a box or class value or address (fir.box<T>,
    fir.ref<fir.box<T>>, fir.class<T>, fir.ref<fir.class<T>>).

    The shape argument encodes explicit extents and lower bounds. It must be
    provided if the memref is the raw address of an array.
    The shape argument must not be provided if memref operand is a box or
    class value or address, unless the shape is a shift (encodes lower bounds)
    and the memref if a box value (this covers assumed shapes with local lower
    bounds).

    The typeparams values are meant to carry the non-deferred length parameters
    (this includes both Fortran assumed and explicit length parameters).
    It must always be provided for characters and parametrized derived types
    when memref is not a box value or address.

    Example:

    CHARACTER(n), OPTIONAL, TARGET :: c(10:, 20:)

    Can be represented as:
    func.func @foo(%arg0: !fir.box<!fir.array<?x?x!fir.char<1,?>>>, %arg1: !fir.ref<i64>) {
      %c10 = arith.constant 10 : index
      %c20 = arith.constant 20 : index
      %1 = fir.load %ag1 : fir.ref<i64>
      %2 = fir.shift %c10, %c20 : (index, index) -> !fir.shift<2>
      %3 = fir.declare %arg0(%2) typeparams %1 {fortran_attrs = #fir.var_attrs<optional, target>, uniq_name = "c"}
      // ... uses %3 as "c"
    }
    """

    name = "fir.declare"
    memref = operand_def()
    shape = operand_def()
    typeparams = var_operand_def()
    uniq_name = prop_def(StringAttr)
    fortran_attrs = opt_prop_def(FortranVariableFlagsAttr)


@irdl_op_definition
class DtEntryOp(IRDLOperation):
    """
    An entry in a dispatch table.  Allows a function symbol to be bound
    to a specifier method identifier.  A dispatch operation uses the dynamic
    type of a distinguished argument to determine an exact dispatch table
    and uses the method identifier to select the type-bound procedure to
    be called.

    fir.dt_entry method_name, @uniquedProcedure
    """

    name = "fir.dt_entry"
    regs = var_region_def()


@irdl_op_definition
class DispatchOp(IRDLOperation):
    """
    Perform a dynamic dispatch on the method name via the dispatch table
    associated with the first operand.  The attribute 'pass_arg_pos' can be
    used to select a dispatch operand other than the first one.  The absence of
    'pass_arg_pos' attribute means nopass.

    // fir.dispatch with no attribute.
    %r = fir.dispatch "methodA"(%o) : (!fir.class<T>) -> i32

    // fir.dispatch with the `pass_arg_pos` attribute.
    %r = fir.dispatch "methodA"(%o : !fir.class<T>) (%o : !fir.class<T>) -> i32 {pass_arg_pos = 0 : i32}
    """

    name = "fir.dispatch"
    pass_arg_pos = opt_prop_def(IntegerAttr)
    object = operand_def()
    args = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class DispatchTableOp(IRDLOperation):
    """
    Define a dispatch table for a derived type with type-bound procedures.

    A dispatch table is an untyped symbol that contains a list of associations
    between method identifiers and corresponding 'FuncOp' symbols.

    The ordering of associations in the map is determined by the front end.

    fir.dispatch_table @_QDTMquuzTfoo {
      fir.dt_entry method1, @_QFNMquuzTfooPmethod1AfooR
      fir.dt_entry method2, @_QFNMquuzTfooPmethod2AfooII
    }
    """

    name = "fir.dispatch_table"

    sym_name = prop_def(SymbolNameConstraint())
    regs = var_region_def()

    traits = traits_def(SymbolOpInterface())


@irdl_op_definition
class DivcOp(IRDLOperation):
    name = "fir.divc"
    lhs = operand_def()
    rhs = operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()
    regs = var_region_def()


@irdl_op_definition
class DoLoopOp(IRDLOperation):
    """
    Generalized high-level looping construct. This operation is similar to
    MLIR's 'scf.for'.

    %l = arith.constant 0 : index
    %u = arith.constant 9 : index
    %s = arith.constant 1 : index
    fir.do_loop %i = %l to %u step %s unordered {
      %x = fir.convert %i : (index) -> i32
      %v = fir.call @compute(%x) : (i32) -> f32
      %p = fir.coordinate_of %A, %i : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      fir.store %v to %p : !fir.ref<f32>
    }

    The above example iterates over the interval '[%l, %u]'. The unordered
    keyword indicates that the iterations can be executed in any order.
    """

    name = "fir.do_loop"
    lowerBound = operand_def()
    upperBound = operand_def()
    step = operand_def()
    reduceOperands = var_operand_def()
    initArgs = var_operand_def()
    finalValue = opt_prop_def()
    initArgs = opt_operand_def()
    _results = var_result_def()
    regs = var_region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class DummyScopeOp(IRDLOperation):
    """
    An abstract handle to be used to associate dummy arguments of the same
    subroutine between each other. By lowering, all [hl]fir.declare
    operations representing declarations of dummy arguments of a subroutine
    use the result of this operation. This allows recognizing the references
    of these dummy arguments as belonging to the same runtime instance
    of the subroutine even after MLIR inlining. Thus, the Fortran aliasing
    rules might be applied to those references based on the original
    declarations of the dummy arguments.
    For example:
      subroutine test(x, y)
        real, target :: x, y
        x = y ! may alias
        call inner(x, y)
      contains
        subroutine inner(x, y)
          real :: x, y
          x = y ! may not alias
        end subroutine inner
      end subroutine test

    After MLIR inlining this may look like this:

      func.func @_QPtest(
          %arg0: !fir.ref<f32> {fir.target},
          %arg1: !fir.ref<f32> {fir.target}) {
        %0 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<target>} :
            (!fir.ref<f32>) -> !fir.ref<f32>
        %1 = fir.declare %arg1 {fortran_attrs = #fir.var_attrs<target>} :
            (!fir.ref<f32>) -> !fir.ref<f32>
        %2 = fir.load %1 : !fir.ref<f32>
        fir.store %2 to %0 : !fir.ref<f32>
        %3 = fir.declare %0 : (!fir.ref<f32>) -> !fir.ref<f32>
        %4 = fir.declare %1 : (!fir.ref<f32>) -> !fir.ref<f32>
        %5 = fir.load %4 : !fir.ref<f32>
        fir.store %5 to %3 : !fir.ref<f32>
        return
      }

    Without marking %3 and %4 as declaring the dummy arguments
    of the same runtime instance of `inner` subroutine the FIR
    AliasAnalysis cannot deduce non-aliasing for the second load/store pair.
    This information may be preserved by using fir.dummy_scope operation:

      func.func @_QPtest(
          %arg0: !fir.ref<f32> {fir.target},
          %arg1: !fir.ref<f32> {fir.target}) {
        %h1 = fir.dummy_scope : i1
        %0 = fir.declare %arg0 dummy_scope(%h1)
            {fortran_attrs = #fir.var_attrs<target>} :
            (!fir.ref<f32>) -> !fir.ref<f32>
        %1 = fir.declare %arg1 dummy_scope(%h1)
            {fortran_attrs = #fir.var_attrs<target>} :
            (!fir.ref<f32>) -> !fir.ref<f32>
        %2 = fir.load %1 : !fir.ref<f32>
        fir.store %2 to %0 : !fir.ref<f32>
        %h2 = fir.dummy_scope : i1
        %3 = fir.declare %0 dummy_scope(%h2) : (!fir.ref<f32>) -> !fir.ref<f32>
        %4 = fir.declare %1 dummy_scope(%h2) : (!fir.ref<f32>) -> !fir.ref<f32>
        %5 = fir.load %4 : !fir.ref<f32>
        fir.store %5 to %3 : !fir.ref<f32>
        return
      }

    Note that even if `inner` is called and inlined twice inside
    `test`, the two inlined instances of `inner` must use two different
    fir.dummy_scope operations for their fir.declare ops. This
    two distinct fir.dummy_scope must remain distinct during the optimizations.
    This is guaranteed by the write memory effect on the DebuggingResource.
    """

    name = "fir.dummy_scope"

    result = result_def()


@irdl_op_definition
class EmboxcharOp(IRDLOperation):
    """
    Create a boxed CHARACTER value. The CHARACTER type has the LEN type
    parameter, the value of which may only be known at runtime.  Therefore,
    a variable of type CHARACTER has both its data reference as well as a
    LEN type parameter.

    CHARACTER(LEN=10) :: var

    %4 = ...         : !fir.ref<!fir.array<10 x !fir.char<1>>>
    %5 = arith.constant 10 : i32
    %6 = fir.emboxchar %4, %5 : (!fir.ref<!fir.array<10 x !fir.char<1>>>, i32) -> !fir.boxchar<1>

    In the above '%4' is a memory reference to a buffer of 10 CHARACTER units.
    This buffer and its LEN value (10) are wrapped into a pair in '%6'.
    """

    name = "fir.emboxchar"
    memref = operand_def()
    len = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class EmboxOp(IRDLOperation):
    """
    Create a boxed reference value. In Fortran, the implementation can require
    extra information about an entity, such as its type, rank, etc.  This
    auxiliary information is packaged and abstracted as a value with box type
    by the calling routine. (In Fortran, these are called descriptors.)

    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %5 = ... : !fir.ref<!fir.array<10 x i32>>
    %6 = fir.embox %5 : (!fir.ref<!fir.array<10 x i32>>) -> !fir.box<!fir.array<10 x i32>>

    The descriptor tuple may contain additional implementation-specific
    information through the use of additional attributes.
    Specifically,
        - shape: emboxing an array may require shape information (an array's
          lower bounds and extents may not be known until runtime),
        - slice: an array section can be described with a slice triple,
        - typeparams: for emboxing a derived type with LEN type parameters,
        - accessMap: unused/experimental.
    """

    name = "fir.embox"
    memref = operand_def()
    shape = var_operand_def()
    slice = var_operand_def()
    typeparams = var_operand_def()
    sourceBox = var_operand_def()
    result_0 = result_def()
    regs = var_region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class EmboxprocOp(IRDLOperation):
    """
    Creates an abstract encapsulation of a PROCEDURE POINTER along with an
    optional pointer to a host instance context. If the pointer is not to an
    internal procedure or the internal procedure does not need a host context
    then the form takes only the procedure's symbol.

    %f = ... : (i32) -> i32
    %0 = fir.emboxproc %f : ((i32) -> i32) -> !fir.boxproc<(i32) -> i32>

    An internal procedure requiring a host instance for correct execution uses
    the second form. The closure of the host procedure's state is passed as a
    reference to a tuple. It is the responsibility of the host to manage the
    context's values accordingly, up to and including inhibiting register
    promotion of local values.

    %4 = ... : !fir.ref<tuple<!fir.ref<i32>, !fir.ref<i32>>>
    %g = ... : (i32) -> i32
    %5 = fir.emboxproc %g, %4 : ((i32) -> i32, !fir.ref<tuple<!fir.ref<i32>, !fir.ref<i32>>>) -> !fir.boxproc<(i32) -> i32>
    """

    name = "fir.emboxproc"
    func = operand_def()
    host = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class ExtractValueOp(IRDLOperation):
    """
    Extract a value from an entity with a type composed of tuples, arrays,
    and/or derived types. Returns the value from entity with the type of the
    specified component. Cannot be used on values of '!fir.box' type.
    It can also be used to access complex parts and elements of a character
    string.

    Note that the entity ssa-value must be of compile-time known size in order
    to use this operation.

    %f = fir.field_index field, !fir.type<X{field:i32}>
    %s = ... : !fir.type<X>
    %v = fir.extract_value %s, %f : (!fir.type<X>, !fir.field) -> i32
    """

    name = "fir.extract_value"
    adt = operand_def()
    coor = opt_prop_def()
    res = result_def()
    regs = var_region_def()


@irdl_op_definition
class FieldIndexOp(IRDLOperation):
    """
    Generate a field (offset) value from an identifier.  Field values may be
    lowered into exact offsets when the layout of a Fortran derived type is
    known at compile-time. The type of a field value is '!fir.field' and
    these values can be used with the 'fir.coordinate_of', 'fir.extract_value',
    or 'fir.insert_value' instructions to compute (abstract) addresses of
    subobjects.

    %f = fir.field_index field, !fir.type<X{field:i32}>
    """

    name = "fir.field_index"
    typeparams = operand_def()
    res = result_def()
    regs = var_region_def()


@irdl_op_definition
class EndOp(IRDLOperation):
    """
    The end terminator is a special terminator used inside various FIR
    operations that have regions.  End is thus the custom invisible terminator
    for these operations.  It is implicit and need not appear in the textual
    representation.
    """

    name = "fir.end"
    regs = var_region_def()

    traits = traits_def(IsTerminator())


@irdl_op_definition
class FreememOp(IRDLOperation):
    """
    Deallocates a heap memory reference that was allocated by an 'allocmem'.
    The memory object that is deallocated is placed in an undefined state
    after 'fir.freemem'.  Optimizations may treat the loading of an object
    in the undefined state as undefined behavior.  This includes aliasing
    references, such as the result of an 'fir.embox'.

    %21 = fir.allocmem !fir.type<ZT(p:i32){field:i32}>
    ...
    fir.freemem %21 : !fir.heap<!fir.type<ZT>>
    """

    name = "fir.freemem"
    heapref = operand_def()
    regs = var_region_def()


@irdl_op_definition
class GentypedescOp(IRDLOperation):
    """
    Generates a constant object that is an abstract type descriptor of the
    specified type.  The meta-type of a type descriptor for the type 'T'
    is '!fir.tdesc<T>'.

    !T = !fir.type<T{...}>
    %t = fir.gentypedesc !T  // returns value of !fir.tdesc<!T>
    """

    name = "fir.gentypedesc"
    res = result_def()
    regs = var_region_def()


@irdl_op_definition
class GlobalLenOp(IRDLOperation):
    """
    A global entity (that is not an automatic data object) can have extra LEN
    parameter (compile-time) constants associated with the instance's type.
    These values can be bound to the global instance used 'fir.global_len'.

    global @g : !fir.type<t(len1:i32)> {
      fir.global_len len1, 10 : i32
      %1 = fir.undefined !fir.type<t(len1:i32)>
      fir.has_value %1 : !fir.type<t(len1:i32)>
    }
    """

    name = "fir.global_len"
    regs = var_region_def()


@irdl_op_definition
class GlobalOp(IRDLOperation):
    """
    A global variable or constant with initial values.

    The example creates a global variable (writable) named
    '@_QV_Mquark_Vvarble' with some initial values. The initializer should
    conform to the variable's type.

    fir.global @_QV_Mquark_Vvarble : tuple<i32, f32> {
      %1 = arith.constant 1 : i32
      %2 = arith.constant 2.0 : f32
      %3 = fir.undefined tuple<i32, f32>
      %z = arith.constant 0 : index
      %o = arith.constant 1 : index
      %4 = fir.insert_value %3, %1, %z : (tuple<i32, f32>, i32, index) -> tuple<i32, f32>
      %5 = fir.insert_value %4, %2, %o : (tuple<i32, f32>, f32, index) -> tuple<i32, f32>
      fir.has_value %5 : tuple<i32, f32>
    }
    """

    name = "fir.global"
    regs = var_region_def()
    sym_name = prop_def(SymbolNameConstraint())
    symref = prop_def(SymbolRefAttr)
    type = prop_def()
    initVal = opt_prop_def()
    constant = opt_prop_def(UnitAttr)
    target = opt_prop_def(UnitAttr)
    linkName = opt_prop_def(StringAttr)
    data_attr = opt_prop_def()
    alignment = opt_prop_def(IntegerAttr)

    traits = traits_def(SymbolOpInterface())


@irdl_op_definition
class HasValueOp(IRDLOperation):
    """
    The terminator for a GlobalOp with a body.

    global @variable : tuple<i32, f32> {
      %0 = arith.constant 45 : i32
      %1 = arith.constant 100.0 : f32
      %2 = fir.undefined tuple<i32, f32>
      %3 = arith.constant 0 : index
      %4 = fir.insert_value %2, %0, %3 : (tuple<i32, f32>, i32, index) -> tuple<i32, f32>
      %5 = arith.constant 1 : index
      %6 = fir.insert_value %4, %1, %5 : (tuple<i32, f32>, f32, index) -> tuple<i32, f32>
      fir.has_value %6 : tuple<i32, f32>
    }
    """

    name = "fir.has_value"
    resval = operand_def()
    regs = var_region_def()

    traits = traits_def(IsTerminator())


@irdl_op_definition
class IfOp(IRDLOperation):
    """
    Used to conditionally execute operations. This operation is the FIR
    dialect's version of 'loop.if'.

    %56 = ... : i1
    %78 = ... : !fir.ref<!T>
    fir.if %56 {
      fir.store %76 to %78 : !fir.ref<!T>
    } else {
      fir.store %77 to %78 : !fir.ref<!T>
    }
    """

    name = "fir.if"
    condition = operand_def()
    regs = var_region_def()


@irdl_op_definition
class InsertOnRangeOp(IRDLOperation):
    """
    Insert copies of a value into an entity with an array type of constant shape
    and size.
    Returns a new ssa-value with the same type as the original entity.
    The values are inserted at a contiguous range of indices in Fortran
    row-to-column element order as specified by lower and upper bound
    coordinates.

    %a = fir.undefined !fir.array<10x10xf32>
    %c = arith.constant 3.0 : f32
    %1 = fir.insert_on_range %a, %c from (0, 0) to (7, 2) : (!fir.array<10x10xf32>, f32) -> !fir.array<10x10xf32>

    The first 28 elements of %1, with coordinates from (0,0) to (7,2), have
    the value 3.0.
    """

    name = "fir.insert_on_range"
    seq = operand_def()
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class InsertValueOp(IRDLOperation):
    """
    Insert a value into an entity with a type composed of tuples, arrays,
    and/or derived types. Returns a new ssa-value with the same type as the
    original entity. Cannot be used on values of '!fir.box' type.
    It can also be used to set complex parts and elements of a character
    string.

    Note that the entity ssa-value must be of compile-time known size in order
    to use this operation.

    %a = ... : !fir.array<10xtuple<i32, f32>>
    %f = ... : f32
    %o = ... : i32
    %c = arith.constant 1 : i32
    %b = fir.insert_value %a, %f, %o, %c : (!fir.array<10x20xtuple<i32, f32>>, f32, i32, i32) -> !fir.array<10x20xtuple<i32, f32>>
    """

    name = "fir.insert_value"
    adt = operand_def()
    val = operand_def()
    coor = opt_prop_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class IsPresentOp(IRDLOperation):
    """
    Determine if an optional function argument is PRESENT (i.e. that it was not
    created by a fir.absent op on the caller side).

    func @_QPfoo(%arg0: !fir.box<!fir.array<?xf32>>) {
      %0 = fir.is_present %arg0 : (!fir.box<!fir.array<?xf32>>) -> i1
      ...
    """

    name = "fir.is_present"
    val = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class IterateWhileOp(IRDLOperation):
    """
    This single-entry, single-exit looping construct is useful for lowering
    counted loops that can exit early such as, for instance, implied-DO loops.
    It is very similar to fir.DoLoopOp with the addition that it requires
    a single loop-carried bool value that signals an early exit condition to
    the operation. A true disposition means the next loop iteration should
    proceed. A false indicates that the fir.iterate_while operation should
    terminate and return its iteration arguments. This is a degenerate counted
    loop in that the loop is not guaranteed to execute all iterations.

    An example iterate_while that returns the counter value, the early
    termination condition, and an extra loop-carried value is shown here. This
    loop counts from %lo to %up (inclusive), stepping by %c1, so long as the
    early exit (%ok) is true. The iter_args %sh value is also carried by the
    loop. The result triple is the values of %i=phi(%lo,%i+%c1),
    %ok=phi(%okIn,%okNew), and %sh=phi(%shIn,%shNew) from the last executed
    iteration.

    %v:3 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok = %okIn) iter_args(%sh = %shIn) -> (index, i1, i16) {
      %shNew = fir.call @bar(%sh) : (i16) -> i16
      %okNew = fir.call @foo(%sh) : (i16) -> i1
      fir.result %i, %okNew, %shNew : index, i1, i16
    }
    """

    name = "fir.iterate_while"
    lowerBound = operand_def()
    upperBound = operand_def()
    step = operand_def()
    iterateIn = operand_def()
    initArgs = operand_def()
    _results = var_result_def()
    regs = var_region_def()


@irdl_op_definition
class LenParamIndexOp(IRDLOperation):
    """
    Generate a LEN parameter (offset) value from a LEN parameter identifier.
    The type of a LEN parameter value is '!fir.len' and these values can be
    used with the fir.coordinate_of instructions to compute (abstract)
    addresses of LEN parameters.

    %e = fir.len_param_index len1, !fir.type<X(len1:i32)>
    %p = ... : !fir.box<!fir.type<X>>
    %q = fir.coordinate_of %p, %e : (!fir.box<!fir.type<X>>, !fir.len) -> !fir.ref<i32>
    """

    name = "fir.len_param_index"
    typeparams = operand_def()
    res = result_def()
    regs = var_region_def()


@irdl_op_definition
class LoadOp(IRDLOperation):
    """
    Load a value from a memory reference into an ssa-value (virtual register).
    Produces an immutable ssa-value of the referent type. A memory reference
    has type '!fir.ref<T>', '!fir.heap<T>', or '!fir.ptr<T>'.

    %a = fir.alloca i32
    %l = fir.load %a : !fir.ref<i32>

    The ssa-value has an undefined value if the memory reference is undefined
    or null.
    """

    name = "fir.load"
    memref = operand_def()
    res = result_def()
    regs = var_region_def()


@irdl_op_definition
class MulcOp(IRDLOperation):
    name = "fir.mulc"
    lhs = operand_def()
    rhs = operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()
    regs = var_region_def()


@irdl_op_definition
class NegcOp(IRDLOperation):
    name = "fir.negc"
    operand = operand_def()
    result = result_def()
    regs = var_region_def()


@irdl_op_definition
class NoReassocOp(IRDLOperation):
    """
    Primitive operation meant to intrusively prevent operator reassociation.
    The operation is otherwise a nop and the value returned is the same as the
    argument.

    The presence of this operation prevents any local optimizations. In the
    example below, this would prevent possibly replacing the multiply and add
    operations with a single FMA operation.

    %98 = arith.mulf %96, %97 : f32
    %99 = fir.no_reassoc %98 : f32
    %a0 = arith.addf %99, %95 : f32
    """

    name = "fir.no_reassoc"
    val = operand_def()
    res = result_def()
    regs = var_region_def()


@irdl_op_definition
class ReboxOp(IRDLOperation):
    """
    Create a new boxed reference value from another box. This is meant to be
    used when the taking a reference to part of a boxed value, or to an entire
    boxed value with new shape or type information.

    The new extra information can be:
      - new shape information (new lower bounds, new rank, or new extents.
        New rank/extents can only be provided if the original fir.box is
        contiguous in all dimension but maybe the first row). The shape
        operand must be provided to set new shape information.
      - new type (only for derived types). It is possible to set the dynamic
        type of the new box to one of the parent types of the input box dynamic
        type. Type parameters cannot be changed. This change is reflected in
        the requested result type of the new box.

    A slice argument can be provided to build a reference to part of a boxed
    value. In this case, the shape operand must be absent or be a fir.shift
    that can be used to provide a non default origin for the slice.

    The following example illustrates creating a fir.box for x(10:33:2)
    where x is described by a fir.box and has non default lower bounds,
    and then applying a new 2-dimension shape to this fir.box.

    %0 = fir.slice %c10, %c33, %c2 : (index, index, index) -> !fir.slice<1>
    %1 = fir.shift %c0 : (index) -> !fir.shift<1>
    %2 = fir.rebox %x(%1) [%0] : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
    %3 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
    %4 = fir.rebox %2(%3) : (!fir.box<!fir.array<?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
    """

    name = "fir.rebox"
    box = opt_operand_def()
    shape = opt_operand_def()
    slice = opt_operand_def()
    result_0 = result_def()
    regs = var_region_def()

    irdl_options = [AttrSizedOperandSegments(as_property=True)]


@irdl_op_definition
class ResultOp(IRDLOperation):
    """
    Result takes a list of ssa-values produced in the block and forwards them
    as a result to the operation that owns the region of the block. The
    operation can retain the values or return them to its parent block
    depending upon its semantics.
    """

    name = "fir.result"
    regs = var_region_def()
    _results = opt_operand_def()

    traits = traits_def(IsTerminator())


@irdl_op_definition
class SaveResultOp(IRDLOperation):
    """
    Save the result of a function returning an array, box, or record type value
    into a memory location given the shape and LEN parameters of the result.

    Function results of type fir.box, fir.array, or fir.rec are abstract values
    that require a storage to be manipulated on the caller side. This operation
    allows associating such abstract result to a storage. In later lowering of
    the function interfaces, this storage might be used to pass the result in
    memory.

    For arrays, result, it is required to provide the shape of the result. For
    character arrays and derived types with LEN parameters, the LEN parameter
    values must be provided.

    The fir.save_result associated to a function call must immediately follow
    the call and be in the same block.

    %buffer = fir.alloca fir.array<?xf32>, %c100
    %shape = fir.shape %c100
    %array_result = fir.call @foo() : () -> fir.array<?xf32>
    fir.save_result %array_result to %buffer(%shape)
    %coor = fir.array_coor %buffer%(%shape), %c5
    %fifth_element = fir.load %coor : f32

    The above fir.save_result allows saving a fir.array function result into
    a buffer to later access its 5th element.
    """

    name = "fir.save_result"
    value = operand_def()
    memref = operand_def()
    shape = operand_def()
    typeparams = operand_def()
    regs = var_region_def()


@irdl_op_definition
class SelectCaseOp(IRDLOperation):
    """
    Similar to 'select', select_case provides a way to express Fortran's
    SELECT CASE construct.  In this case, the selector value is matched
    against variables (not just constants) and ranges.  The structure is
    the same as 'select', but select_case allows for the expression of
    more complex match conditions.

    fir.select_case %arg : i32 [
          #fir.point, %0, ^bb1(%0 : i32),
          #fir.lower, %1, ^bb2(%2,%arg,%arg2,%1 : i32,i32,i32,i32),
          #fir.interval, %2, %3, ^bb3(%2,%arg2 : i32,i32),
          #fir.upper, %arg, ^bb4(%1 : i32),
          unit, ^bb5]
    """

    name = "fir.select_case"
    selector = operand_def()
    compareArgs = operand_def()
    targetArgs = operand_def()
    regs = var_region_def()


@irdl_op_definition
class SelectOp(IRDLOperation):
    """
    A multiway branch terminator with similar semantics to C's 'switch'
    statement.  A selector value is matched against a list of constants
    of the same type for a match.  When a match is found, control is
    transferred to the corresponding basic block.  A 'select' must have
    at least one basic block with a corresponding unit match, and
    that block will be selected when all other conditions fail to match.

    fir.select %arg:i32 [1, ^bb1(%0 : i32),
                         2, ^bb2(%2,%arg,%arg2 : i32,i32,i32),
                        -3, ^bb3(%arg2,%2 : i32,i32),
                         4, ^bb4(%1 : i32),
                      unit, ^bb5]
    """

    name = "fir.select"
    selector = operand_def()
    compareArgs = operand_def()
    targetArgs = operand_def()
    regs = var_region_def()


@irdl_op_definition
class SelectRankOp(IRDLOperation):
    """
    Similar to 'select', select_rank provides a way to express Fortran's
    SELECT RANK construct.  In this case, the rank of the selector value
    is matched against constants of integer type.  The structure is the
    same as 'select', but select_rank determines the rank of the selector
    variable at runtime to determine the best match.

    fir.select_rank %arg:i32 [1, ^bb1(%0 : i32),
                              2, ^bb2(%2,%arg,%arg2 : i32,i32,i32),
                              3, ^bb3(%arg2,%2 : i32,i32),
                             -1, ^bb4(%1 : i32),
                           unit, ^bb5]
    """

    name = "fir.select_rank"
    selector = operand_def()
    compareArgs = operand_def()
    targetArgs = operand_def()
    regs = var_region_def()


@irdl_op_definition
class SelectTypeOp(IRDLOperation):
    """
    Similar to 'select', select_type provides a way to express Fortran's
    SELECT TYPE construct.  In this case, the type of the selector value
    is matched against a list of type descriptors.  The structure is the
    same as 'select', but select_type determines the type of the selector
    variable at runtime to determine the best match.

    fir.select_type %arg : !fir.box<()> [
        #fir.type_is<!fir.type<type1>>, ^bb1(%0 : i32),
        #fir.type_is<!fir.type<type2>>, ^bb2(%2 : i32),
        #fir.class_is<!fir.type<type3>>, ^bb3(%2 : i32),
        #fir.type_is<!fir.type<type4>>, ^bb4(%1,%3 : i32,f32),
        unit, ^bb5]
    """

    name = "fir.select_type"
    selector = operand_def()
    compareArgs = operand_def()
    targetArgs = operand_def()
    regs = var_region_def()


@irdl_op_definition
class ShapeOp(IRDLOperation):
    """
    The arguments are an ordered list of integral type values that define the
    runtime extent of each dimension of an array. The shape information is
    given in the same row-to-column order as Fortran. This abstract shape value
    must be applied to a reified object, so all shape information must be
    specified.  The extent must be nonnegative.

    %d = fir.shape %row_sz, %col_sz : (index, index) -> !fir.shape<2>
    """

    name = "fir.shape"
    extents = var_operand_def()
    result_0 = result_def()


@irdl_op_definition
class ShapeShiftOp(IRDLOperation):
    """
    The arguments are an ordered list of integral type values that is a multiple
    of 2 in length. Each such pair is defined as: the lower bound and the
    extent for that dimension. The shifted shape information is given in the
    same row-to-column order as Fortran. This abstract shifted shape value must
    be applied to a reified object, so all shifted shape information must be
    specified.  The extent must be nonnegative.

    %d = fir.shape_shift %lo, %extent : (index, index) -> !fir.shapeshift<1>
    """

    name = "fir.shape_shift"
    pairs = var_operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class ShiftOp(IRDLOperation):
    """
    The arguments are an ordered list of integral type values that define the
    runtime lower bound of each dimension of an array. The shape information is
    given in the same row-to-column order as Fortran. This abstract shift value
    must be applied to a reified object, so all shift information must be
    specified.

    %d = fir.shift %row_lb, %col_lb : (index, index) -> !fir.shift<2>
    """

    name = "fir.shift"
    origins = var_operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class SliceOp(IRDLOperation):
    """
    The array slicing arguments are an ordered list of integral type values
    that must be a multiple of 3 in length.  Each such triple is defined as:
    the lower bound, the upper bound, and the stride for that dimension, as in
    Fortran syntax. Both bounds are inclusive. The array slice information is
    given in the same row-to-column order as Fortran. This abstract slice value
    must be applied to a reified object, so all slice information must be
    specified.  The extent must be nonnegative and the stride must not be zero.

    %d = fir.slice %lo, %hi, %step : (index, index, index) -> !fir.slice<1>

    To support generalized slicing of Fortran's dynamic derived types, a slice
    op can be given a component path (narrowing from the product type of the
    original array to the specific elemental type of the sliced projection).

    %fld = fir.field_index component, !fir.type<t{...component:ct...}>
    %d = fir.slice %lo, %hi, %step path %fld :
        (index, index, index, !fir.field) -> !fir.slice<1>

    Projections of '!fir.char' type can be further narrowed to invariant
    substrings.

      %d = fir.slice %lo, %hi, %step substr %offset, %width :
          (index, index, index, index, index) -> !fir.slice<1>
    """

    name = "fir.slice"
    triples = operand_def()
    fields = operand_def()
    substr = operand_def()
    result_0 = result_def()
    regs = var_region_def()


@irdl_op_definition
class StoreOp(IRDLOperation):
    """
    Store an ssa-value (virtual register) to a memory reference.  The stored
    value must be of the same type as the referent type of the memory
    reference.

    %v = ... : f64
    %p = ... : !fir.ptr<f64>
    fir.store %v to %p : !fir.ptr<f64>

    The above store changes the value to which the pointer is pointing and not
    the pointer itself. The operation is undefined if the memory reference,
    '%p', is undefined or null.
    """

    name = "fir.store"
    value = operand_def()
    memref = operand_def()
    regs = var_region_def()


@irdl_op_definition
class StringLitOp(IRDLOperation):
    """
    An FIR constant that represents a sequence of characters that correspond
    to Fortran's CHARACTER type, including a LEN.  We support CHARACTER values
    of different KINDs (different constant sizes).

    ```mlir
    %1 = fir.string_lit "Hello, World!"(13) : !fir.char<1> // ASCII
    %2 = fir.string_lit [158, 2345](2) : !fir.char<2>      // Wide chars
    ```
    """

    name = "fir.string_lit"
    size = attr_def(IntegerAttr[IntegerType])
    value = attr_def(StringAttr)
    result_0 = result_def()


@irdl_op_definition
class SubcOp(IRDLOperation):
    name = "fir.subc"
    lhs = operand_def()
    rhs = operand_def()
    fastmath = opt_prop_def(FastMathFlagsAttr)
    result = result_def()
    regs = var_region_def()


@irdl_op_definition
class UnboxcharOp(IRDLOperation):
    """
    Unboxes a value of 'boxchar' type into a pair consisting of a memory
    reference to the CHARACTER data and the LEN type parameter.

    %45   = ... : !fir.boxchar<1>
    %46:2 = fir.unboxchar %45 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i32)
    """

    name = "fir.unboxchar"
    boxchar = operand_def()
    result_0 = result_def()
    result_1 = result_def()
    regs = var_region_def()


@irdl_op_definition
class UnboxprocOp(IRDLOperation):
    """
    Unboxes a value of 'boxproc' type into a pair consisting of a procedure
    pointer and a pointer to a host context.

    %47   = ... : !fir.boxproc<() -> i32>
    %48:2 = fir.unboxproc %47 : (!fir.ref<() -> i32>, !fir.ref<tuple<f32, i32>>)
    """

    name = "fir.unboxproc"
    boxproc = operand_def()
    result_0 = result_def()
    refTuple = result_def()
    regs = var_region_def()


@irdl_op_definition
class UndefinedOp(IRDLOperation):
    """
    Constructs an ssa-value of the specified type with an undefined value.
    This operation is typically created internally by the mem2reg conversion
    pass. An undefined value can be of any type except '!fir.ref<T>'.

    %a = fir.undefined !fir.array<10 x !fir.type<T>>

    The example creates an array shaped ssa-value. The array is rank 1, extent
    10, and each element has type '!fir.type<T>'.
    """

    name = "fir.undefined"
    intype = result_def()
    regs = var_region_def()


@irdl_op_definition
class UnreachableOp(IRDLOperation):
    """
    Terminates a basic block with the assertion that the end of the block
    will never be reached at runtime.  This instruction can be used
    immediately after a call to the Fortran runtime to terminate the
    program, for example.  This instruction corresponds to the LLVM IR
    instruction 'unreachable'.

    fir.unreachable
    """

    name = "fir.unreachable"
    regs = var_region_def()

    traits = traits_def(IsTerminator())


@irdl_op_definition
class ZeroBitsOp(IRDLOperation):
    """
    Constructs an ssa-value of the specified type with a value of zero for all
    bits.

    %a = fir.zero_bits !fir.box<!fir.array<10 x !fir.type<T>>>

    The example creates a value of type box where all bits are zero.
    """

    name = "fir.zero_bits"
    intype = result_def()
    regs = var_region_def()


FIR = Dialect(
    "fir",
    [
        AbsentOp,
        AddcOp,
        AddressOfOp,
        AllocmemOp,
        AllocaOp,
        ArrayAccessOp,
        ArrayAmendOp,
        ArrayCoorOp,
        ArrayFetchOp,
        ArrayLoadOp,
        ArrayMergeStoreOp,
        ArrayModifyOp,
        ArrayUpdateOp,
        BoxAddrOp,
        BoxcharLenOp,
        BoxDimsOp,
        BoxElesizeOp,
        BoxIsallocOp,
        BoxIsarrayOp,
        BoxIsptrOp,
        BoxOffsetOp,
        BoxprocHostOp,
        BoxRankOp,
        BoxTdescOp,
        CallOp,
        CharConvertOp,
        CmpcOp,
        ConstcOp,
        ConvertOp,
        CoordinateOfOp,
        DeclareOp,
        DtEntryOp,
        DispatchOp,
        DispatchTableOp,
        DivcOp,
        DoLoopOp,
        DummyScopeOp,
        EmboxcharOp,
        EmboxOp,
        EmboxprocOp,
        ExtractValueOp,
        FieldIndexOp,
        EndOp,
        FreememOp,
        GentypedescOp,
        GlobalLenOp,
        GlobalOp,
        HasValueOp,
        IfOp,
        InsertOnRangeOp,
        InsertValueOp,
        IsPresentOp,
        IterateWhileOp,
        LenParamIndexOp,
        LoadOp,
        MulcOp,
        NegcOp,
        NoReassocOp,
        ReboxOp,
        ResultOp,
        SaveResultOp,
        SelectCaseOp,
        SelectOp,
        SelectRankOp,
        SelectTypeOp,
        ShapeOp,
        ShapeShiftOp,
        ShiftOp,
        SliceOp,
        StoreOp,
        StringLitOp,
        SubcOp,
        UnboxcharOp,
        UnboxprocOp,
        UndefinedOp,
        UnreachableOp,
        ZeroBitsOp,
    ],
    [
        FortranVariableFlagsAttr,
        ReferenceType,
        DeferredAttr,
        DummyScopeType,
        LLVMPointerType,
        PointerType,
        LogicalType,
        NoneType,
        SequenceType,
        CharacterType,
        ShapeType,
        ShapeShiftType,
        HeapType,
        BoxType,
        BoxCharType,
        ShiftType,
        ComplexType,
    ],
)
