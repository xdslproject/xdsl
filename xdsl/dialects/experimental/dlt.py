"""
Data Layout Trees 'DLT' is a dialect / DSL for specifying the data-layout of multiple tensors in a unified tree
structure. The objective is to provide access to named tensors by named dimensions such that the actual layout can be
modified at compile time without changing the code that uses the tensors. This then allows us to have complex layouts
that combine structures with members, dense dimensions, and compressed/sparse layouts as well as affine transformations
on dimensions, masking. From a normalised description of the data that needs to exist (and potentially knowledge about
redundancy / structured sparsity) a physically layout can be produced. Then with rewrite rules that preserve soundness
(that all values that must be stored are stored) we can modify the structure automatically to produce a large search
space of different physical layouts to then find more optimal solutions that produce more efficient code.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from xdsl.dialects import builtin
from xdsl.dialects.builtin import ArrayAttr, StringAttr, IntegerAttr, i64, IntegerType, IndexType, AnyFloat, \
    AnyFloatAttr
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import TypeAttribute, Dialect, AttributeCovT, BlockArgument
from xdsl.irdl import *
from xdsl.parser import AttrParser
from xdsl.traits import IsTerminator, HasParent, SingleBlockImplicitTerminator, HasAncestor
from xdsl.utils.hints import isa

@dataclass
class SetOfConstraint(AttrConstraint):
    """
    A constraint that enforces an SetData whose elements all satisfy
    the elem_constr.
    """

    elem_constr: AttrConstraint

    def __init__(self, constr: Attribute | type[Attribute] | AttrConstraint):
        self.elem_constr = attr_constr_coercion(constr)

    def verify(self, attr: Attribute, constraint_vars: dict[str, Attribute]) -> None:
        if not isinstance(attr, SetAttr):
            raise VerifyException(f"expected SetData attribute, but got {attr}")
        for e in cast(SetAttr[Attribute], attr).data:
            self.elem_constr.verify(e, constraint_vars)


@irdl_attr_definition
class SetAttr(GenericData[frozenset[AttributeCovT, ...]], Iterable[AttributeCovT]):
    """Based on ArrayAttr but for sets. Used for when the order shouldn't matter. By Default putting duplicate values in
    raises an error as it is expected that duplicates should not be in the input for well-formed code.
    This implementation requires that Attributes contained within are hashable which is not necessarily true. """
    name = "dlt.set"

    def __init__(self, param: Iterable[AttributeCovT]) -> None:
        p = list(param)
        s = frozenset(p)
        if len(s) != len(p):
            raise ValueError(f"Cannot form SetAttr with duplicate elements in: {str(p)}")

        super().__init__(frozenset(param))

    @classmethod
    def from_duplicates(cls, param: Iterable[AttributeCovT]) -> SetAttr[AttributeCovT]:
        return SetAttr(set(param))

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> frozenset[AttributeCovT, ...]:
        data = parser.parse_comma_separated_list(
            parser.Delimiter.BRACES, parser.parse_attribute
        )
        # the type system can't ensure that the elements are of type _SetAttrT
        result = cast(tuple[AttributeCovT, ...], tuple(data))
        return result

    def print_parameter(self, printer: Printer) -> None:
        """We sort the elements by their str() value just to maintain determinism of output"""
        printer.print_string("{")
        values = list(self.data)
        sorted_values = [values[i] for s,i in sorted([(str(s),i) for i,s in enumerate(values)])]
        printer.print_list(sorted_values, printer.print_attribute)
        printer.print_string("}")

    @staticmethod
    def generic_constraint_coercion(args: tuple[Any]) -> AttrConstraint:
        if len(args) == 1:
            return SetOfConstraint(irdl_to_attr_constraint(args[0]))
        if len(args) == 0:
            return SetOfConstraint(AnyAttr())
        raise TypeError(
            f"Attribute SetAttr expects at most 1 type"
            f" parameter, but {len(args)} were given"
        )

    def verify(self) -> None:
        for idx, val in enumerate(self.data): # this check shouldn't be needed? as the Type checking should sort this out
            if not isinstance(val, Attribute):
                raise VerifyException(
                    f"{self.name} data expects attribute list, but {idx} "
                    f"element is of type {type(val)}"
                )

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> Iterator[AttributeCovT]:
        return iter(self.data)

    def without(self, val:AttributeCovT):
        return SetAttr(self.data.difference([val]))



@irdl_attr_definition
class MemberAttr(ParametrizedAttribute):
    name = "dlt.member"
    structName: ParameterDef[StringAttr]
    memberName: ParameterDef[StringAttr]


    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            result = MemberAttr.internal_parse_parameters(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_parameters(printer)

    @classmethod
    def internal_parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        sn = StringAttr(parser.parse_identifier())
        parser.parse_punctuation(":")
        mn = StringAttr(parser.parse_identifier())
        return [sn, mn]

    def internal_print_parameters(self, printer: Printer) -> None:
        printer.print(self.structName.data)
        printer.print(":")
        printer.print(self.memberName.data)

    @classmethod
    def internal_print_members(cls, members: SetAttr[MemberAttr], printer: Printer) -> None:
        printer.print("{")
        m_values = list(members.data)
        sorted_m_values = [m_values[i] for s, i in sorted([(str(s), i) for i, s in enumerate(m_values)])]
        printer.print_list(sorted_m_values, lambda v: v.internal_print_parameters(printer))
        printer.print("}")

@irdl_attr_definition
class DimensionAttr(ParametrizedAttribute):
    name = "dlt.dimension"
    dimensionName: ParameterDef[StringAttr]
    extent: ParameterDef[StringAttr | IntegerAttr]
    # extent: ParameterDef[StringAttr|(IntegerAttr[Annotated[IntegerType, i64]])]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        with parser.in_angle_brackets():
            result = DimensionAttr.internal_parse_parameters(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_parameters(printer)

    @classmethod
    def internal_parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        dn = StringAttr(parser.parse_identifier())
        parser.parse_punctuation(":")
        i = parser.parse_optional_integer(allow_boolean=False, allow_negative=False)
        if i is None:
            e = parser.parse_optional_identifier()
            if e is None:
                parser.raise_error("Int or identifier expected")
            ext = StringAttr(e)
        else:
            ext = IntegerAttr(i, i64)
        return [dn, ext]

    def internal_print_parameters(self, printer: Printer) -> None:
        printer.print(self.dimensionName.data)
        printer.print(":")
        if isinstance(self.extent, StringAttr):
            printer.print(self.extent.data)
        elif isinstance(self.extent, IntegerAttr):
            printer.print(self.extent.value.data)
        else:
            raise ValueError()

    def verify(self) -> None:
        if isinstance(self.extent, IntegerAttr):
            assert self.extent.value.data >= 0

@irdl_attr_definition
class ElementAttr(ParametrizedAttribute):
    name = "dlt.element"
    member_specifiers: ParameterDef[SetAttr[MemberAttr]]
    dimensions: ParameterDef[SetAttr[DimensionAttr]]
    base_type: ParameterDef[Attribute]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[Attribute,...]:
        with parser.in_angle_brackets():
            result = ElementAttr.internal_parse_parameters(parser)
        return result

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            self.internal_print_parameters(printer)

    @classmethod
    def internal_parse_parameters(cls, parser: AttrParser) -> tuple[Attribute,...]:
        parser.parse_punctuation("(")
        ms = parser.parse_comma_separated_list(
            parser.Delimiter.BRACES, lambda: MemberAttr(tuple(MemberAttr.internal_parse_parameters(parser)))
        )
        ms_set = SetAttr(tuple(ms))
        parser.parse_punctuation(",")
        dims = []
        baseType = parser.parse_optional_type()
        while baseType is None:
            dims.append(DimensionAttr(tuple(DimensionAttr.internal_parse_parameters(parser))))
            parser.parse_punctuation("->")
            baseType = parser.parse_optional_type()
        parser.parse_punctuation(")")
        dims = SetAttr(tuple(dims))

        return tuple([ms_set, dims, baseType])

    def internal_print_parameters(self, printer: Printer) -> None:
        printer.print("(")
        MemberAttr.internal_print_members(self.member_specifiers, printer)
        printer.print(",")

        d_values = list(self.dimensions.data)
        sorted_d_values = [d_values[i] for s, i in sorted([(str(s), i) for i, s in enumerate(d_values)])]
        for dim in sorted_d_values:
            dim.internal_print_parameters(printer)
            printer.print("->")
        printer.print(self.base_type)
        printer.print(")")

    def verify(self) -> None:
        dim_names = [dim.dimensionName for dim in self.dimensions]
        if len(dim_names) != len(set(dim_names)):
            raise VerifyException("Dimensions in an dlt.element must not have repeated dimension names.")

    def get_dimension(self, name: StringAttr):
        for dim in self.dimensions:
            if name == dim.dimensionName:
                return dim
        return None

    def get_dimension_names(self):
        names = []
        for dim in self.dimensions:
            names.append(dim.dimensionName)
        return names

    def __lt__(self, other):
        assert isinstance(other, ElementAttr)
        if self.member_specifiers.data < other.member_specifiers.data:
            return True
        elif self.member_specifiers.data > other.member_specifiers.data:
            return False

        if self.dimensions.data < other.dimensions.data:
            return True
        elif self.dimensions.data > other.dimensions.data:
            return False

        return False


@irdl_attr_definition
class TypeType(ParametrizedAttribute, TypeAttribute):
    name = "dlt.type"
    elements: ParameterDef[SetAttr[ElementAttr]]

    def __init__(self, elements: Iterable[ElementAttr]):
        if not isinstance(elements, SetAttr):
            elements = SetAttr(elements)
        super().__init__(tuple([elements]))

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> tuple[Attribute,...]:
        es = parser.parse_comma_separated_list(
            parser.Delimiter.ANGLE, lambda: ElementAttr(tuple(ElementAttr.internal_parse_parameters(parser)))
        )

        es_set = SetAttr(tuple(es))
        return tuple([es_set])

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            values = list(self.elements)
            sorted_values = [values[i] for s, i in sorted([(str(s), i) for i, s in enumerate(values)])]
            printer.print_list(sorted_values, lambda v: v.internal_print_parameters(printer))

    def verify(self) -> None:
        # sorting not required if we can use SetAttr
        # assert self.elements.data == sorted(self.elements.data),\
        #     "Elements in the type must be sorted"
        elems = [(elem.member_specifiers.data, elem.dimensions.data) for elem in self.elements]
        if len(elems) != len(set(elems)):
            raise VerifyException("Each element in the type must have a unique sets of memberSpecifiers")

    def selectMember(self, member: MemberAttr) -> TypeType:
        elems = [elem for elem in self.elements if member in elem.member_specifiers.data]
        new_elems = []
        for elem in elems:
            set = elem.member_specifiers.without(member)
            new_elem = ElementAttr(tuple([set, elem.dimensions, elem.base_type]))
            new_elems.append(new_elem)
        return TypeType(new_elems)

    def selectDimension(self, dimension_name: StringAttr) -> TypeType:
        elems = [elem for elem in self.elements if any(dimension_name == dim.dimensionName for dim in elem.dimensions)]
        new_elems = []
        for elem in elems:
            set = SetAttr([dim for dim in elem.dimensions.data if dim.dimensionName != dimension_name])
            new_elem = ElementAttr(tuple([elem.member_specifiers, set, elem.base_type]))
            new_elems.append(new_elem)
        return TypeType(new_elems)

    def get_single_element(self) -> None | ElementAttr:
        if len(self.elements) == 1:
            return list(self.elements)[0]
        else:
            return None


@irdl_attr_definition
class PtrType(ParametrizedAttribute, TypeAttribute):
    name = "dlt.ptr"
    contents_type: ParameterDef[TypeType]

    def __init__(self, type_type: TypeType):
        super().__init__(tuple([type_type]))


@irdl_attr_definition
class IndexRangeType(ParametrizedAttribute, TypeAttribute):
    name = "dlt.indexRange"

@irdl_attr_definition
class IndexedTypeType(ParametrizedAttribute, TypeAttribute):
    name = "dlt.indexedType"
    base: ParameterDef[TypeType]
    index: ParameterDef[IndexType | IndexRangeType]


@irdl_op_definition
class StructOp(IRDLOperation):
    name = "dlt.struct"

    res: OpResult = result_def(TypeType)
    region: Region = region_def("single_block")

    def verify_(self) -> None:
        isa(self.region.block.last_op, StructYieldOp)
        if self.res.type != self.region.block.last_op.output_type():
            raise VerifyException("Struct result type must be the dlt.type corrosponding to the elements in the yield op")
        pass

@irdl_op_definition
class StructYieldOp(IRDLOperation):
    name = "dlt.structYield"
    arguments: VarOperand = var_operand_def(TypeType)

    traits = traits_def(
        lambda: frozenset([IsTerminator(), HasParent(StructOp)])
    )

    def verify_(self) -> None:
        elements = [elem for arg in self.arguments if isinstance(arg.type, TypeType) for elem in arg.type.elements]
        TypeType(elements) # check that making this type doesn't cause an error

    def output_type(self):
        elements = [elem for arg in self.arguments if isinstance(arg.type, TypeType) for elem in arg.type.elements]
        type = TypeType(elements)
        return type

#TODO

@irdl_op_definition
class IndexingOp(IRDLOperation):
    name = "dlt.indexing"
#TODO

@irdl_op_definition
class MemberOp(IRDLOperation):
    name = "dlt.member"
#TODO


AcceptedTypes: TypeAlias = IntegerType | AnyFloat | IndexType | IndexRangeType
@irdl_op_definition
class PrimitiveOp(IRDLOperation):
    name = "dlt.primitive"
    of: AcceptedTypes = attr_def(AcceptedTypes)
    res: OpResult = result_def(TypeType)

    def __init__(self, of: AcceptedTypes):
        type = TypeType([ElementAttr(tuple([SetAttr([]), SetAttr([]), of]))])
        super().__init__(attributes={"of":of}, result_types=[type])
#TODO

@irdl_op_definition
class ConstOp(IRDLOperation):
    name = "dlt.const"
    value: AnyFloatAttr | IntegerAttr = attr_def(AnyFloatAttr | IntegerAttr)
    res: OpResult = result_def(TypeType)

#TODO

@irdl_op_definition
class DenseOp(IRDLOperation):
    name = "dlt.dense"
    child: OperandDef = operand_def(TypeType)
    dimension: DimensionAttr = attr_def(DimensionAttr)
    res: OpResult = result_def(TypeType)

    def verify_(self) -> None:
        elements = [elem for elem in self.child.type.elements]
        new_elements = []
        for elem in elements:
            dims = list(elem.dimensions)
            dims.append(self.dimension)
            new_dims = SetAttr(dims)
            new_elem = ElementAttr(tuple([elem.member_specifiers, new_dims, elem.base_type]))
            new_elements.append(new_elem)
        new_type = TypeType(new_elements)
        res_type = self.res.type
        if new_type != res_type:
            raise VerifyException("Result type does not match input type with added dimension")

#TODO

@irdl_op_definition
class UnpackedCoordinateFormatOp(IRDLOperation):
    name = "dlt.upcoo"
#TODO

@irdl_op_definition
class IndexAffineOp(IRDLOperation):
    name = "dlt.indexAffine"
#TODO



@irdl_op_definition
class SelectOp(IRDLOperation):
    name = "dlt.select" # take a ptrType and constrain a member field or a dimension value.
    tree: OperandDef = operand_def(PtrType)
    dimensions: AttributeDef = attr_def(ArrayAttr[StringAttr])
    members: AttributeDef = attr_def(SetAttr[MemberAttr])
    values: VarOperand = var_operand_def(IndexType)

    res: OpResult = result_def(PtrType)

    def verify_(self) -> None:
        calc_type = SelectOp.calculateResultType(self.tree.type, self.members, self.dimensions)
        if calc_type != self.res.type:
            raise VerifyException(f"dlt.select, result type mismatch. got: {self.res.type}, expected: {calc_type}")
        # assert not isinstance(self.tree, BlockArgument)
        # print("verify")
        pass

    @classmethod
    def parse(cls: type[SelectOp], parser: Parser) -> SelectOp:
        # dlt.select{root:e, node:x}(A:0, B:10) from %1
        ms = parser.parse_comma_separated_list(
            parser.Delimiter.BRACES, lambda: MemberAttr(tuple(MemberAttr.internal_parse_parameters(parser)))
        )
        members = SetAttr(ms)
        def parseDim() -> tuple[StringAttr, SSAValue]:
            ident = parser.parse_identifier()
            parser.parse_punctuation(":")
            operand = parser.parse_operand()
            dim_name = StringAttr(ident)
            return (dim_name, operand)
        dims = parser.parse_comma_separated_list(parser.Delimiter.PAREN, parseDim)
        dim_names, dim_operands = zip(*dims)
        dimensions = ArrayAttr(dim_names)
        parser.parse_keyword("from")
        tree = parser.parse_operand()

        if parser.parse_optional_punctuation(":"):
            parser.parse_punctuation("(")
            tree_type = parser.parse_type()
            if tree.type != tree_type:
                parser.raise_error(f"Type given: {tree_type} does not match expected: {tree.type}")
            parser.parse_punctuation(")")
            parser.parse_punctuation("->")
            res_type = parser.parse_type()
            res_type_clac = SelectOp.calculateResultType(tree.type, members, dimensions)
            if res_type != res_type_clac:
                parser.raise_error(f"parsed type {res_type} does not match expected type f{res_type_clac}")
        else:
            res_type = SelectOp.calculateResultType(tree.type, members, dimensions)

        selectOp = SelectOp(operands=[tree, dim_operands], attributes={"dimensions":dimensions, "members":members}, result_types=[res_type])
        return selectOp

    @staticmethod
    def calculateResultType(input_type: PtrType, members: Iterable[MemberAttr], dimension_names: Iterable[StringAttr]) -> PtrType:
        current_type = input_type.contents_type
        for m in members:
            current_type = current_type.selectMember(m)
        for d in dimension_names:
            current_type = current_type.selectDimension(d)
        return PtrType(current_type)


    def print(self, printer: Printer):
        MemberAttr.internal_print_members(self.members, printer)
        def print_d_v(dv: tuple[StringAttr, SSAValue]):
            d, v = dv
            printer.print(d.data)
            printer.print(":")
            printer.print(v)
        printer.print("(")
        printer.print_list(zip(self.dimensions, self.values), print_d_v)
        printer.print(")")
        printer.print(" from ")
        printer.print(self.tree)
        printer.print(" : ")
        printer.print("(")
        printer.print(self.tree.type)
        printer.print(")")
        printer.print(" -> ")
        printer.print(self.res.type)



@irdl_op_definition
class GetOp(IRDLOperation):
    name = "dlt.get" # take a PtrType that points only to primitives (no member fields or dimensions) and get the value
    tree: OperandDef = operand_def(PtrType)
    get_type: AttributeDef = attr_def(AcceptedTypes)
    res: OpResult = result_def(AcceptedTypes)
    #TODO Verify the tree layout type accesses one and only one element (with no dims or member names)


@irdl_op_definition
class SetOp(IRDLOperation):
    name = "dlt.set" # take a PtrType that points only to primitives (no member fields or dimensions) and set the value
    tree: OperandDef = operand_def(PtrType)
    value: OperandDef = operand_def(AcceptedTypes)
    set_type: AttributeDef = attr_def(AcceptedTypes)
    # TODO Verify the tree layout type accesses one and only one element (with no dims or member names)


@irdl_op_definition
class CopyOp(IRDLOperation):
    name = "dlt.copy" # take src and dst Ptrtypes and copy all values of the copy_type primitive from one to the other.
    src: OperandDef = operand_def(PtrType)
    dst: OperandDef = operand_def(PtrType)
    src_dimensions: AttributeDef = attr_def(ArrayAttr[DimensionAttr])
    dst_dimensions: AttributeDef = attr_def(ArrayAttr[DimensionAttr])
    copy_type: AttributeDef = attr_def(AcceptedTypes)

    # TODO Verify the tree layout types match perfectly


@irdl_op_definition
class ClearOp(IRDLOperation):
    name = "dlt.clear"  # take a Ptrtype and set all the values of clear_type to 0 - possibly changing the runtime sparsity
    tree: OperandDef = operand_def(PtrType)
    # dimensions: AttributeDef = attr_def(SetAttr[DimensionAttr])
    clear_type: AttributeDef = attr_def(AcceptedTypes)

    traits = traits_def(
        lambda: frozenset([HasAncestor(builtin.ModuleOp, True)])
    )

@irdl_op_definition
class IterateYieldOp(AbstractYieldOperation[Attribute]):
    name = "dlt.iterateYield"

    traits = traits_def(
        lambda: frozenset([IsTerminator(), HasParent(IterateOp)])
    )

@irdl_op_definition
class IterateOp(IRDLOperation):
    name = "dlt.iterate" # Iterate over a multiple dimension-extent pairs, given some context tensors that might be used inside.
    #TODO attribute for type of itteration - Non-zero vs whole space
    dimensions: ArrayAttr[StringAttr] = attr_def(ArrayAttr[StringAttr])
    extents: VarOperand = var_operand_def(IndexType)
    order: StringAttr = attr_def(StringAttr) # "nested" | "none" | "stored"
    tensors: VarOperand = var_operand_def(PtrType)


    iter_args: VarOperand = var_operand_def(AnyAttr())

    res: VarOpResult = var_result_def(AnyAttr())

    body: Region = region_def("single_block")

    traits = frozenset([SingleBlockImplicitTerminator(IterateYieldOp), HasAncestor(builtin.ModuleOp, True)])
    irdl_options = [AttrSizedOperandSegments()]


    def __init__(
        self,
        dimensions: ArrayAttr[StringAttr],
        extents: Sequence[SSAValue | IndexType],
        order: StringAttr,
        tensors: Sequence[SSAValue | PtrType],
        iter_args: Sequence[SSAValue | Operation],
        body: Region | Sequence[Operation] | Sequence[Block] | Block,
    ):
        if isinstance(body, Block):
            body = [body]
        assert order.data in ["nested", "none", "stored"]

        super().__init__(
            operands=[extents, tensors, iter_args],
            result_types=[[SSAValue.get(a).type for a in iter_args]],
            regions=[body],
            attributes={"dimensions":dimensions, "order":order}
        )

    def verify_(self):
        if (len(self.iter_args) + len(self.dimensions)) != len(self.body.block.args):
            raise VerifyException(
                f"Wrong number of block arguments, expected {len(self.iter_args)+len(self.dimensions)}, got "
                f"{len(self.body.block.args)}. The body must have the induction "
                f"variables and loop-carried variables as arguments."
            )
        if self.body.block.args and (iter_vars := [(i,self.body.block.args[i]) for i in range(len(self.dimensions))]):
            if any(var_i:= i and not isinstance(var.type, IndexType) for i, var in iter_vars):
                raise VerifyException(
                    f"The first {len(self.dimensions)} block argument(s) of the body must be of type index, but arg "
                    f"{var_i} is of type {self.body.block.args[var_i].type} instead of index."
                )
        for idx, arg in enumerate(self.iter_args):
            if self.body.block.args[idx + len(self.dimensions)].type != arg.type:
                raise VerifyException(
                    f"Block arguments with wrong type, expected {arg.type}, "
                    f"got {self.body.block.args[idx].type}. Arguments after the "
                    f"induction variables must match the carried variables."
                )
        if len(self.body.ops) > 0 and isinstance(self.body.block.last_op, IterateYieldOp):
            yieldop = self.body.block.last_op
            if len(yieldop.arguments) != len(self.iter_args):
                raise VerifyException(
                    f"Expected {len(self.iter_args)} args, got {len(yieldop.arguments)}. "
                    f"The dlt.iterate must yield its carried variables."
                )
            for idx, arg in enumerate(yieldop.arguments):
                if self.iter_args[idx].type != arg.type:
                    raise VerifyException(
                        f"Expected {self.iter_args[idx].type}, got {arg.type}. The "
                        f"dlt.iterate's dlt.iterateYield must match carried variables types."
                    )

    # def print(self, printer: Printer):
    #     block = self.body.block
    #     indices = [block.args[i] for i in range(len(self.dimensions))]
    #     iter_args = [block.args[i] for i in range(len(self.dimensions), len(block.args))]
    #     printer.print_string(" ")
    #     printer.print_list(
    #         zip(indices, self.iter_args),
    #         lambda pair: print_assignment(printer, *pair),
    #     )
    #     printer.print_ssa_value(index)
    #     printer.print_string(" = ")
    #     printer.print_ssa_value(self.lb)
    #     printer.print_string(" to ")
    #     printer.print_ssa_value(self.ub)
    #     printer.print_string(" step ")
    #     printer.print_ssa_value(self.step)
    #     printer.print_string(" ")
    #     if iter_args:
    #         printer.print_string("iter_args(")
    #         printer.print_list(
    #             zip(iter_args, self.iter_args),
    #             lambda pair: print_assignment(printer, *pair),
    #         )
    #         printer.print_string(") -> (")
    #         printer.print_list((a.type for a in iter_args), printer.print_attribute)
    #         printer.print_string(") ")
    #     printer.print_region(
    #         self.body,
    #         print_entry_block_args=False,
    #         print_empty_block=False,
    #         print_block_terminators=bool(iter_args),
    #     )
#TODO

@irdl_op_definition
class InitOp(IRDLOperation):
    name = "dlt.init" # take a dlt layout as a TypeType and do the memory allocations etc to form a ptrType

    layout: OperandDef = operand_def(TypeType)
    initialValues: VarOperand = var_operand_def(PtrType)
    res: OpResult = result_def(PtrType)

#TODO

@irdl_op_definition
class AssertLayoutOp(IRDLOperation):
    name = "dlt.assert" # take a dlt layout as a TypeType and assert that a given memref has the layout to form a ptrType
#TODO


@irdl_op_definition
class layoutScopeOp(IRDLOperation):
    name = "dlt.layoutScope" # Be the point that all internal dlt operations use a reference and source for layout information
    body: Region = region_def("single_block")

#TODO







DLT = Dialect("DLT",
    [#ops
        StructOp,
        StructYieldOp,
        IndexingOp,
        MemberOp,
        PrimitiveOp,
        ConstOp,
        DenseOp,
        UnpackedCoordinateFormatOp,
        IndexAffineOp,
        SelectOp,
        GetOp,
        CopyOp,
        ClearOp,
        IterateYieldOp,
        IterateOp,
        InitOp
    ],
    [#attrs
        SetAttr,
        MemberAttr,
        DimensionAttr,
        ElementAttr,
        TypeType,
        IndexRangeType,
        IndexedTypeType,
        PtrType
    ]
)