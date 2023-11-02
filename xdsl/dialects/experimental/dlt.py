from __future__ import annotations

from typing import Iterable, Iterator

from xdsl.dialects.builtin import ArrayAttr, StringAttr, IntegerAttr, IntAttr, i64, IntegerType, IndexType, AnyFloat, \
    AnyFloatAttr, FunctionType
from xdsl.ir import TypeAttribute, Dialect, AttributeCovT
from xdsl.irdl import *
from xdsl.parser import AttrParser
from xdsl.traits import IsTerminator, HasParent
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
        isa(self.region.block.last_op, YieldOp)
        if self.res.type != self.region.block.last_op.output_type():
            raise VerifyException("Struct result type must be the dlt.type corrosponding to the elements in the yield op")
        pass

@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "dlt.yield"
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
    name = "dlt.select"
    tree: OperandDef = operand_def(TypeType)
    dimensions: AttributeDef = attr_def(ArrayAttr[StringAttr])
    members: AttributeDef = attr_def(SetAttr[MemberAttr])
    values: VarOperand = var_operand_def(IndexType)

    res: OpResult = result_def(TypeType)

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

    @classmethod
    def calculateResultType(cls, input_type: TypeType, members: Iterable[MemberAttr], dimension_names: Iterable[StringAttr]) -> TypeType:
        current_type = input_type
        for m in members:
            current_type = current_type.selectMember(m)
        for d in dimension_names:
            current_type = current_type.selectDimension(d)
        return current_type


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
        # super().print(printer)


#TODO


@irdl_op_definition
class GetOp(IRDLOperation):
    name = "dlt.get"
#TODO

@irdl_op_definition
class SetOp(IRDLOperation):
    name = "dlt.set"
#TODO

@irdl_op_definition
class IterateOp(IRDLOperation):
    name = "dlt.iterate"
#TODO

@irdl_op_definition
class InitOp(IRDLOperation):
    name = "dlt.init"
#TODO





DLT = Dialect("DLT",
    [#ops
        StructOp,
        YieldOp,
        IndexingOp,
        MemberOp,
        PrimitiveOp,
        ConstOp,
        DenseOp,
        UnpackedCoordinateFormatOp,
        IndexAffineOp,
        SelectOp,
        GetOp,
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
        IndexedTypeType
    ]
)