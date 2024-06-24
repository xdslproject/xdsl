from collections.abc import Sequence
from itertools import pairwise
from typing import Annotated, cast

from xdsl.dialects import builtin, memref, stencil
from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType
from xdsl.dialects.experimental import dmp
from xdsl.ir import Attribute, Dialect, Operation, ParametrizedAttribute, SSAValue
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    Operand,
    ParameterDef,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_prop_def,
    prop_def,
    result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import HasAncestor, Pure
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_attr_definition
class ExchangeDeclarationAttr(ParametrizedAttribute):
    """
    A simplified version of dmp.exchange, from which it should be lowered
    """

    name = "csl_stencil.exchange"

    neighbor_param: ParameterDef[builtin.DenseArrayBase]

    def __init__(
        self,
        neighbor: Sequence[int] | builtin.DenseArrayBase,
    ):
        data_type = builtin.i64
        super().__init__(
            [
                (
                    neighbor
                    if isinstance(neighbor, builtin.DenseArrayBase)
                    else builtin.DenseArrayBase.from_list(data_type, neighbor)
                ),
            ]
        )

    @classmethod
    def from_dmp_exch_decl_attr(cls, src: dmp.ExchangeDeclarationAttr):
        return cls(src.neighbor)

    @property
    def neighbor(self) -> tuple[int, ...]:
        data = self.neighbor_param.as_tuple()
        assert isa(data, tuple[int, ...])
        return data

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<to {list(self.neighbor)}>")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        parser.parse_characters("<")
        parser.parse_characters("to")
        to = parser.parse_comma_separated_list(
            parser.Delimiter.SQUARE, parser.parse_integer
        )
        parser.parse_characters(">")

        return [builtin.DenseArrayBase.from_list(builtin.i64, to)]


@irdl_op_definition
class PrefetchOp(IRDLOperation):
    """
    An op to indicate buffer prefetches.
    """

    name = "csl_stencil.prefetch"

    input_stencil: Operand = operand_def(
        stencil.TempType[Attribute] | memref.MemRefType[Attribute]
    )

    swaps: builtin.ArrayAttr[ExchangeDeclarationAttr] | None = opt_prop_def(
        builtin.ArrayAttr[ExchangeDeclarationAttr]
    )

    size = prop_def(IntegerAttr[IntegerType])

    topo: dmp.RankTopoAttr | None = opt_prop_def(dmp.RankTopoAttr)

    result = result_def(memref.MemRefType)

    def __init__(
        self,
        input_stencil: SSAValue | Operation,
        size: IntegerAttr[IntegerType],
        topo: dmp.RankTopoAttr | None = None,
        swaps: Sequence[ExchangeDeclarationAttr] = [],
        result_type: memref.MemRefType[Attribute] | None = None,
    ):
        super().__init__(
            operands=[input_stencil],
            properties={"size": size, "topo": topo, "swaps": builtin.ArrayAttr(swaps)},
            result_types=[result_type],
        )


@irdl_op_definition
class AccessOp(IRDLOperation):
    """
    A CSL stencil access that operates on data prefetched by `csl_stencil.prefetch`
    """

    T = Annotated[Attribute, ConstraintVar("T")]

    name = "csl_stencil.access"
    op: Operand = operand_def(memref.MemRefType)
    offset: stencil.IndexAttr = attr_def(stencil.IndexAttr)
    offset_mapping = opt_attr_def(stencil.IndexAttr)
    result = result_def(TensorType)

    traits = frozenset([HasAncestor(stencil.ApplyOp), Pure()])

    def __init__(
        self,
        op: Operand,
        offset: stencil.IndexAttr,
        result_type: TensorType[Attribute],
        offset_mapping: stencil.IndexAttr | None = None,
    ):
        super().__init__(
            operands=[op],
            attributes={"offset": offset, "offset_mapping": offset_mapping},
            result_types=[result_type],
        )

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.op)
        printer.print_op_attributes(
            self.attributes,
            reserved_attr_names={"offset", "offset_mapping"},
            print_keyword=True,
        )

        mapping = self.offset_mapping
        if mapping is None:
            mapping = range(len(self.offset))
        offset = list(self.offset)

        printer.print("[")
        index = 0
        for i in range(len(self.offset)):
            if i in mapping:
                printer.print(offset[index])
                index += 1
            else:
                printer.print("_")
            if i != len(self.offset) - 1:
                printer.print(", ")
        printer.print("]")

        printer.print_string(" : ")
        printer.print_attribute(self.op.type)

    @classmethod
    def parse(cls, parser: Parser):
        temp = parser.parse_operand()

        index = 0
        offset = list[int]()
        offset_mapping = list[int]()
        parser.parse_punctuation("[")
        while True:
            o = parser.parse_optional_integer()
            if o is None:
                parser.parse_characters("_")
            else:
                offset.append(o)
                offset_mapping.append(index)
            if parser.parse_optional_punctuation("]"):
                break
            parser.parse_punctuation(",")
            index += 1

        attrs = parser.parse_optional_attr_dict_with_keyword(
            {"offset", "offset_mapping"}
        )
        attrs = attrs.data if attrs else dict[str, Attribute]()
        attrs["offset"] = stencil.IndexAttr.get(*offset)
        if offset_mapping:
            attrs["offset_mapping"] = stencil.IndexAttr.get(*offset_mapping)
        parser.parse_punctuation(":")
        res_type = parser.parse_attribute()
        if not isa(res_type, memref.MemRefType[Attribute]):
            parser.raise_error("Expected return type to be a memref")
        return cls.build(
            operands=[temp], result_types=[res_type.element_type], attributes=attrs
        )

    def verify_(self) -> None:
        assert isa(self.op.type, memref.MemRefType[Attribute])
        assert self.result.type == self.op.type.get_element_type()

        # As promised by HasAncestor(ApplyOp)
        trait = cast(HasAncestor, AccessOp.get_trait(HasAncestor, (stencil.ApplyOp,)))
        apply = trait.get_ancestor(self)
        assert isinstance(apply, stencil.ApplyOp)

        # TODO This should be handled by infra, having a way to verify things on ApplyOp
        # **before** its children.
        # cf https://github.com/xdslproject/xdsl/issues/1112
        apply.verify_()

        if self.offset_mapping is not None and len(self.offset_mapping) != len(
            self.offset
        ):
            raise VerifyException(
                f"Expected stencil.access offset mapping be of length {len(self.offset)} "
                f"to match the provided offsets, but it is {len(self.offset_mapping)} "
                f"instead"
            )

        if self.offset_mapping is not None:
            prev_offset = None
            for prev_offset, offset in pairwise(self.offset_mapping):
                if prev_offset >= offset:
                    raise VerifyException(
                        "Offset mapping in stencil.access must be strictly increasing."
                        "increasing"
                    )
            for offset in self.offset_mapping:
                if offset >= apply.get_rank():
                    raise VerifyException(
                        f"Offset mappings in stencil.access must be within the rank of the "
                        f"apply, got {offset} >= {apply.get_rank()}"
                    )

    def get_apply(self):
        """
        Simple helper to get the parent apply and raise otherwise.
        """
        trait = cast(HasAncestor, self.get_trait(HasAncestor, (stencil.ApplyOp,)))
        ancestor = trait.get_ancestor(self)
        if ancestor is None:
            raise ValueError(
                "stencil.apply not found, this function should be called on"
                "verified accesses only."
            )
        return cast(stencil.ApplyOp, ancestor)


CSL_STENCIL = Dialect(
    "csl_stencil",
    [
        PrefetchOp,
        AccessOp,
    ],
    [
        ExchangeDeclarationAttr,
    ],
)
