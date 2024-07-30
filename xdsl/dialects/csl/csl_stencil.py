from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import cast

from xdsl.dialects import builtin, memref, stencil
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    AnyMemRefType,
    IndexType,
    TensorType,
)
from xdsl.dialects.experimental import dmp
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
)
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    ParameterDef,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import (
    HasAncestor,
    HasCanonicalisationPatternsTrait,
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    Pure,
    RecursiveMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa
from xdsl.utils.isattr import isattr


@irdl_attr_definition
class ExchangeDeclarationAttr(ParametrizedAttribute):
    """
    A simplified version of dmp.exchange, from which it should be lowered.

    `neighbor` is a list (e.g. [x, y]) where
      - the index encodes dimension,
      - the sign encodes direction,
      - the magnitude encodes distance
    As such, the values correspond to those used by both stencil.access and dmp.exchange

    This works irrespective of whether the accesses are diagonal or not.
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
    An op to indicate a symmetric (send and receive) buffer prefetch across the stencil shape.

    This should be irrespective of the stencil shape (and whether it does or does not include diagonals).

    Returns memref<${len(self.swaps}xtensor<{buffer size}x{data type}>>
    """

    name = "csl_stencil.prefetch"

    input_stencil = operand_def(
        base(stencil.TempType[Attribute]) | base(memref.MemRefType[Attribute])
    )

    swaps = prop_def(builtin.ArrayAttr[ExchangeDeclarationAttr])

    topo = prop_def(dmp.RankTopoAttr)

    result = result_def(memref.MemRefType)

    def __init__(
        self,
        input_stencil: SSAValue | Operation,
        topo: dmp.RankTopoAttr,
        swaps: Sequence[ExchangeDeclarationAttr],
        result_type: memref.MemRefType[Attribute] | TensorType[Attribute] | None = None,
    ):
        super().__init__(
            operands=[input_stencil],
            properties={
                "topo": topo,
                "swaps": builtin.ArrayAttr(swaps),
            },
            result_types=[result_type],
        )


class ApplyOpHasCanonicalizationPatternsTrait(HasCanonicalisationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.csl_stencil import (
            RedundantIterArgInitialisation,
        )

        return (RedundantIterArgInitialisation(),)


@irdl_op_definition
class ApplyOp(IRDLOperation):
    """
    This operation combines a `csl_stencil.prefetch` (symmetric buffer communication across a given stencil shape)
    with a `stencil.apply` (a stencil function plus parameters and applies the stencil function to the output temp).

    As communication may be done in chunks, this operation provides two regions for computation:
      - the `chunk_reduce` region to reduce a chunk of data received from several neighbours to one chunk of data.
        this region is invoked once per communicated chunks and effectively acts as a loop body.
        It uses `iter_arg` to concatenate the chunks
      - the `post_process` region (invoked once when communication has finished) that takes the concatenated
        chunk of the `chunk_reduce` region and applies any further processing here - for instance, it may handle
        the computation of 'own' (non-communicated) or otherwise prefetched data

    Further fields:
      - `communicated_stencil` - the stencil to communicate (send and receive)
      - `args`       - arguments to the stencil computation, may include other prefetched buffers
      - `topo`       - as received from `csl_stencil.prefetch`/`dmp.swap`
      - `num_chunks` - number of chunks into which to slice the communication
      - `swaps`      - a set of neighbouring points in the stencil, whose value we wish to retain
                       (note, these are not guaranteed to be lowered as true point-to-point communication, and
                       redundant communication should be irgnored)

    Function signatures:
    Before lowering (from `csl_stencil.prefetch` and `stencil.apply`):
        %pref = csl_stencil.prefetch(%communicated_stencil : stencil.Temp)
        stencil.apply( ..some args.. , %communicated_stencil, ..some more args.., %pref)

    After lowering:
        op:             csl_stencil.apply(%communicated_stencil, %iter_arg, chunk_reduce_args..., post_process_args...)
        chunk_reduce:   block_args(slice of type(%pref), %offset, %iter_arg, args...)
        post_process:   block_args(%communicated_stencil, %iter_arg, args...)

    Note, that %pref can be dropped (as communication is done by the op rather than before the op),
    and that a new %iter_arg is required, an empty tensor which is filled by `chunk_reduce` and
    consumed by `post_process`
    """

    name = "csl_stencil.apply"

    communicated_stencil = operand_def(
        base(stencil.TempType[Attribute]) | base(memref.MemRefType[Attribute])
    )

    iter_arg = operand_def(TensorType[Attribute])

    args = var_operand_def(Attribute)

    chunk_reduce = region_def()
    post_process = region_def()

    swaps = prop_def(builtin.ArrayAttr[ExchangeDeclarationAttr])

    topo = prop_def(dmp.RankTopoAttr)

    num_chunks = prop_def(AnyIntegerAttr)

    res = var_result_def(stencil.TempType)

    traits = frozenset(
        [
            IsolatedFromAbove(),
            ApplyOpHasCanonicalizationPatternsTrait(),
            RecursiveMemoryEffect(),
        ]
    )

    def print(self, printer: Printer):
        def print_arg(arg: tuple[SSAValue, Attribute]):
            printer.print(arg[0])
            printer.print(" : ")
            printer.print(arg[1])

        printer.print("(")

        # args required by function signature, plus optional args for regions
        args = [self.communicated_stencil, self.iter_arg, *self.args]

        printer.print_list(
            zip(args, (a.type for a in args)),
            print_arg,
        )
        printer.print(") <")
        printer.print_attr_dict(self.properties)
        printer.print("> -> (")
        printer.print_list((r.type for r in self.res), printer.print_attribute)
        printer.print(") ")
        printer.print_op_attributes(self.attributes, print_keyword=True)
        printer.print("(")
        printer.print_region(self.chunk_reduce, print_entry_block_args=True)
        printer.print(", ")
        printer.print_region(self.post_process, print_entry_block_args=True)
        printer.print(")")

    @classmethod
    def parse(cls, parser: Parser):
        def parse_args():
            value = parser.parse_unresolved_operand()
            parser.parse_punctuation(":")
            type = parser.parse_attribute()
            value = parser.resolve_operand(value, type)
            return value

        operands = parser.parse_comma_separated_list(parser.Delimiter.PAREN, parse_args)

        props = parser.parse_optional_properties_dict()

        parser.parse_punctuation("->")
        result_types = parser.parse_comma_separated_list(
            parser.Delimiter.PAREN, parser.parse_attribute
        )
        attrs = parser.parse_optional_attr_dict_with_keyword()
        if attrs is not None:
            attrs = attrs.data
        parser.parse_punctuation("(")
        chunk_reduce = parser.parse_region()
        parser.parse_punctuation(",")
        post_process = parser.parse_region()
        parser.parse_punctuation(")")
        return cls(
            operands=[operands[0], operands[1], operands[2:]],
            result_types=[result_types],
            regions=[chunk_reduce, post_process],
            properties=props,
            attributes=attrs,
        )

    def verify_(self) -> None:
        # typecheck op arguments
        if (
            len(self.chunk_reduce.block.args) < 3
            or len(self.post_process.block.args) < 2
        ):
            raise VerifyException("Missing required block args on region")
        op_args = (
            self.post_process.block.args[0],
            self.chunk_reduce.block.args[2],
            *self.chunk_reduce.block.args[3:],
            *self.post_process.block.args[2:],
        )
        for operand, argument in zip(self.operands, op_args):
            if operand.type != argument.type:
                raise VerifyException(
                    f"Expected argument type of {type(self)} to match operand type, got {argument.type} != {operand.type} at index {argument.index}"
                )

        # typecheck required (only) block arguments
        assert isa(self.iter_arg.type, TensorType[Attribute])
        chunk_reduce_req_types = [
            TensorType(
                self.iter_arg.type.get_element_type(),
                (
                    len(self.swaps),
                    self.iter_arg.type.get_shape()[0] // self.num_chunks.value.data,
                ),
            ),
            IndexType(),
            self.iter_arg.type,
        ]
        post_process_req_types = [
            self.communicated_stencil.type,
            self.iter_arg.type,
        ]
        for arg, expected_type in zip(
            self.chunk_reduce.block.args, chunk_reduce_req_types
        ):
            if arg.type != expected_type:
                raise VerifyException(
                    f"Unexpected block argument type of chunk_reduce, got {arg.type} != {expected_type} at index {arg.index}"
                )
        for arg, expected_type in zip(
            self.post_process.block.args, post_process_req_types
        ):
            if arg.type != expected_type:
                raise VerifyException(
                    f"Unexpected block argument type of post_process, got {arg.type} != {expected_type} at index {arg.index}"
                )

        if len(self.res) < 1:
            raise VerifyException(
                f"Expected stencil.apply to have at least 1 result, got {len(self.res)}"
            )
        res_type = cast(stencil.TempType[Attribute], self.res[0].type)
        for other in self.res[1:]:
            other = cast(stencil.TempType[Attribute], other.type)
            if res_type.bounds != other.bounds:
                raise VerifyException("Expected all output types bounds to be equals.")

    def get_rank(self) -> int:
        res_type = self.res[0].type
        assert isa(res_type, stencil.TempType[Attribute])
        return res_type.get_num_dims()

    def get_accesses(self) -> Iterable[stencil.AccessPattern]:
        """
        Return the access patterns of each input.

         - An offset is a tuple describing a relative access
         - An access pattern is a class wrapping a sequence of offsets
         - This method returns an access pattern for each stencil
           field of the apply operation.
        """
        # iterate over the block arguments
        for arg in self.chunk_reduce.block.args + self.post_process.block.args:
            accesses: list[tuple[int, ...]] = []
            # walk the uses of the argument
            for use in arg.uses:
                # filter out all non access ops
                if not isinstance(use.operation, AccessOp):
                    continue
                access: AccessOp = use.operation
                # grab the offsets as a tuple[int, ...]
                offsets = tuple(access.offset)
                # account for offset_mappings:
                if access.offset_mapping is not None:
                    offsets = tuple(offsets[i] for i in access.offset_mapping)
                accesses.append(offsets)
            yield stencil.AccessPattern(tuple(accesses))


@irdl_op_definition
class AccessOp(IRDLOperation):
    """
    A CSL stencil access that operates on own data or data prefetched from neighbors via `csl_stencil.prefetch`

    The source of data determines the type `op` is required to have:

      ${type(op) == memref.MemRefType}  -  for accesses to data prefetched from neighbors
      ${type(op) == stencil.Temp}       -  for accesses to own data

    """

    name = "csl_stencil.access"
    op = operand_def(
        base(AnyMemRefType) | base(stencil.AnyTempType) | base(TensorType[Attribute])
    )
    offset = prop_def(stencil.IndexAttr)
    offset_mapping = opt_prop_def(stencil.IndexAttr)
    result = result_def(TensorType)

    traits = frozenset([HasAncestor(stencil.ApplyOp, ApplyOp), Pure()])

    def __init__(
        self,
        op: Operand,
        offset: stencil.IndexAttr,
        result_type: TensorType[Attribute],
        offset_mapping: stencil.IndexAttr | None = None,
    ):
        super().__init__(
            operands=[op],
            properties={"offset": offset, "offset_mapping": offset_mapping},
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

        props = parser.parse_optional_attr_dict_with_keyword(
            {"offset", "offset_mapping"}
        )
        props = props.data if props else dict[str, Attribute]()
        props["offset"] = stencil.IndexAttr.get(*offset)
        if offset_mapping:
            props["offset_mapping"] = stencil.IndexAttr.get(*offset_mapping)
        parser.parse_punctuation(":")
        res_type = parser.parse_attribute()
        if isattr(res_type, base(AnyMemRefType)):
            return cls.build(
                operands=[temp], result_types=[res_type.element_type], properties=props
            )
        elif isattr(res_type, base(TensorType[Attribute])):
            return cls.build(
                operands=[temp],
                result_types=[
                    TensorType(res_type.element_type, res_type.get_shape()[1:])
                ],
                properties=props,
            )
        elif isattr(res_type, base(AnyMemRefType)):
            return cls.build(
                operands=[temp],
                result_types=[
                    memref.MemRefType(res_type.element_type, res_type.get_shape()[1:])
                ],
                properties=props,
            )
        parser.raise_error(
            "Expected return type to be a tensor, memref, or stencil.temp"
        )

    def verify_(self) -> None:
        if tuple(self.offset) == (0, 0):
            if not isa(self.op.type, stencil.TempType[Attribute]):
                raise VerifyException(
                    f"{type(self)} access to own data requires type stencil.TempType but found {self.op.type}"
                )
            assert self.result.type == self.op.type.get_element_type()
        else:
            if not isa(
                self.op.type, TensorType[Attribute] | memref.MemRefType[Attribute]
            ):
                raise VerifyException(
                    f"{type(self)} access to neighbor data requires type memref.MemRefType or TensorType but found {self.op.type}"
                )

        # As promised by HasAncestor(ApplyOp)
        trait = cast(
            HasAncestor, AccessOp.get_trait(HasAncestor(stencil.ApplyOp, ApplyOp))
        )
        apply = trait.get_ancestor(self)
        assert isinstance(apply, stencil.ApplyOp | ApplyOp)

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

    def get_apply(self) -> stencil.ApplyOp | ApplyOp:
        """
        Simple helper to get the parent apply and raise otherwise.
        """
        trait = cast(
            HasAncestor,
            self.get_trait(HasAncestor(stencil.ApplyOp, ApplyOp)),
        )
        ancestor = trait.get_ancestor(self)
        if ancestor is None:
            raise ValueError(
                "stencil.apply not found, this function should be called on"
                "verified accesses only."
            )
        assert isinstance(ancestor, stencil.ApplyOp | ApplyOp)
        return ancestor


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "csl_stencil.yield"

    traits = traits_def(lambda: frozenset([IsTerminator(), HasParent(ApplyOp), Pure()]))


CSL_STENCIL = Dialect(
    "csl_stencil",
    [
        PrefetchOp,
        AccessOp,
        ApplyOp,
        YieldOp,
    ],
    [
        ExchangeDeclarationAttr,
    ],
)
