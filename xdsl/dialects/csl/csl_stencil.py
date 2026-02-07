from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import TypeAlias, cast

from xdsl.dialects import builtin, memref, stencil
from xdsl.dialects.builtin import (
    I64,
    AnyTensorTypeConstr,
    DenseArrayBase,
    Float16Type,
    Float32Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    TensorType,
    i64,
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
    AttrSizedOperandSegments,
    IRDLOperation,
    Operand,
    irdl_attr_definition,
    irdl_op_definition,
    lazy_traits_def,
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
    HasCanonicalizationPatternsTrait,
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    MemoryReadEffect,
    MemoryWriteEffect,
    Pure,
    RecursiveMemoryEffect,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


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

    neighbor_param: DenseArrayBase[I64]

    def __init__(
        self,
        neighbor: Sequence[int] | DenseArrayBase,
    ):
        data_type = builtin.i64
        super().__init__(
            neighbor
            if isinstance(neighbor, DenseArrayBase)
            else DenseArrayBase.from_list(data_type, neighbor)
        )

    @classmethod
    def from_dmp_exch_decl_attr(cls, src: dmp.ExchangeDeclarationAttr):
        return cls(src.neighbor)

    @property
    def neighbor(self) -> tuple[int, ...]:
        return self.neighbor_param.get_values()

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

        return [DenseArrayBase.from_list(i64, to)]


@irdl_op_definition
class PrefetchOp(IRDLOperation):
    """
    An op to indicate a symmetric (send and receive) buffer prefetch across the stencil shape.

    This should be irrespective of the stencil shape (and whether it does or does not include diagonals).

    Returns memref<${len(self.swaps}xtensor<{buffer size}x{data type}>>
    """

    name = "csl_stencil.prefetch"

    input_stencil = operand_def(
        stencil.StencilTypeConstr | MemRefType.constr() | AnyTensorTypeConstr
    )

    swaps = prop_def(builtin.ArrayAttr[ExchangeDeclarationAttr])

    topo = prop_def(dmp.RankTopoAttr)

    num_chunks = prop_def(IntegerAttr)

    result = result_def(MemRefType.constr() | AnyTensorTypeConstr)

    def __init__(
        self,
        input_stencil: SSAValue | Operation,
        topo: dmp.RankTopoAttr,
        num_chunks: IntegerAttr,
        swaps: Sequence[ExchangeDeclarationAttr],
        result_type: memref.MemRefType | TensorType[Attribute] | None = None,
    ):
        super().__init__(
            operands=[input_stencil],
            properties={
                "topo": topo,
                "swaps": builtin.ArrayAttr(swaps),
                "num_chunks": num_chunks,
            },
            result_types=[result_type],
        )


CslFloat: TypeAlias = Float16Type | Float32Type


@irdl_attr_definition
class CoeffAttr(ParametrizedAttribute):
    name = "csl_stencil.coeff"
    offset: stencil.IndexAttr
    coeff: FloatAttr


class ApplyOpHasCanonicalizationPatternsTrait(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl.transforms.canonicalization_patterns.csl_stencil import (
            RedundantAccumulatorInitialisation,
        )

        return (RedundantAccumulatorInitialisation(),)


@irdl_op_definition
class ApplyOp(IRDLOperation):
    """
    This operation combines a `csl_stencil.prefetch` (symmetric buffer communication across a given stencil shape)
    with a `stencil.apply` (a stencil function plus parameters and applies the stencil function to the output temp).

    As communication may be done in chunks, this operation provides two regions for computation:
      - the `receive_chunk` region to reduce a chunk of data received from several neighbours to one chunk of data.
        this region is invoked once per communicated chunks and effectively acts as a loop body.
        It uses `accumulator` to concatenate the chunks
      - the `done_exchange` region (invoked once when communication has finished) that takes the concatenated
        chunk of the `receive_chunk` region and applies any further processing here - for instance, it may handle
        the computation of 'own' (non-communicated) or otherwise prefetched data

    Further fields:
      - `field`      - the stencil field to communicate (send and receive)
      - `args_rchunk`  - arguments passed to the `receive_chunk` region, may include other prefetched buffers
      - `args_dexchng` - arguments passed to the `done_exchange` region, may include other prefetched buffers
      - `args`       - arguments to the stencil computation, may include other prefetched buffers
      - `topo`       - as received from `csl_stencil.prefetch`/`dmp.swap`
      - `num_chunks` - number of chunks into which to slice the communication
      - `swaps`      - a set of neighbouring points in the stencil, whose value we wish to retain
                       (note, these are not guaranteed to be lowered as true point-to-point communication, and
                       redundant communication should be irgnored)

    Function signatures:
    Before lowering (from `csl_stencil.prefetch` and `stencil.apply`):
        %pref = csl_stencil.prefetch(%field : stencil.Temp)
        stencil.apply( ..some args.. , %field, ..some more args.., %pref)

    After lowering:
        op:             csl_stencil.apply(%field, %accumulator, receive_chunk_args..., done_exchange_args...)
        receive_chunk:   block_args(slice of type(%pref), %offset, %accumulator, args...)
        done_exchange:   block_args(%field, %accumulator, args...)

    Note, that %pref can be dropped (as communication is done by the op rather than before the op),
    and that a new %accumulator is required, an empty tensor which is filled by `receive_chunk` and
    consumed by `done_exchange`
    """

    name = "csl_stencil.apply"

    field = operand_def(stencil.StencilTypeConstr | MemRefType.constr())

    accumulator = operand_def(TensorType | MemRefType)

    args_rchunk = var_operand_def(Attribute)
    args_dexchng = var_operand_def(Attribute)
    dest = var_operand_def(stencil.FieldTypeConstr | MemRefType.constr())

    receive_chunk = region_def()
    done_exchange = region_def()

    swaps = prop_def(builtin.ArrayAttr[ExchangeDeclarationAttr])

    topo = prop_def(dmp.RankTopoAttr)

    num_chunks = prop_def(IntegerAttr)

    bounds = opt_prop_def(stencil.StencilBoundsAttr)

    coeffs = opt_prop_def(builtin.ArrayAttr[CoeffAttr])

    res = var_result_def(stencil.StencilTypeConstr)

    traits = traits_def(
        IsolatedFromAbove(),
        ApplyOpHasCanonicalizationPatternsTrait(),
        MemoryReadEffect(),
        MemoryWriteEffect(),
        RecursiveMemoryEffect(),
    )

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    def print(self, printer: Printer):
        def print_arg(arg: SSAValue):
            printer.print_ssa_value(arg)
            printer.print_string(" : ")
            printer.print_attribute(arg.type)

        with printer.in_parens():
            # args required by function signature, plus optional args for regions
            args = [self.field, self.accumulator, *self.args_rchunk, *self.args_dexchng]

            printer.print_list(args, print_arg)
        if self.dest:
            printer.print_string(" outs ")
            with printer.in_parens():
                printer.print_list(self.dest, print_arg)
        else:
            printer.print_string(" -> ")
            with printer.in_parens():
                printer.print_list(self.res.types, printer.print_attribute)

        printer.print_string(" ")
        with printer.in_angle_brackets():
            printer.print_attr_dict(self.properties)
        printer.print_string(" ")
        printer.print_op_attributes(self.attributes, print_keyword=True)
        with printer.in_parens():
            printer.print_region(self.receive_chunk, print_entry_block_args=True)
            printer.print_string(", ")
            printer.print_region(self.done_exchange, print_entry_block_args=True)
        if self.bounds is not None:
            printer.print_string(" to ")
            self.bounds.print_parameters(printer)

    @classmethod
    def parse(cls, parser: Parser):
        def parse_args():
            value = parser.parse_unresolved_operand()
            parser.parse_punctuation(":")
            type = parser.parse_attribute()
            value = parser.resolve_operand(value, type)
            return value

        ops = parser.parse_comma_separated_list(parser.Delimiter.PAREN, parse_args)

        if parser.parse_optional_punctuation("->"):
            parser.parse_punctuation("(")
            result_types = parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_attribute, parser.parse_attribute
            )
            destinations = []
        else:
            parser.parse_keyword("outs")
            parser.parse_punctuation("(")
            destinations = parser.parse_comma_separated_list(
                parser.Delimiter.NONE, parse_args
            )
            result_types = []
        parser.parse_punctuation(")")

        props = parser.parse_optional_properties_dict()
        attrs = parser.parse_optional_attr_dict_with_keyword()
        if attrs is not None:
            attrs = attrs.data
        parser.parse_punctuation("(")
        receive_chunk = parser.parse_region()
        parser.parse_punctuation(",")
        done_exchange = parser.parse_region()
        parser.parse_punctuation(")")
        if parser.parse_optional_keyword("to"):
            props["bounds"] = stencil.StencilBoundsAttr.new(
                stencil.StencilBoundsAttr.parse_parameters(parser)
            )
        # `-3` fixed block args, `+2` offset for operands with fixed use
        split = len(receive_chunk.block.args) - 3 + 2
        return cls(
            operands=[ops[0], ops[1], ops[2:split], ops[split:], destinations],
            result_types=[result_types],
            regions=[receive_chunk, done_exchange],
            properties=props,
            attributes=attrs,
        )

    def verify_(self) -> None:
        # typecheck op arguments
        if (
            len(self.receive_chunk.block.args) < 3
            or len(self.done_exchange.block.args) < 2
        ):
            raise VerifyException("Missing required block args on region")
        op_args = (
            self.done_exchange.block.args[0],
            self.receive_chunk.block.args[2],
            *self.receive_chunk.block.args[3:],
            *self.done_exchange.block.args[2:],
        )
        for operand, argument in zip(self.operands, op_args):
            if operand.type != argument.type:
                raise VerifyException(
                    f"Expected argument type of {type(self)} to match operand type, "
                    f"got {argument.type} != {operand.type} at index {argument.index}"
                )

        # typecheck required (only) block arguments
        assert isa(self.accumulator.type, TensorType | MemRefType)
        chunk_region_req_types = [
            type(self.accumulator.type)(
                self.accumulator.type.get_element_type(),
                (
                    len(self.swaps),
                    self.accumulator.type.get_shape()[-1] // self.num_chunks.value.data,
                ),
            ),
            IndexType(),
            self.accumulator.type,
        ]
        done_exchange_req_types = [
            self.field.type,
            self.accumulator.type,
        ]
        for arg, expected_type in zip(
            self.receive_chunk.block.args, chunk_region_req_types
        ):
            if arg.type != expected_type:
                raise VerifyException(
                    f"Unexpected block argument type of receive_chunk, got {arg.type} != {expected_type} at index {arg.index}"
                )
        for arg, expected_type in zip(
            self.done_exchange.block.args, done_exchange_req_types
        ):
            if arg.type != expected_type:
                raise VerifyException(
                    f"Unexpected block argument type of done_exchange, got {arg.type} != {expected_type} at index {arg.index}"
                )

        if (len(self.res) > 0) and (len(self.dest) > 0):
            raise VerifyException(
                "Cannot specify both results and dest on stencil.apply"
            )

    def get_rank(self) -> int:
        if self.dest:
            res_type = self.dest[0].type
        elif self.res:
            res_type = self.res[0].type
        else:
            return 2
        if isinstance(res_type, stencil.StencilType):
            return res_type.get_num_dims()
        elif self.bounds:
            return len(self.bounds.ub)
        raise ValueError("Cannot derive rank")

    def get_accesses(self) -> Iterable[stencil.AccessPattern]:
        """
        Return the access patterns of each input.

         - An offset is a tuple describing a relative access
         - An access pattern is a class wrapping a sequence of offsets
         - This method returns an access pattern for each stencil
           field of the apply operation.
        """
        # iterate over the block arguments
        for arg in self.receive_chunk.block.args + self.done_exchange.block.args:
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

    def add_coeff(self, offset: stencil.IndexAttr, coeff: FloatAttr):
        self.coeffs = builtin.ArrayAttr(
            list(self.coeffs or []) + [CoeffAttr(offset, coeff)]
        )


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
        MemRefType.constr() | stencil.StencilTypeConstr | AnyTensorTypeConstr
    )
    offset = prop_def(stencil.IndexAttr)
    offset_mapping = opt_prop_def(stencil.IndexAttr)
    result = result_def(TensorType | MemRefType)

    traits = traits_def(HasAncestor(stencil.ApplyOp, ApplyOp), Pure())

    def __init__(
        self,
        op: Operand,
        offset: stencil.IndexAttr,
        result_type: TensorType[Attribute] | MemRefType,
        offset_mapping: stencil.IndexAttr | None = None,
    ):
        super().__init__(
            operands=[op],
            properties={"offset": offset, "offset_mapping": offset_mapping},
            result_types=[result_type],
        )

    def print(self, printer: Printer):
        printer.print_string(" ")
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

        with printer.in_square_brackets():
            index = 0
            for i in range(len(self.offset)):
                if i in mapping:
                    printer.print_int(offset[index])
                    index += 1
                else:
                    printer.print_string("_")
                if i != len(self.offset) - 1:
                    printer.print_string(", ")

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
        props = dict(props.data) if props else {}
        props["offset"] = stencil.IndexAttr.get(*offset)
        if offset_mapping:
            props["offset_mapping"] = stencil.IndexAttr.get(*offset_mapping)
        parser.parse_punctuation(":")
        res_type = parser.parse_attribute()
        if stencil.StencilTypeConstr.verifies(res_type):
            return cls.build(
                operands=[temp],
                result_types=[res_type.get_element_type()],
                properties=props,
            )
        elif isa(res_type, TensorType):
            return cls.build(
                operands=[temp],
                result_types=[
                    TensorType(res_type.element_type, res_type.get_shape()[-1:])
                ],
                properties=props,
            )
        elif MemRefType.constr().verifies(res_type):
            return cls.build(
                operands=[temp],
                result_types=[
                    memref.MemRefType(res_type.element_type, res_type.get_shape()[-1:])
                ],
                properties=props,
            )
        parser.raise_error(
            "Expected return type to be a tensor, memref, or stencil.temp"
        )

    def verify_(self) -> None:
        if tuple(self.offset) == (0, 0):
            if isa(self.op.type, memref.MemRefType):
                if not self.result.type == self.op.type:
                    raise VerifyException(
                        f"{type(self)} access to own data requires{self.op.type} but "
                        f"found {self.result.type}"
                    )
            elif stencil.StencilTypeConstr.verifies(self.op.type):
                if not self.result.type == self.op.type.get_element_type():
                    raise VerifyException(
                        f"{type(self)} access to own data requires "
                        f"{self.op.type.get_element_type()} but found "
                        f"{self.result.type}"
                    )
            else:
                raise VerifyException(
                    f"{type(self)} access to own data requires type "
                    f"stencil.StencilType or memref.MemRefType but found {self.op.type}"
                )
        else:
            if not isa(self.op.type, TensorType | MemRefType):
                raise VerifyException(
                    f"{type(self)} access to neighbor data requires type "
                    f"memref.MemRefType or TensorType but found {self.op.type}"
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

    traits = lazy_traits_def(lambda: (IsTerminator(), HasParent(ApplyOp), Pure()))


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
        CoeffAttr,
    ],
)
