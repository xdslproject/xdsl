import abc
import typing
from collections.abc import Callable
from typing import overload

from xdsl.dialects import arith, builtin, llvm, printf, scf
from xdsl.dialects.builtin import (
    AnyFloat,
    DenseArrayBase,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    i64,
)
from xdsl.dialects.experimental import dlt
from xdsl.dialects.experimental.dlt import SetAttr
from xdsl.ir import Block, Operation, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.transforms.experimental.dlt.layout_manipulation import Manipulator


class NumericResult:
    def __init__(
        self,
        operations: typing.Iterable[Operation],
        static_values: tuple[int, ...],
        ssa_values: tuple[typing.Union[None | SSAValue], ...],
    ):
        self.operations: tuple[Operation, ...] = tuple(operations)
        assert len(static_values) == len(ssa_values)
        self.static_values = static_values
        self.ssa_values = ssa_values
        self.used = False

    @staticmethod
    @typing.overload
    def from_mixed(
        operations: typing.Iterable[Operation], value1: typing.Union[int, SSAValue]
    ) -> typing.Annotated["NumericResult", 1]: ...

    @staticmethod
    @typing.overload
    def from_mixed(
        operations: typing.Iterable[Operation],
        value1: typing.Union[int, SSAValue],
        value2: typing.Union[int, SSAValue],
    ) -> typing.Annotated["NumericResult", 2]: ...

    @staticmethod
    @typing.overload
    def from_mixed(
        operations: typing.Iterable[Operation],
        value1: typing.Union[int, SSAValue],
        value2: typing.Union[int, SSAValue],
        value3: typing.Union[int, SSAValue],
    ) -> typing.Annotated["NumericResult", 3]: ...

    @staticmethod
    @typing.overload
    def from_mixed(
        operations: typing.Iterable[Operation],
        value1: typing.Union[int, SSAValue],
        value2: typing.Union[int, SSAValue],
        value3: typing.Union[int, SSAValue],
        value4: typing.Union[int, SSAValue],
    ) -> typing.Annotated["NumericResult", 4]: ...

    @staticmethod
    def from_mixed(
        operations: typing.Iterable[Operation],
        *values: typing.Union[int, SSAValue],
    ) -> "NumericResult":
        for v in values:
            assert isinstance(v, int | SSAValue)
        static_values = tuple([v if isinstance(v, int) else 0 for v in values])
        ssa_values = tuple([v if isinstance(v, SSAValue) else None for v in values])
        result = typing.cast(
            typing.Annotated[NumericResult, len(values)],
            NumericResult(operations, static_values, ssa_values),
        )
        return result

    @staticmethod
    def from_ssa(*values: SSAValue) -> "NumericResult":
        return NumericResult.from_mixed([], *values)

    @staticmethod
    def from_const(*values: int) -> "NumericResult":
        return NumericResult.from_mixed([], *values)

    def _check_bin_op_compatibility(self, other):
        if len(self.static_values) < len(other.static_values):
            raise ValueError()
        if self.used | other.used:
            raise ValueError()

    def __add__(
        self, other: typing.Self | int | SSAValue | tuple[int | SSAValue, ...]
    ) -> typing.Self:
        if isinstance(other, int | SSAValue):
            other = NumericResult.from_mixed([], other)
        elif isinstance(other, tuple) and all(
            isinstance(v, int | SSAValue) for v in other
        ):
            other = NumericResult.from_mixed([], *other)

        if not isinstance(other, NumericResult):
            return NotImplemented
        self._check_bin_op_compatibility(other)

        all_ops = self.operations + other.operations
        new_static = []
        for i, a in enumerate(self.static_values):
            if i < len(other.static_values):
                b = other.static_values[i]
            else:
                b = 0
            new_static.append(a + b)

        new_ssa = []
        for i, a in enumerate(self.ssa_values):
            if i < len(other.ssa_values):
                b = other.ssa_values[i]
                if a is None:
                    new_ssa.append(b)
                elif b is None:
                    new_ssa.append(a)
                else:
                    all_ops += (sum_op := arith.Addi(a, b),)
                    new_ssa.append(sum_op.result)
            else:
                new_ssa.append(a)
        self.used = True
        other.used = True
        return NumericResult(all_ops, tuple(new_static), tuple(new_ssa))

    def __sub__(
        self, other: typing.Self | int | SSAValue | tuple[int | SSAValue, ...]
    ) -> typing.Self:
        if isinstance(other, int | SSAValue):
            other = NumericResult.from_mixed([], other)
        elif isinstance(other, tuple) and all(
            isinstance(v, int | SSAValue) for v in other
        ):
            other = NumericResult.from_mixed([], *other)

        if not isinstance(other, NumericResult):
            return NotImplemented
        self._check_bin_op_compatibility(other)

        if any(a is None for a, b in zip(self.ssa_values, other.ssa_values)):
            new_self = self.to_ssa()
            return new_self - other

        all_ops = self.operations + other.operations
        new_static = []
        for i, a in enumerate(self.static_values):
            if i < len(other.static_values):
                b = other.static_values[i]
            else:
                b = 0
            new_static.append(a - b)

        new_ssa = []
        for i, a in enumerate(self.ssa_values):
            if i < len(other.ssa_values):
                b = other.ssa_values[i]
                if a is None:
                    assert False
                elif b is None:
                    new_ssa.append(a)
                else:
                    all_ops += (sub_op := arith.Subi(a, b),)
                    new_ssa.append(sub_op.result)
            else:
                new_ssa.append(a)
        self.used = True
        other.used = True
        return NumericResult(all_ops, tuple(new_static), tuple(new_ssa))

    def __mul__(
        self, other: typing.Self | int | SSAValue | tuple[int | SSAValue, ...]
    ) -> typing.Self:
        if isinstance(other, int | SSAValue):
            other = NumericResult.from_mixed([], other)
        elif isinstance(other, tuple) and all(
            isinstance(v, int | SSAValue) for v in other
        ):
            other = NumericResult.from_mixed([], *other)

        if not isinstance(other, NumericResult):
            return NotImplemented
        self._check_bin_op_compatibility(other)

        all_ops = self.operations + other.operations
        new_static = []
        new_ssa = []
        for i, (a_s, a_d) in enumerate(zip(self.static_values, self.ssa_values)):
            if i >= len(other.ssa_values):
                new_static.append(a_s)
                new_ssa.append(a_d)
            elif other.ssa_values[i] is None and other.static_values[i] == 1:
                new_static.append(a_s)
                new_ssa.append(a_d)
            elif other.ssa_values[i] is None and other.static_values[i] == 0:
                new_static.append(0)
                new_ssa.append(None)
            elif a_d is None and a_s == 1:
                new_static.append(other.static_values[i])
                new_ssa.append(other.ssa_values[i])
            elif a_d is None and a_s == 0:
                new_static.append(0)
                new_ssa.append(None)
            elif a_d is None and other.ssa_values[i] is None:
                new_static.append(a_s * other.static_values[i])
                new_ssa.append(None)
            else:
                a_ops, a_ssa = NumericResult._ssa_of(a_s, a_d)
                b_ops, b_ssa = NumericResult._ssa_of(
                    other.static_values[i], other.ssa_values[i]
                )
                all_ops += a_ops
                all_ops += b_ops
                all_ops += (product := arith.Muli(a_ssa, b_ssa),)
                new_static.append(0)
                new_ssa.append(product.result)

        self.used = True
        other.used = True
        return NumericResult(all_ops, tuple(new_static), tuple(new_ssa))

    def keep(self, n: int | typing.Iterable[bool]) -> "NumericResult":
        if self.used:
            raise ValueError()
        if isinstance(n, int):
            self.used = True
            return typing.cast(
                typing.Annotated[NumericResult, n],
                NumericResult(
                    self.operations, self.static_values[0:n], self.ssa_values[0:n]
                ),
            )
        else:
            ops = self.operations
            new_static = []
            new_ssa = []
            for i, b in enumerate(n):
                new_static.append(self.static_values[i])
                new_ssa.append(self.ssa_values[i])
            self.used = True
            return NumericResult(ops, tuple(new_static), tuple(new_ssa))

    def extend(self, val: int | SSAValue):
        if self.used:
            raise ValueError()
        self.used = True
        if isinstance(val, int):
            return NumericResult(
                self.operations, self.static_values + (val,), self.ssa_values + (None,)
            )
        elif isinstance(val, SSAValue):
            return NumericResult(
                self.operations, self.static_values + (0,), self.ssa_values + (val,)
            )
        else:
            raise ValueError()

    def sum(self) -> typing.Annotated["NumericResult", 1]:
        if self.used:
            raise ValueError()
        if self.size == 1:
            return self
        if self.size == 0:
            self.used = True
            return NumericResult.from_mixed(self.operations, 0)
        new_static = 0
        ops = list(self.operations)
        for s in self.static_values:
            new_static += s

        parts = tuple([s for s in self.ssa_values if s is not None])
        while len(parts) > 1:
            new_parts = []
            for i in range(0, len(parts) - 1, 2):
                ops.append(add_op := arith.Addi(parts[i], parts[i + 1]))
                new_parts.append(add_op.result)
            parts = tuple(new_parts)
        if len(parts) == 0:
            parts = (None,)

        self.used = True
        return NumericResult(ops, (new_static,), parts)

    @staticmethod
    def _ssa_of(
        static: int, ssa: SSAValue | None
    ) -> tuple[tuple[Operation, ...], SSAValue]:
        if ssa is None:
            return (c := arith.Constant(IntegerAttr(static, IndexType())),), c.result
        elif static == 0:
            return tuple(), ssa
        else:
            ops = (
                c := arith.Constant(IntegerAttr(static, IndexType())),
                s := arith.Addi(c.result, ssa),
            )
            return ops, s.result

    def to_ssa(self) -> typing.Self:
        if self.used:
            raise ValueError()
        all_ops = list(self.operations)
        new_ssa = []
        for i, (a, b) in enumerate(zip(self.static_values, self.ssa_values)):
            ops, ssa = NumericResult._ssa_of(a, b)
            all_ops.extend(ops)
            new_ssa.append(ssa)

        self.used = True
        return NumericResult(all_ops, tuple([0] * len(new_ssa)), tuple(new_ssa))

    def add_to_llvm_pointer(self, ptr: SSAValue) -> tuple[list[Operation], SSAValue]:
        assert self.size == 1
        assert isinstance(ptr.type, llvm.LLVMPointerType)
        ops, (val,) = self.output()
        ops = list(ops)
        ops.append(cast_op := arith.IndexCastOp(val, i64))
        val = cast_op.result
        ops.append(ptr_to_int_op := llvm.PtrToIntOp(ptr))
        ops.append(add_op := arith.Addi(ptr_to_int_op.output, val))
        ops.append(int_to_ptr_op := llvm.IntToPtrOp(add_op.result))
        return ops, int_to_ptr_op.output

    def output(self) -> tuple[list[Operation], tuple[SSAValue, ...]]:
        s = self.to_ssa()
        assert all(a == 0 for a in s.static_values)
        s.used = True
        return list(s.operations), s.ssa_values

    def split(self) -> tuple[list[Operation], "NumericResult"]:
        ops, ssa_vals = self.output()
        return ops, NumericResult.from_mixed([], *ssa_vals)

    def split_n(self, n: int) -> tuple[list[Operation], tuple["NumericResult", ...]]:
        ops, ssa_vals = self.output()
        outputs = [NumericResult.from_mixed([], *ssa_vals) for _ in range(n)]
        return ops, tuple(outputs)

    @property
    def size(self) -> int:
        assert len(self.static_values) == len(self.ssa_values)
        if self.used:
            raise ValueError()
        return len(self.ssa_values)


NumericResult1 = typing.Annotated[NumericResult, 1]
NumericResult2 = typing.Annotated[NumericResult, 2]
NumericResult3 = typing.Annotated[NumericResult, 3]
NumericResult4 = typing.Annotated[NumericResult, 4]


def index_convert(
        input: SSAValue, target_type: IntegerType | IndexType
) -> tuple[list[Operation], SSAValue]:
    if input.type == target_type:
        return [], input
    ops = []
    input_type = input.type
    if not isinstance(input_type, IntegerType):
        assert isinstance(input_type, IndexType)
        input_cast_op = arith.IndexCastOp(input, builtin.i64)
        ops.append(input_cast_op)
        input = input_cast_op.result
        input_type = builtin.i64

    if isinstance(target_type, IntegerType):
        target_width = target_type.width
        target_int_type = target_type
    else:
        target_width = builtin.i64.width
        target_int_type = builtin.i64

    if input_type.width.data > target_width.data:
        trunc_op = arith.TruncIOp(input, target_int_type)
        ops.append(trunc_op)
        input = trunc_op.result
    elif input_type.width.data < target_width.data:
        extend_op = arith.ExtUIOp(input, target_int_type)
        ops.append(extend_op)
        input = extend_op.result

    if target_type == IndexType():
        output_cast_op = arith.IndexCastOp(input, IndexType())
        ops.append(output_cast_op)
        input = output_cast_op.result

    return ops, input

def _get_as_i64(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    return index_convert(value, i64)


def _get_as_index(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    return index_convert(value, IndexType())


class IndexGetter(abc.ABC):
    @abc.abstractmethod
    def get(self) -> NumericResult1:
        pass


class ArgIndexGetter(IndexGetter):
    def __init__(self, arg: SSAValue):
        self.arg = arg

    def get(self) -> NumericResult1:
        return NumericResult.from_mixed([], self.arg)


class ExtentGetter(abc.ABC):
    @abc.abstractmethod
    def get(self) -> NumericResult1:
        pass


# class StaticExtentGetter(ExtentGetter):
#     def __init__(self, arg: dlt.StaticExtentAttr):
#         assert arg.get_stage() <= dlt.Stage.STATIC
#         if isinstance(arg, dlt.StaticExtentAttr):
#             self.extent = arg.value.value.data
#         else:
#             raise NotImplementedError()
#
#     def get(self) -> NumericResult1:
#         return NumericResult.from_mixed([], self.extent)


class ScopeExtentGetter(ExtentGetter):
    def __init__(self, arg: IntegerAttr):
        if isinstance(arg.type, IndexType):
            self.extent = arg.value.data
        else:
            raise NotImplementedError()

    def get(self) -> NumericResult1:
        return NumericResult.from_mixed([], self.extent)


class SSAExtentGetter(ExtentGetter):
    def __init__(self, arg: SSAValue):
        self.arg = arg

    def get(self) -> NumericResult1:
        return NumericResult.from_mixed([], self.arg)


class PtrCarriedIndexGetter(IndexGetter):
    def __init__(
        self,
        dlt_llvm_ptr: SSAValue,
        ptr_type: dlt.PtrType,
        dim: dlt.DimensionAttr,
    ):
        self.input = dlt_llvm_ptr
        self.ptr_type = ptr_type
        self.dim = dim

    def get(self) -> NumericResult1:
        index = 1 + self.ptr_type.filled_dimensions.data.index(self.dim)
        pos = DenseArrayBase.from_list(i64, [index])
        op = llvm.ExtractValueOp(pos, self.input, i64)
        cast_ops, res = _get_as_index(op.res)
        return NumericResult.from_mixed([op] + cast_ops, res)


class PtrCarriedExtentGetter(ExtentGetter):
    def __init__(
        self,
        dlt_llvm_ptr: SSAValue,
        ptr_type: dlt.PtrType,
        extent: dlt.Extent,
    ):
        self.input = dlt_llvm_ptr
        self.ptr_type = ptr_type
        self.extent = extent

    def get(self) -> NumericResult1:
        index = (
            1
            + len(self.ptr_type.filled_dimensions)
            + self.ptr_type.filled_extents.data.index(self.extent)
        )
        pos = DenseArrayBase.from_list(i64, [index])
        op = llvm.ExtractValueOp(pos, self.input, i64)
        cast_ops, res = _get_as_index(op.res)
        return NumericResult.from_mixed([op] + cast_ops, res)


class ExtentResolver:

    def __init__(
        self, extent_map: dict[dlt.Extent, ExtentGetter], scope_op: dlt.LayoutScopeOp
    ):
        self.scope_op = scope_op
        new_map: dict[dlt.Extent, ExtentGetter] = {}
        for i, scope_extent in enumerate(scope_op.extent_names):
            new_map[scope_extent] = ScopeExtentGetter(scope_op.extent_values.data[i])
        new_map.update(extent_map)
        self.map: dict[dlt.Extent, ExtentGetter] = new_map

    def resolve(self, extent: dlt.Extent) -> NumericResult1:
        if extent.is_static():
            if isinstance(extent, dlt.StaticExtentAttr):
                extent = typing.cast(dlt.StaticExtentAttr, extent)
                return NumericResult.from_mixed([], extent.as_int())
            else:
                raise NotImplementedError()
        if extent in self.map:
            getter = self.map[extent]
            return getter.get()
        else:
            raise KeyError(
                f"Cannot resolve Extent {extent} in ExtentResolver map {self.map}"
            )

    def with_new(
        self, new_extent_mappings: dict[dlt.Extent, ExtentGetter]
    ) -> typing.Self:
        return ExtentResolver(self.map | new_extent_mappings, self.scope_op)


class Initialiser(abc.ABC):
    @abc.abstractmethod
    def get_value(
        self,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, IndexGetter],
        base_type: dlt.AcceptedTypes,
    ) -> tuple[list[Operation], None | SSAValue, bool | SSAValue]:
        raise NotImplementedError

    # returns ops, value, found where value is what should be written into storage. None is only allowed if found is
    # False. junk SSA of the correct type can be given when found (SSA) indicates it isn't found

    @abc.abstractmethod
    def get_non_zero(
        self,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, None | IndexGetter],
        type_type: dlt.TypeType,
    ) -> tuple[list[Operation], bool | SSAValue]:
        raise NotImplementedError

    # returns ops, non-zero where non-zero tells us if any value at these specifiers should be considered as non-zero
    # the type_type allows us to express that we wish to know if any of the values in that type (with the members and
    # dim_map dims added) are non-zero


class DerivedInitialiser(Initialiser):
    def __init__(
        self,
        original: Initialiser,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, IndexGetter | SSAValue],
    ):
        self.original = original
        self.members = members
        self.dim_map: dict[dlt.DimensionAttr, IndexGetter] = {
            dim: (ArgIndexGetter(idx) if isinstance(idx, SSAValue) else idx)
            for dim, idx in dim_map.items()
        }

    def get_value(
        self,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, IndexGetter],
        base_type: dlt.AcceptedTypes,
    ) -> tuple[list[Operation], None | SSAValue, bool | SSAValue]:
        return self.original.get_value(
            members | self.members, dim_map | self.dim_map, base_type
        )

    def get_non_zero(
        self,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, None | IndexGetter],
        type_type: dlt.TypeType,
    ) -> tuple[list[Operation], bool | SSAValue]:
        return self.original.get_non_zero(
            members | self.members, dim_map | self.dim_map, type_type
        )


class LoopCallback(abc.ABC):
    def __init__(self, initial_args: list[SSAValue] = None):
        self._initial_args = initial_args

    def initial_iter_args(self) -> list[SSAValue]:
        return [] if self._initial_args is None else self._initial_args

    @abc.abstractmethod
    def callback(
        self,
        terminal_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, IndexGetter],
        dims_left_to_loop: set[dlt.DimensionAttr],
        extent_resolver: ExtentResolver,
        ptr: SSAValue,
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue]]:
        raise NotImplementedError

    # returns ops, list of iter args to be passed in to the next invocation


class DerivedLoopCallback(LoopCallback):
    def __init__(
        self,
        original: LoopCallback,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, SSAValue | IndexGetter],
    ):
        self.original = original
        self.members = members
        self.dim_map = dim_map
        if original is not None:
            super().__init__(
                # original.typetype.add_members(members).add_dimensions(dim_map.keys()),
                original.initial_iter_args(),
            )

    def callback(
        self,
        terminal_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, SSAValue],
        dims_left_to_loop: set[dlt.DimensionAttr],
        extent_resolver: ExtentResolver,
        ptr: SSAValue,
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue]]:
        new_dim_map_parts = {}
        for dim, val in self.dim_map.items():
            if isinstance(val, SSAValue):
                val = ArgIndexGetter(val)
            new_dim_map_parts[dim] = val
        ops, iter_args_out = self.original.callback(
            terminal_layout,
            members | self.members,
            (dim_map | new_dim_map_parts),
            dims_left_to_loop,
            extent_resolver,
            ptr,
            iter_args,
        )
        return ops, iter_args_out


class Callback(abc.ABC):
    def __init__(self, initial_args: list[SSAValue] = None, can_exits_early=False):
        # self.typetype = typetype
        self._initial_args = initial_args
        self._can_exit_early = can_exits_early

    def initial_iter_args(self) -> list[SSAValue]:
        return [] if self._initial_args is None else self._initial_args

    def can_exits_early(self) -> bool:
        return self._can_exit_early

    @abc.abstractmethod
    def callback(
        self,
        terminal_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, SSAValue],
        extent_resolver: ExtentResolver,
        base_type: dlt.AcceptedTypes,
        ptr: SSAValue,
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        raise NotImplementedError

    # returns ops, list of iter args to be passed in to the next invocation, and a value deciding if the iterating
    # should stop early.


class NoCallback(Callback):

    def __init__(self):
        super().__init__([])

    def callback(
        self,
        terminal_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, SSAValue],
        extent_resolver: ExtentResolver,
        base_type: dlt.AcceptedTypes,
        ptr: SSAValue,
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        # print("DEBUG CALL BACK!")
        # debug_const = arith.Constant(IntegerAttr(100, IntegerType(20)))
        # debug_const.attributes["debug"] = builtin.StringAttr(
        #     "          NoCallBack here"
        # )
        # return [debug_const], [], False
        # ops = [printf.PrintFormatOp(f"NO-Callback ({len(dim_map)}):" + ",".join([str(d)+"{}" for d, v in dim_map.items()]), *dim_map.values())]
        # return ops, [], False
        return [], [], False


class DerivedCallback(Callback):
    def __init__(
        self,
        original: Callback,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, SSAValue],
    ):
        self.original = original
        self.members = members
        self.dim_map = dim_map
        if original is not None:
            super().__init__(
                # original.typetype.add_members(members).add_dimensions(dim_map.keys()),
                original.initial_iter_args(),
                original.can_exits_early(),
            )

    def callback(
        self,
        terminal_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, SSAValue],
        extent_resolver: ExtentResolver,
        base_type: dlt.AcceptedTypes,
        ptr: SSAValue,
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        ops, iter_args_out, exit_early = self.original.callback(
            terminal_layout,
            members | self.members,
            dim_map | self.dim_map,
            extent_resolver,
            base_type,
            ptr,
            has_extra_space,
            is_last_element,
            iter_args,
        )
        return ops, iter_args_out, exit_early


FillValueGetter = Callable[[], tuple[list[Operation], SSAValue]]
EnsureSpaceFunc = Callable[[FillValueGetter], tuple[list[Operation], SSAValue]]
LinearIterCallbackFunc = Callable[
    [Callback, bool | SSAValue], tuple[list[Operation], list[SSAValue], bool | SSAValue]
]
IterateInnerBodyFunc = Callable[
    [dict[int, SSAValue], list[SSAValue], InsertPoint], list[SSAValue]
]


class SemanticsMapper:
    def __init__(self, print_memory_calls: bool = False):
        self.print_memory_calls = print_memory_calls

        self.direct_map: dict[typing.Type[dlt.Layout], DirectLayoutNodeSemantics] = {}
        self.indexed_map: dict[typing.Type[dlt.Layout], IndexedLayoutNodeSemantics] = {}

    def add_direct(
        self, typ: typing.Type[dlt.Layout], node_semantics: "DirectLayoutNodeSemantics"
    ):
        self.direct_map[typ] = node_semantics

    def add_indexed(
        self, typ: typing.Type[dlt.Layout], node_semantics: "IndexedLayoutNodeSemantics"
    ):
        self.indexed_map[typ] = node_semantics

    def get_direct(self, layout: dlt.Layout) -> "DirectLayoutNodeSemantics":
        for t, s in reversed(self.direct_map.items()):
            if isinstance(layout, t):
                return s
        raise KeyError(f"Cannot find semantics for layout: {layout}")

    def get_indexed(self, layout: dlt.Layout) -> "IndexedLayoutNodeSemantics":
        for t, s in reversed(self.indexed_map.items()):
            if isinstance(layout, t):
                return s
        raise KeyError(f"Cannot find semantics for layout: {layout}")

    def get_any(self, layout: dlt.Layout) -> "LayoutNodeSemantics":
        for t, s in reversed(self.direct_map.items()):
            if isinstance(layout, t):
                return s
        for t, s in reversed(self.indexed_map.items()):
            if isinstance(layout, t):
                return s
        raise KeyError(f"Cannot find semantics for layout: {layout}")

    def get_size(
        self, layout: dlt.Layout, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        result = self.get_any(layout).get_size(layout, extent_resolver)
        assert result.size == 2
        return result

    def get_select_for(
        self,
        starting_layout: dlt.Layout,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        assert starting_layout.has_sub_layout(ending_layout)
        assert starting_layout.contents_type.has_selectable(members, dim_mapping.keys())
        if starting_layout == ending_layout:
            return [], input_ptr
        ops, val = self.get_direct(starting_layout).get_select_for(
            starting_layout,
            ending_layout,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
        )
        assert isinstance(val.type, llvm.LLVMPointerType)
        assert val.owner in ops or val == input_ptr
        return ops, val

    def get_getter_for(
        self,
        starting_layout: dlt.DirectLayout,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        assert starting_layout.contents_type.has_selectable(
            members, dim_mapping.keys(), get_type
        )
        ops, val, found = self.get_direct(starting_layout).get_getter_for(
            starting_layout, get_type, members, dim_mapping, extent_resolver, input_ptr
        )
        assert val.type == get_type
        assert val.owner in ops
        return ops, val, found

    def get_indexed_getter_for(
        self,
        starting_layout: dlt.IndexedLayout,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index: SSAValue,
        index_found: bool | SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        assert starting_layout.contents_type.has_selectable(
            members, dim_mapping.keys(), get_type
        )
        assert index.type == starting_layout.indexed_by()
        ops, val, found = self.get_indexed(starting_layout).get_getter_for(
            starting_layout,
            get_type,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
            index,
            index_found,
        )
        assert val.type == get_type
        assert val.owner in ops
        return ops, val, found

    def get_setter_for(
        self,
        starting_layout: dlt.DirectLayout,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        if not isinstance(input_ptr.type, llvm.LLVMPointerType):
            raise AssertionError()
        if not starting_layout.contents_type.has_selectable(
            members, dim_mapping.keys(), set_val.type
        ):
            raise AssertionError()
        ops = self.get_direct(starting_layout).get_setter_for(
            starting_layout, set_val, members, dim_mapping, extent_resolver, input_ptr
        )
        return ops

    def get_indexed_setter_for(
        self,
        starting_layout: dlt.IndexedLayout,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index: SSAValue,
        index_found: bool | SSAValue,
        ensure_space_func: EnsureSpaceFunc,
        direct_iterate_func: LinearIterCallbackFunc,
        direct_members: set[dlt.MemberAttr],
        direct_dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> list[Operation]:
        if not isinstance(input_ptr.type, llvm.LLVMPointerType):
            raise AssertionError()
        if not starting_layout.contents_type.has_selectable(
            members, dim_mapping.keys(), set_val.type
        ):
            raise AssertionError()
        assert index.type == starting_layout.indexed_by()
        ops = self.get_indexed(starting_layout).get_setter_for(
            starting_layout,
            set_val,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
            index,
            index_found,
            ensure_space_func,
            direct_iterate_func,
            direct_members,
            direct_dim_mapping,
        )
        return ops

    def init_layout(
        self,
        layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: bool | SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        assert len(callback_args) == len(init_callback.initial_iter_args())
        assert isinstance(
            input_ptr.type, llvm.LLVMPointerType
        ), f"Input pointer expected to be LLVMPointerType but found {type(input_ptr.type)}"
        # assert all(
        #     isinstance(val.type, dlt.PtrType) and val.type.contents_type == t
        #     for t, val in initial_values.items()
        # )
        ops: list[Operation] = []
        if isinstance(has_extra_space, bool):
            ops.append(
                op := arith.Constant(IntegerAttr(int(has_extra_space), IntegerType(1)))
            )
            has_extra_space = op.result
        init_ops, iter_args = self.get_direct(layout).init_layout(
            layout,
            extent_resolver,
            input_ptr,
            initial_values,
            init_callback,
            callback_args,
            has_extra_space,
            is_last_element,
        )
        ops.extend(init_ops)
        assert len(callback_args) == len(iter_args)
        return ops, iter_args

    def init_indexed_layout(
        self,
        layout: dlt.IndexedLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        is_last_element: bool | SSAValue,
        direct_init_func: LinearIterCallbackFunc,
        direct_iter_func: LinearIterCallbackFunc,
    ) -> tuple[list[Operation], list[SSAValue]]:
        assert isinstance(
            input_ptr.type, llvm.LLVMPointerType
        ), f"Input pointer expected to be LLVMPointerType but found {type(input_ptr.type)}"
        assert len(callback_args) == len(init_callback.initial_iter_args())
        ops: list[Operation] = []
        init_ops, iter_args = self.get_indexed(layout).init_layout(
            layout,
            extent_resolver,
            input_ptr,
            initial_values,
            init_callback,
            callback_args,
            is_last_element,
            direct_init_func,
            direct_iter_func,
        )
        ops.extend(init_ops)
        assert len(callback_args) == len(iter_args)
        return ops, iter_args

    def dealloc_layout(
        self,
        layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        assert isinstance(
            input_ptr.type, llvm.LLVMPointerType
        ), f"Input pointer expected to be LLVMPointerType but found {type(input_ptr.type)}"
        ops: list[Operation] = []

        # ptr_op = llvm.PtrToIntOp(input_ptr)
        # ops.append(ptr_op)
        # print_op = printf.PrintFormatOp(f"dealloc {type(layout)}) {{}}", ptr_op.output)
        # ops.append(print_op)

        init_ops = self.get_direct(layout).dealloc_layout(
            layout,
            extent_resolver,
            input_ptr,
        )
        ops.extend(init_ops)
        return ops

    def dealloc_indexed_layout(
        self,
        layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        direct_iter_func: LinearIterCallbackFunc,
    ) -> list[Operation]:
        assert isinstance(
            input_ptr.type, llvm.LLVMPointerType
        ), f"Input pointer expected to be LLVMPointerType but found {type(input_ptr.type)}"
        ops: list[Operation] = []

        # ptr_op = llvm.PtrToIntOp(input_ptr)
        # ops.append(ptr_op)
        # print_op = printf.PrintFormatOp(f"dealloc Indexed {type(layout)}) {{}}", ptr_op.output)
        # ops.append(print_op)

        init_ops = self.get_indexed(layout).dealloc_layout(
            layout,
            extent_resolver,
            input_ptr,
            direct_iter_func,
        )
        ops.extend(init_ops)
        return ops

    def linear_iterate(
        self,
        layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: bool | SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool = False,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        assert isinstance(
            input_ptr.type, llvm.LLVMPointerType
        ), f"Input pointer expected to be LLVMPointerType but found {type(input_ptr.type)}"
        # assert all(
        #     isinstance(val.type, dlt.PtrType) and val.type.contents_type == t
        #     for t, val in selected_data.items()
        # )
        ops: list[Operation] = []
        if isinstance(has_extra_space, bool):
            converted = True
            ops.append(
                op := arith.Constant(IntegerAttr(int(has_extra_space), IntegerType(1)))
            )
            has_extra_space = op.result
        else:
            converted = False
        init_ops, iter_args, exited_early = self.get_direct(layout).linear_iterate(
            layout,
            extent_resolver,
            input_ptr,
            callback,
            callback_args,
            has_extra_space,
            is_last_element,
            reversed_direction,
        )
        ops.extend(init_ops)
        return ops, iter_args, exited_early

    def linear_iterate_indexed(
        self,
        layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        is_last_element: bool | SSAValue,
        direct_iter_func: LinearIterCallbackFunc,
        reversed_direction: bool = False,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        assert isinstance(
            input_ptr.type, llvm.LLVMPointerType
        ), f"Input pointer expected to be LLVMPointerType but found {type(input_ptr.type)}"
        # assert all(
        #     isinstance(val.type, dlt.PtrType) and val.type.contents_type == t
        #     for t, val in selected_data.items()
        # )
        ops: list[Operation] = []
        init_ops, iter_args, exited_early = self.get_indexed(layout).linear_iterate(
            layout,
            extent_resolver,
            input_ptr,
            callback,
            callback_args,
            is_last_element,
            direct_iter_func,
            reversed_direction,
        )
        ops.extend(init_ops)
        return ops, iter_args, exited_early

    def ensure_space(
        self,
        layout: dlt.DirectLayout,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        # Return Ops, Index to use, if_unknown - return true if we cannot decide the indexrange
        return self.get_direct(layout).ensure_space(
            layout,
            base_type,
            members,
            dim_mapping,
            fill_value_getter,
            extent_resolver,
            input_ptr,
        )

    def ensure_indexed_space(
        self,
        layout: dlt.IndexedLayout,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index: SSAValue,
        direct_iterate_func: LinearIterCallbackFunc,
        direct_members: set[dlt.MemberAttr],
        direct_dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> tuple[list[Operation], SSAValue]:
        assert all(m not in members for m in direct_members)
        assert all(d not in dim_mapping for d in direct_dim_mapping)
        # Return Ops, Index to use, if_unknown - return true if we cannot decide the indexrange
        return self.get_indexed(layout).ensure_space(
            layout,
            base_type,
            members,
            dim_mapping,
            fill_value_getter,
            extent_resolver,
            input_ptr,
            index,
            direct_iterate_func,
            direct_members,
            direct_dim_mapping,
        )

    def make_sparse_loop_for(
        self,
        starting_layout: dlt.DirectLayout,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        assert starting_layout.has_sub_layout(ending_layout)
        assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        assert len(dims_to_loop & set(dim_mapping.keys())) == 0
        return self.get_direct(starting_layout).make_sparse_loop_for(
            starting_layout,
            ending_layout,
            extent_resolver,
            input_ptr,
            callback,
            callback_args,
            set(members),
            dict(dim_mapping),
            set(dims_to_loop),
        )

    def make_sparse_loop_callback_for(
        self,
        starting_layout: dlt.IndexedLayout,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], LoopCallback]:
        assert starting_layout.has_sub_layout(ending_layout)
        assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        assert len(dims_to_loop & set(dim_mapping.keys())) == 0
        return self.get_indexed(starting_layout).make_sparse_loop_callback_for(
            starting_layout,
            ending_layout,
            extent_resolver,
            input_ptr,
            callback,
            callback_args,
            set(members),
            dict(dim_mapping),
            set(dims_to_loop),
        )

    @staticmethod
    def get_data_type_from_dlt_ptr(ptr_type: dlt.PtrType) -> llvm.LLVMStructType:
        return llvm.LLVMStructType.from_type_list(
            [llvm.LLVMPointerType.opaque()]
            + len(ptr_type.filled_dimensions.data) * [i64]
            + len(ptr_type.filled_extents.data) * [i64]
        )

    def generate_ptr_struct(
        self,
        output_type: dlt.PtrType,
        allocated_ptr: SSAValue,
        dim_map: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
    ) -> tuple[list[Operation], SSAValue]:
        ops = [undef_op := llvm.UndefOp(self.get_data_type_from_dlt_ptr(output_type))]
        ptr_struct_result = undef_op.res
        ops.append(
            set_ptr_op := llvm.InsertValueOp(
                DenseArrayBase.from_list(i64, [0]), ptr_struct_result, allocated_ptr
            )
        )
        ptr_struct_result = set_ptr_op.res
        idx = 1
        for dim in output_type.filled_dimensions:
            dim_ops, (dim_result,) = dim_map[dim].get().output()
            ops.extend(dim_ops)
            as_i64_ops, dim_result = _get_as_i64(dim_result)
            ops.extend(as_i64_ops)
            ops.append(
                insert_op := llvm.InsertValueOp(
                    DenseArrayBase.from_list(i64, [idx]), ptr_struct_result, dim_result
                )
            )
            idx += 1
            ptr_struct_result = insert_op.res
        for extent in output_type.filled_extents:
            extent_ops, (extent_result,) = extent_resolver.resolve(extent).output()
            ops.extend(extent_ops)
            as_i64_ops, extent_result = _get_as_i64(extent_result)
            ops.extend(as_i64_ops)
            ops.append(
                insert_op := llvm.InsertValueOp(
                    DenseArrayBase.from_list(i64, [idx]),
                    ptr_struct_result,
                    extent_result,
                )
            )
            idx += 1
            ptr_struct_result = insert_op.res
        return ops, ptr_struct_result

    def extract_from_ptr_struct(
        self, dlt_ptr_type: dlt.PtrType, ptr: SSAValue
    ) -> tuple[
        list[Operation],
        SSAValue,
        dict[dlt.DimensionAttr, IndexGetter],
        dict[dlt.Extent, ExtentGetter],
    ]:
        ops = []
        if ptr.type != self.get_data_type_from_dlt_ptr(dlt_ptr_type):
            assert ptr.type == dlt_ptr_type
            cast = builtin.UnrealizedConversionCastOp.get(
                [ptr], [self.get_data_type_from_dlt_ptr(dlt_ptr_type)]
            )
            ops.append(cast)
            ptr = cast.outputs[0]
        index = 0
        get_ptr_op = llvm.ExtractValueOp(
            DenseArrayBase.from_list(i64, [index]), ptr, llvm.LLVMPointerType.opaque()
        )
        ops.append(get_ptr_op)
        data_ptr = SSAValue.get(get_ptr_op.res)
        index += 1
        dim_map = {
            dim: PtrCarriedIndexGetter(ptr, dlt_ptr_type, dim=dim)
            for dim in dlt_ptr_type.filled_dimensions
        }
        extent_map = {
            typing.cast(dlt.Extent, extent): PtrCarriedExtentGetter(
                ptr, dlt_ptr_type, extent=extent
            )
            for extent in dlt_ptr_type.filled_extents
        }
        return ops, data_ptr, dim_map, extent_map


T = typing.TypeVar("T", bound=dlt.Layout)


class LayoutNodeSemantics(abc.ABC, typing.Generic[T]):

    def __init__(self, semantics: SemanticsMapper):
        self.semantics = semantics

    @abc.abstractmethod
    def get_size(self, layout: T, extent_resolver: ExtentResolver) -> NumericResult2:
        raise NotImplementedError

    # @abc.abstractmethod
    # def get_iteration_for(
    #     self,
    #     starting_layout: dlt.Layout,
    #     members: set[dlt.MemberAttr],
    #     dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    #     extent_resolver: ExtentResolver,
    #     input_ptr: SSAValue,
    #     iterate_op: dlt.IterateOp,
    #     order: dlt.IterationOrder,
    #     iteration_args: list[SSAValue],
    #     indices_map: dict[int, SSAValue],
    #     insert_point: InsertPoint,
    #     inner_body_function: IterateInnerBodyFunc,
    # ) -> list[SSAValue]:
    #     raise NotImplementedError

    @staticmethod
    def _select_values_selector(
        values: dict[dlt.TypeType, SSAValue],
        members: typing.Iterable[dlt.MemberAttr],
        dims: dict[dlt.DimensionAttr, SSAValue],
    ) -> tuple[list[Operation], dict[dlt.TypeType, SSAValue]]:
        new_init_values = {}
        ops: list[Operation] = []
        for dlt_type, dlt_ptr in values.items():
            if dlt_type.has_selectable(members, dims.keys()):
                ops.append(
                    select_op := dlt.SelectOp(
                        dlt_ptr, [], list(dims.keys()), list(dims.values())
                    )
                )
                new_elem = typing.cast(dlt.PtrType, select_op.res.type).contents_type
                assert new_elem == dlt_type.select_members(members).select_dimensions(
                    dims.keys()
                )
                new_init_values[new_elem] = select_op.res

        return ops, new_init_values


class DirectLayoutNodeSemantics(typing.Generic[T], LayoutNodeSemantics[T], abc.ABC):

    @abc.abstractmethod
    def get_select_for(
        self,
        starting_layout: T,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        # assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        # assert starting_layout.has_sub_layout(ending_layout)
        # assert starting_layout.contents_type.has_selectable(members, dim_mapping.keys())
        raise NotImplementedError

    @abc.abstractmethod
    def get_getter_for(
        self,
        starting_layout: T,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        # Return ops, value (get_type), found (bool)
        # assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        # assert starting_layout.contents_type.has_selectable(members, dim_mapping.keys(), get_type)
        raise NotImplementedError

    @abc.abstractmethod
    def get_setter_for(
        self,
        starting_layout: T,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        # assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        # assert starting_layout.contents_type.has_selectable(members, dim_mapping.keys(), set_val.type)
        raise NotImplementedError

    @abc.abstractmethod
    def init_layout(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        # assert isinstance(input_ptr, llvm.LLVMPointerType)
        raise NotImplementedError

    @abc.abstractmethod
    def dealloc_layout(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        raise NotImplementedError

    @abc.abstractmethod
    def ensure_space(
        self,
        layout: T,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        raise NotImplementedError

    @abc.abstractmethod
    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        raise NotImplementedError

    @abc.abstractmethod
    def make_sparse_loop_for(
        self,
        starting_layout: T,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        raise NotImplementedError


class IndexedLayoutNodeSemantics(typing.Generic[T], LayoutNodeSemantics[T], abc.ABC):

    @abc.abstractmethod
    def get_getter_for(
        self,
        starting_layout: T,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index: SSAValue,
        index_found: bool | SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        # Return ops, value (get_type), found (bool)
        # assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        # assert starting_layout.contents_type.has_selectable(members, dim_mapping.keys(), get_type)
        raise NotImplementedError

    @abc.abstractmethod
    def get_setter_for(
        self,
        starting_layout: T,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index: SSAValue,
        index_found: bool | SSAValue,
        ensure_space_func: EnsureSpaceFunc,
        direct_iterate_func: LinearIterCallbackFunc,
        direct_members: set[dlt.MemberAttr],
        direct_dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> list[Operation]:
        # assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        # assert starting_layout.contents_type.has_selectable(members, dim_mapping.keys(), set_val.type)
        raise NotImplementedError

    @abc.abstractmethod
    def init_layout(
        self,
        layout: dlt.IndexedLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        is_last_element: bool | SSAValue,
        direct_init_func: LinearIterCallbackFunc,
        direct_iter_func: LinearIterCallbackFunc,
    ) -> tuple[list[Operation], list[SSAValue]]:
        raise NotImplementedError

    @abc.abstractmethod
    def dealloc_layout(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        direct_iter_func: LinearIterCallbackFunc,
    ) -> list[Operation]:
        raise NotImplementedError

    @abc.abstractmethod
    def ensure_space(
        self,
        layout: T,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index_range: SSAValue,
        direct_iterate_func: LinearIterCallbackFunc,
        direct_members: set[dlt.MemberAttr],
        direct_dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> tuple[list[Operation], SSAValue]:
        raise NotImplementedError

    @abc.abstractmethod
    def linear_iterate(
        self,
        layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        is_last_element: bool | SSAValue,
        direct_iter_func: LinearIterCallbackFunc,
        reversed_direction: bool = False,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        raise NotImplementedError

    @abc.abstractmethod
    def make_sparse_loop_callback_for(
        self,
        starting_layout: dlt.IndexedLayout,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], LoopCallback]:
        raise NotImplementedError


class PrimitiveSemantics(DirectLayoutNodeSemantics[dlt.PrimitiveLayoutAttr]):

    def get_size(
        self, layout: dlt.PrimitiveLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        return _get_accepted_type_size(layout.base_type)

    def get_select_for(
        self,
        starting_layout: dlt.PrimitiveLayoutAttr,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        if starting_layout != ending_layout:
            raise ValueError(
                "Cannot get select for terminal layout node dlt.layout.Primitive"
            )
        return [], input_ptr

    def get_getter_for(
        self,
        starting_layout: dlt.PrimitiveLayoutAttr,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        load_op = llvm.LoadOp(input_ptr, get_type)
        return [load_op], load_op.dereferenced_value, True

    def get_setter_for(
        self,
        starting_layout: dlt.PrimitiveLayoutAttr,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        assert set_val.type == starting_layout.base_type
        return [llvm.StoreOp(set_val, input_ptr)]

    @staticmethod
    def _set_zero_ops(
        base_type: dlt.AcceptedTypes, ptr: SSAValue, has_extra_space: SSAValue
    ):
        ops = []
        zero_ops, zero_val, zero_extra_val = _get_unpacked_zero_for_accepted_type(
            base_type
        )
        ops.extend(
            zero_ops
            + [
                llvm.StoreOp(zero_val, ptr),
            ]
        )
        if zero_extra_val is not None:
            if_extra_space = []
            ptr_ops, extra_ptr = (
                _get_accepted_type_size(base_type).keep(1).add_to_llvm_pointer(ptr)
            )
            if_extra_space.extend(
                ptr_ops + [llvm.StoreOp(zero_extra_val, extra_ptr), scf.Yield()]
            )
            if_op = scf.If(has_extra_space, [], if_extra_space)
            # if_op.attributes["debug"] = StringAttr(f"Prim-_set_zero_ops")
            ops.append(if_op)
        return ops

    def init_layout(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        ops: list[Operation] = []

        initialiser_ops, initial_value, initial_found = initial_values.get_value(
            set(), {}, layout.base_type
        )
        ops.extend(initialiser_ops)
        if initial_found is not False:
            assert initial_value.type == layout.base_type
            ssa_ops, initial_found = _make_bool_ssa(initial_found)
            ops.extend(ssa_ops)
            if_op = scf.If(
                initial_found,
                [],
                [llvm.StoreOp(initial_value, input_ptr), scf.Yield()],
                self._set_zero_ops(layout.base_type, input_ptr, has_extra_space)
                + [scf.Yield()],
            )
            # if_op.attributes["debug"] = StringAttr(f"Prim-_init_layout")
            ops.append(if_op)
        else:
            ops.extend(self._set_zero_ops(layout.base_type, input_ptr, has_extra_space))

        callback_ops, iter_args_out, exited = init_callback.callback(
            layout,
            set(),
            {},
            extent_resolver,
            layout.base_type,
            input_ptr,
            has_extra_space,
            is_last_element,
            callback_args,
        )
        ops.extend(callback_ops)

        return ops, iter_args_out

    def dealloc_layout(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        return []

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        callback_ops, iter_args_out, exited = callback.callback(
            layout,
            set(),
            {},
            extent_resolver,
            layout.base_type,
            input_ptr,
            has_extra_space,
            is_last_element,
            callback_args,
        )
        return callback_ops, iter_args_out, exited

    def ensure_space(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        # Not much to do here - the space already exist by definition.
        ops, val, found = self.get_getter_for(
            layout, base_type, members, dim_mapping, extent_resolver, input_ptr
        )
        # since a primitive always exists, there's no need to use the fill_value_getter
        return ops, val

    def make_sparse_loop_for(
        self,
        starting_layout: dlt.PrimitiveLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        assert starting_layout == ending_layout
        assert len(members) == 0
        assert len(dim_mapping) == 0
        assert len(dims_to_loop) == 0
        callback_ops, iter_args_out = callback.callback(
            starting_layout,
            set(),
            {},
            set(),
            extent_resolver,
            input_ptr,
            callback_args,
        )
        return callback_ops, iter_args_out


class ConstantSemantics(DirectLayoutNodeSemantics[dlt.PrimitiveLayoutAttr]):

    def get_size(
        self, layout: dlt.ConstantLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        return NumericResult.from_mixed([], 0, 0)

    def get_select_for(
        self,
        starting_layout: dlt.ConstantLayoutAttr,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        if starting_layout != ending_layout:
            raise ValueError(
                "Cannot get select for terminal layout node dlt.layout.Constant"
            )
        return [], input_ptr

    def get_getter_for(
        self,
        starting_layout: dlt.ConstantLayoutAttr,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        const_op = arith.Constant(starting_layout.base_data)
        return [const_op], const_op.result, True

    def get_setter_for(
        self,
        starting_layout: dlt.ConstantLayoutAttr,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        raise ValueError(
            "Cannot set value of Constant Layout - this is a Compile time variable."
        )

    def init_layout(
        self,
        layout: dlt.ConstantLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        # There is nothing it init in a constant
        # return [], callback_args]
        pass

    def dealloc_layout(
        self,
        layout: dlt.ConstantLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        return []

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue]]:
        # Nothing to do here - the space cannot exist by definition.
        raise AssertionError(
            "Const does not have 'space'. It cannot be Iterated in any order."
        )

    def make_sparse_loop_for(
        self,
        starting_layout: dlt.ConstantLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        assert starting_layout == ending_layout
        assert len(members) == 0
        assert len(dim_mapping) == 0
        assert len(dims_to_loop) == 0
        callback_ops, iter_args_out = callback.callback(
            starting_layout,
            set(),
            {},
            set(),
            extent_resolver,
            input_ptr,
            callback_args,
        )
        return callback_ops, iter_args_out

    def ensure_space(
        self,
        layout: dlt.ConstantLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        # Nothing to do here - the space cannot exist by definition.
        raise AssertionError("Const does not have 'space'. It cannot be ensured.")


class MemberSemantics(DirectLayoutNodeSemantics[dlt.MemberLayoutAttr]):

    def get_size(
        self, layout: dlt.MemberLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        return self.semantics.get_size(layout.child, extent_resolver)

    def get_select_for(
        self,
        starting_layout: dlt.MemberLayoutAttr,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        assert starting_layout.member_specifier in members
        return self.semantics.get_select_for(
            starting_layout.child,
            ending_layout,
            members - {starting_layout.member_specifier},
            dim_mapping,
            extent_resolver,
            input_ptr,
        )

    def get_getter_for(
        self,
        starting_layout: T,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        return self.semantics.get_getter_for(
            starting_layout.child,
            get_type,
            members - {starting_layout.member_specifier},
            dim_mapping,
            extent_resolver,
            input_ptr,
        )

    def get_setter_for(
        self,
        starting_layout: T,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        return self.semantics.get_setter_for(
            starting_layout.child,
            set_val,
            members - {starting_layout.member_specifier},
            dim_mapping,
            extent_resolver,
            input_ptr,
        )

    def init_layout(
        self,
        layout: dlt.MemberLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        ops = []
        new_initialiser = DerivedInitialiser(
            initial_values, {layout.member_specifier}, {}
        )
        new_callback = DerivedCallback(init_callback, {layout.member_specifier}, {})
        child_ops, iter_args_out = self.semantics.init_layout(
            layout.child,
            extent_resolver,
            input_ptr,
            new_initialiser,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
        )
        ops.extend(child_ops)
        return ops, iter_args_out

    def dealloc_layout(
        self,
        layout: dlt.MemberLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        return self.semantics.dealloc_layout(layout.child, extent_resolver, input_ptr)

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        ops = []
        new_callback = DerivedCallback(callback, {layout.member_specifier}, {})
        child_ops, iter_args_out, exited = self.semantics.linear_iterate(
            layout.child,
            extent_resolver,
            input_ptr,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
            reversed_direction,
        )
        ops.extend(child_ops)
        return ops, iter_args_out, exited

    def make_sparse_loop_for(
        self,
        starting_layout: dlt.MemberLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        if starting_layout == ending_layout:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out
        elif starting_layout.member_specifier in members:
            ops = []
            new_callback = DerivedLoopCallback(
                callback, {starting_layout.member_specifier}, {}
            )
            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                starting_layout.child,
                ending_layout,
                extent_resolver,
                input_ptr,
                new_callback,
                callback_args,
                members - {starting_layout.member_specifier},
                dim_mapping,
                dims_to_loop,
            )
            ops.extend(child_ops)
            return ops, iter_args_out
        else:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out

    def ensure_space(
        self,
        layout: dlt.MemberLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        return self.semantics.ensure_space(
            layout.child,
            base_type,
            members - {layout.member_specifier},
            dim_mapping,
            fill_value_getter,
            extent_resolver,
            input_ptr,
        )


class DenseSemantics(DirectLayoutNodeSemantics[dlt.DenseLayoutAttr]):

    def get_size(
        self, layout: dlt.DenseLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        child_size: NumericResult2 = self.semantics.get_size(
            layout.child, extent_resolver
        )
        extent_size: NumericResult1 = extent_resolver.resolve(layout.dimension.extent)

        return child_size * extent_size

    def get_offset(
        self,
        starting_layout: dlt.DenseLayoutAttr,
        index_getter: IndexGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        child_layout = starting_layout.child
        child_size: NumericResult1 = self.semantics.get_size(
            child_layout, extent_resolver
        ).keep(1)
        dim_val: NumericResult1 = index_getter.get()
        offset: NumericResult1 = child_size * dim_val
        ptr_ops, ptr = offset.add_to_llvm_pointer(input_ptr)
        return ptr_ops, ptr

    def get_select_for(
        self,
        starting_layout: dlt.DenseLayoutAttr,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:

        ptr_ops, ptr = self.get_offset(
            starting_layout,
            dim_mapping.pop(starting_layout.dimension),
            extent_resolver,
            input_ptr,
        )
        child_ops, child_res = self.semantics.get_select_for(
            starting_layout.child,
            ending_layout,
            members,
            dim_mapping,
            extent_resolver,
            ptr,
        )
        return ptr_ops + child_ops, child_res

    def get_getter_for(
        self,
        starting_layout: T,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        ptr_ops, ptr = self.get_offset(
            starting_layout,
            dim_mapping.pop(starting_layout.dimension),
            extent_resolver,
            input_ptr,
        )
        child_ops, child_res, child_found = self.semantics.get_getter_for(
            starting_layout.child, get_type, members, dim_mapping, extent_resolver, ptr
        )
        return ptr_ops + child_ops, child_res, child_found

    def get_setter_for(
        self,
        starting_layout: T,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        ptr_ops, ptr = self.get_offset(
            starting_layout,
            dim_mapping.pop(starting_layout.dimension),
            extent_resolver,
            input_ptr,
        )
        child_ops = self.semantics.get_setter_for(
            starting_layout.child, set_val, members, dim_mapping, extent_resolver, ptr
        )
        return ptr_ops + child_ops

    def ensure_space(
        self,
        layout: dlt.DenseLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        dim_getter = dim_mapping.pop(layout.dimension)
        ops: list[Operation] = []
        ptr_ops, ptr = self.get_offset(
            layout,
            dim_getter,
            extent_resolver,
            input_ptr,
        )
        ops.extend(ptr_ops)

        def current_fill_value_getter() -> tuple[list[Operation], SSAValue]:
            f_ops = []
            true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
            false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
            zero_op = arith.Constant(IntegerAttr(0, IndexType()))
            one_op = arith.Constant(IntegerAttr(1, IndexType()))
            f_ops.extend([true_op, false_op, zero_op, one_op])

            zero_idx_ops, zero_idx_range = _get_packed_zero_for_accepted_type(
                dlt.IndexRangeType()
            )
            f_ops.extend(zero_idx_ops)

            get_index_range_callback = GetFirstValueCallback(
                self.semantics, zero_idx_range
            )

            index_ops, (original_index,) = dim_getter.get().output()
            f_ops.extend(index_ops)

            data_elem_size_ops, (data_elem_size,) = (
                self.semantics.get_size(layout.child, extent_resolver).keep(1).output()
            )
            f_ops.extend(data_elem_size_ops)

            index_sub_one_op = arith.Subi(original_index, one_op.result)
            f_ops.append(index_sub_one_op)

            start_idx = index_sub_one_op.result
            end_idx = zero_op.result
            exit_now = false_op.result
            iter_arg = zero_idx_range

            condition_block = Block()
            c_index = condition_block.insert_arg(IndexType(), 0)
            c_exit = condition_block.insert_arg(IntegerType(1), 1)
            c_out_range = condition_block.insert_arg(dlt.IndexRangeType(), 2)
            condition_block.add_ops(
                [
                    cmp1 := arith.Cmpi(c_index, end_idx, "sge"),
                    cmp2 := arith.Cmpi(false_op.result, c_exit, "eq"),
                    cmp := arith.AndI(cmp1.result, cmp2.result),
                    scf.Condition(cmp, c_index, c_exit, c_out_range),
                ]
            )
            while_block = Block()
            w_index = while_block.insert_arg(IndexType(), 0)
            while_block.insert_arg(IntegerType(1), 1)  # w_exit
            w_index_range = while_block.insert_arg(dlt.IndexRangeType(), 2)

            elem_ops, elem_ptr = (
                NumericResult.from_ssa(data_elem_size) * NumericResult.from_ssa(w_index)
            ).add_to_llvm_pointer(input_ptr)
            while_block.add_ops(elem_ops)

            w_lin_iter_ops, w_callback_results, w_exited_early = (
                self.semantics.linear_iterate(
                    layout.child,
                    extent_resolver,
                    elem_ptr,
                    get_index_range_callback,
                    [w_index_range],
                    False,
                    False,
                    reversed_direction=True,
                )
            )
            while_block.add_ops(w_lin_iter_ops)
            bool_ops, w_exited_early = _make_bool_ssa(w_exited_early)
            while_block.add_ops(bool_ops)
            w_new_index_op = arith.Subi(w_index, one_op)
            while_block.add_op(w_new_index_op)
            while_block.add_op(
                scf.Yield(w_new_index_op.result, w_exited_early, w_callback_results[0])
            )

            while_op = scf.While(
                [start_idx, exit_now, iter_arg],
                [IndexType(), IntegerType(1), dlt.IndexRangeType()],
                [condition_block],
                [while_block],
            )
            f_ops.append(while_op)

            value_found = while_op.res[1]
            value = while_op.res[2]

            unpack_ops, _, fill_val = _extract_indices_from_index_range(value)
            f_ops.extend(unpack_ops)
            pack_ops, fill_value = _pack_indices_in_index_range(fill_val, fill_val)
            f_ops.extend(pack_ops)

            if_escalate, escalated_value = fill_value_getter()
            if_escalate.append(scf.Yield(escalated_value))
            if_op = scf.If(
                value_found, [base_type], [scf.Yield(fill_value)], if_escalate
            )
            # if_op.attributes["debug"] = StringAttr(f"Dense-Ensure_space")
            f_ops.append(if_op)

            return f_ops, if_op.output[0]

        child_ops, val = self.semantics.ensure_space(
            layout.child,
            base_type,
            members,
            dim_mapping,
            current_fill_value_getter,
            extent_resolver,
            ptr,
        )
        ops.extend(child_ops)

        return ops, val

    def init_layout(
        self,
        layout: dlt.DenseLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        child_size: NumericResult1 = self.semantics.get_size(
            layout.child, extent_resolver
        ).keep(1)
        ops: list[Operation] = []
        false_const = arith.Constant(IntegerAttr(0, IntegerType(1)))
        # false_const.attributes["debug"] = builtin.StringAttr(
        #     "                              Dense Init/iter Layout Start"
        # )
        ops.append(false_const)

        child_size_ops, child_size = child_size.split()
        ops.extend(child_size_ops)

        ops.append(lb := arith.Constant(IntegerAttr(0, IndexType())))
        ops.append(step := arith.Constant(IntegerAttr(1, IndexType())))
        ub_ops, (ub,) = extent_resolver.resolve(layout.dimension.extent).output()
        ops.extend(ub_ops)

        one_const = arith.Constant(IntegerAttr(1, IndexType()))
        last_iter_op = arith.Subi(ub, one_const.result)
        ops.extend([one_const, last_iter_op])

        if is_last_element is True:
            true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.append(true_const)
            is_last_element = true_const.result

        block = Block()  # loop body
        index = block.insert_arg(
            IndexType(), 0
        )  # index - to run through the dense dimension
        offset = child_size * NumericResult.from_mixed([], index)
        ptr_add_ops, ptr_arg = offset.add_to_llvm_pointer(input_ptr)
        block.add_ops(ptr_add_ops)

        new_initialiser = DerivedInitialiser(
            initial_values, set(), {layout.dimension: index}
        )

        new_callback = DerivedCallback(init_callback, set(), {layout.dimension: index})
        new_callback_args = []
        for arg in callback_args:
            new_callback_args.append(block.insert_arg(arg.type, len(block.args)))

        is_last_iter_op = arith.Cmpi(index, last_iter_op, "eq")
        block.add_op(is_last_iter_op)
        has_extra_if = scf.If(
            is_last_iter_op.result,
            [IntegerType(1)],
            [scf.Yield(has_extra_space)],
            [scf.Yield(false_const)],
        )
        # has_extra_if.attributes["debug"] = StringAttr(f"Dense-INit")
        block.add_op(has_extra_if)

        if isinstance(is_last_element, SSAValue):
            is_last_element_if = scf.If(
                is_last_iter_op.result,
                [IntegerType(1)],
                [scf.Yield(is_last_element)],
                [scf.Yield(false_const)],
            )
            # is_last_element_if.attributes["debug"] = StringAttr(f"Dense-Init_last_elem")
            block.add_op(is_last_element_if)
            is_last_element = is_last_element_if.output[0]

        child_ops, iter_args_out = self.semantics.init_layout(
            layout.child,
            extent_resolver,
            ptr_arg,
            new_initialiser,
            new_callback,
            new_callback_args,
            has_extra_if.output[0],
            is_last_element,
        )
        block.add_ops(child_ops)
        block.add_op(scf.Yield(*iter_args_out))

        loop = scf.For(lb, ub, step, callback_args, block)
        ops.append(loop)

        # debug_const = arith.Constant(IntegerAttr(100, IntegerType(20)))
        # debug_const.attributes["debug"] = builtin.StringAttr(
        #     "Dense Semantics Init/iter Layout end"
        # )
        # ops.append(debug_const)

        return ops, list(loop.res)

    def dealloc_layout(
        self,
        layout: dlt.DenseLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:

        ops, (extent,) = extent_resolver.resolve(layout.dimension.extent).output()
        zero_ops, (zero,) = NumericResult.from_const(0).output()
        one_ops, (one,) = NumericResult.from_const(1).output()
        size_ops, (elem_size,) = (
            self.semantics.get_size(layout.child, extent_resolver).keep(1).output()
        )
        ops.extend(zero_ops + one_ops + size_ops)
        block = Block()
        index = block.insert_arg(IndexType(), 0)
        child_ptr_ops, child_ptr = (
            NumericResult.from_ssa(index) * NumericResult.from_ssa(elem_size)
        ).add_to_llvm_pointer(input_ptr)
        block.add_ops(child_ptr_ops)
        block.add_ops(
            self.semantics.dealloc_layout(layout.child, extent_resolver, child_ptr)
        )
        block.add_op(scf.Yield())
        ops.append(scf.For(zero, extent, one, [], block))
        return ops

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        child_size: NumericResult1 = self.semantics.get_size(
            layout.child, extent_resolver
        ).keep(1)
        ops: list[Operation] = []
        false_const = arith.Constant(IntegerAttr(0, IntegerType(1)))
        # false_const.attributes["debug"] = builtin.StringAttr(
        #     "                              Dense Init/iter Layout Start"
        # )
        ops.append(false_const)

        child_size_ops, child_size = child_size.split()
        ops.extend(child_size_ops)

        one_const = arith.Constant(IntegerAttr(1, IndexType()))
        ops.append(one_const)

        if reversed_direction:
            start_idx_ops, (start_idx,) = extent_resolver.resolve(
                layout.dimension.extent
            ).output()
            ops.extend(start_idx_ops)
            ops.append(end_idx := arith.Constant(IntegerAttr(0, IndexType())))
            end_idx = end_idx.result
            ops.append(step := arith.Constant(IntegerAttr(-1, IndexType())))
            step = step.result
            ops.append(sub := arith.Subi(start_idx, step))
            start_idx = sub.result
            last_iter_idx = end_idx
        else:
            ops.append(start_idx := arith.Constant(IntegerAttr(0, IndexType())))
            start_idx = start_idx.result
            ops.append(step := arith.Constant(IntegerAttr(1, IndexType())))
            step = step.result
            end_idx_ops, (end_idx,) = extent_resolver.resolve(
                layout.dimension.extent
            ).output()
            ops.extend(end_idx_ops)
            last_iter_op = arith.Subi(end_idx, one_const.result)
            ops.append(last_iter_op)
            last_iter_idx = last_iter_op.result

        true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_const)
        if is_last_element is True:
            is_last_element = true_const.result

        block = Block()  # loop body
        index = block.insert_arg(
            IndexType(), 0
        )  # index - to run through the dense dimension
        block.insert_arg(IntegerType(1), 1)  # exited

        offset = child_size * NumericResult.from_mixed([], index)
        ptr_add_ops, ptr_arg = offset.add_to_llvm_pointer(input_ptr)
        block.add_ops(ptr_add_ops)

        new_callback = DerivedCallback(callback, set(), {layout.dimension: index})
        new_callback_args = []
        for arg in callback_args:
            new_callback_args.append(block.insert_arg(arg.type, len(block.args)))

        is_last_iter_op = arith.Cmpi(index, last_iter_idx, "eq")
        block.add_op(is_last_iter_op)
        has_extra_if = scf.If(
            is_last_iter_op.result,
            [IntegerType(1)],
            [scf.Yield(has_extra_space)],
            [scf.Yield(false_const)],
        )
        # has_extra_if.attributes["debug"] = StringAttr(f"Dense_Lin_iter_exta")
        block.add_op(has_extra_if)

        if isinstance(is_last_element, SSAValue):
            is_last_element_if = scf.If(
                is_last_iter_op.result,
                [IntegerType(1)],
                [scf.Yield(is_last_element)],
                [scf.Yield(false_const)],
            )
            # is_last_element_if.attributes["debug"] = StringAttr(f"Dense_lin)iter_last")
            block.add_op(is_last_element_if)
            is_last_element = is_last_element_if.output[0]

        child_ops, iter_args_out, inner_exited = self.semantics.linear_iterate(
            layout.child,
            extent_resolver,
            ptr_arg,
            new_callback,
            new_callback_args,
            has_extra_if.output[0],
            is_last_element,
            reversed_direction,
        )
        block.add_ops(child_ops)

        exited_ops, inner_exited = _make_bool_ssa(inner_exited)
        block.add_ops(exited_ops)

        inc_index_op = arith.Addi(index, step)
        block.add_op(inc_index_op)

        block.add_op(scf.Yield(inc_index_op.result, inner_exited, *iter_args_out))

        condition_block = Block()
        c_index = condition_block.insert_arg(IndexType(), 0)
        c_exit = condition_block.insert_arg(IntegerType(1), 1)
        c_block_inner_callback_args = [
            condition_block.insert_arg(arg.type, len(condition_block.args))
            for arg in callback_args
        ]
        if reversed_direction:
            cmp1 = arith.Cmpi(c_index, end_idx, "sge")
        else:
            cmp1 = arith.Cmpi(c_index, end_idx, "slt")
        condition_block.add_ops(
            [
                cmp1,
                cmp2 := arith.Cmpi(false_const.result, c_exit, "eq"),
                cmp := arith.AndI(cmp1.result, cmp2.result),
                scf.Condition(cmp, c_index, c_exit, *c_block_inner_callback_args),
            ]
        )

        while_loop = scf.While(
            [start_idx, false_const.result, *callback_args],
            [IndexType(), IntegerType(1), *[a.type for a in callback_args]],
            [condition_block],
            [block],
        )
        ops.append(while_loop)
        output_callback_iter_args = list(while_loop.results[2 : 2 + len(callback_args)])
        did_exit_early = while_loop.results[1]

        # debug_const = arith.Constant(IntegerAttr(100, IntegerType(20)))
        # debug_const.attributes["debug"] = builtin.StringAttr(
        #     "Dense Semantics Init/iter Layout end"
        # )
        # ops.append(debug_const)

        return ops, output_callback_iter_args, did_exit_early

    def make_sparse_loop_for(
        self,
        starting_layout: dlt.DenseLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        if starting_layout == ending_layout:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out
        elif starting_layout.dimension in dim_mapping:
            idx_getter = dim_mapping.pop(starting_layout.dimension)
            idx_ops, (index,) = idx_getter.get().output()
            ptr_ops, ptr = self.get_offset(
                starting_layout,
                ArgIndexGetter(index),
                extent_resolver,
                input_ptr,
            )
            new_callback = DerivedLoopCallback(
                callback, set(), {starting_layout.dimension: index}
            )
            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                starting_layout.child,
                ending_layout,
                extent_resolver,
                ptr,
                new_callback,
                callback_args,
                members,
                dim_mapping,
                dims_to_loop,
            )
            return idx_ops + ptr_ops + child_ops, iter_args_out
        elif starting_layout.dimension in dims_to_loop:
            child_size: NumericResult1 = self.semantics.get_size(
                starting_layout.child, extent_resolver
            ).keep(1)
            ops: list[Operation] = []

            true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.append(true_const)
            false_const = arith.Constant(IntegerAttr(0, IntegerType(1)))
            ops.append(false_const)
            one_const = arith.Constant(IntegerAttr(1, IndexType()))
            ops.append(one_const)
            zero_const = arith.Constant(IntegerAttr(0, IndexType()))
            ops.append(zero_const)

            child_size_ops, child_size = child_size.split()
            ops.extend(child_size_ops)

            start_idx = zero_const.result
            step = one_const.result
            end_idx_ops, (end_idx,) = extent_resolver.resolve(
                starting_layout.dimension.extent
            ).output()
            ops.extend(end_idx_ops)

            block = Block()  # loop body
            index = block.insert_arg(
                IndexType(), 0
            )  # index - to run through the dense dimension
            offset = child_size * NumericResult.from_mixed([], index)
            ptr_add_ops, ptr_arg = offset.add_to_llvm_pointer(input_ptr)
            block.add_ops(ptr_add_ops)

            new_callback = DerivedLoopCallback(
                callback, set(), {starting_layout.dimension: index}
            )
            new_callback_args = []
            for arg in callback_args:
                new_callback_args.append(block.insert_arg(arg.type, len(block.args)))

            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                starting_layout.child,
                ending_layout,
                extent_resolver,
                ptr_arg,
                new_callback,
                new_callback_args,
                members,
                dim_mapping,
                dims_to_loop - {starting_layout.dimension},
            )
            block.add_ops(child_ops)

            block.add_op(scf.Yield(*iter_args_out))
            for_loop = scf.For(start_idx, end_idx, step, callback_args, block)
            ops.append(for_loop)
            output_callback_iter_args = list(for_loop.res)

            return ops, output_callback_iter_args
        else:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out


class StructSemantics(DirectLayoutNodeSemantics[dlt.StructLayoutAttr]):
    def get_size(
        self, layout: dlt.StructLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        result = NumericResult.from_mixed([], 0)
        for child in layout.children:
            c_size = self.semantics.get_size(child, extent_resolver).sum()
            result = result + c_size
        return result.extend(0)

    @overload
    def pick_child(
        self,
        starting_layout: dlt.StructLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        none_okay: typing.Literal[False] = False,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout, int]: ...

    @overload
    def pick_child(
        self,
        starting_layout: dlt.StructLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        none_okay: typing.Literal[True] = False,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout, int] | None: ...

    def pick_child(
        self,
        starting_layout: dlt.StructLayoutAttr,
        members: set[dlt.MemberAttr],
        dimensions: set[dlt.DimensionAttr],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        none_okay: typing.Literal[True] | typing.Literal[False] = False,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout, int] | None:
        if (
            sum(
                1
                for c in starting_layout.children
                if c.contents_type.has_selectable(members, dimensions) > 0
            )
            != 1
        ):
            if none_okay:
                return None
            else:
                raise ValueError(
                    f"Cannot select ambiguously, but there are multiple possible children that could be ment by selecting: "
                    f"{members} and {dimensions} in {[c.contents_type for c in starting_layout.children]}"
                )
        child = None
        child_idx = None
        offset = NumericResult.from_mixed([], 0)
        for i, child_layout in enumerate(starting_layout.children):
            child_layout: dlt.Layout = child_layout
            if child_layout.contents_type.has_selectable(members, dimensions) > 0:
                child = child_layout
                child_idx = i
                break
            else:
                c_offset = self.semantics.get_size(child_layout, extent_resolver).sum()
                offset = offset + c_offset
        ptr_ops, ptr = offset.add_to_llvm_pointer(input_ptr)
        return ptr_ops, ptr, child, child_idx

    def get_select_for(
        self,
        starting_layout: dlt.StructLayoutAttr,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        ptr_ops, ptr, child, _ = self.pick_child(
            starting_layout,
            members,
            set(dim_mapping.keys()),
            extent_resolver,
            input_ptr,
        )
        child_ops, child_res = self.semantics.get_select_for(
            child, ending_layout, members, dim_mapping, extent_resolver, ptr
        )
        return ptr_ops + child_ops, child_res

    def get_getter_for(
        self,
        starting_layout: T,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        ptr_ops, ptr, child, _ = self.pick_child(
            starting_layout,
            members,
            set(dim_mapping.keys()),
            extent_resolver,
            input_ptr,
        )
        child_ops, child_res, child_found = self.semantics.get_getter_for(
            child, get_type, members, dim_mapping, extent_resolver, ptr
        )
        return ptr_ops + child_ops, child_res, child_found

    def get_setter_for(
        self,
        starting_layout: T,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        ptr_ops, ptr, child, _ = self.pick_child(
            starting_layout,
            members,
            set(dim_mapping.keys()),
            extent_resolver,
            input_ptr,
        )
        child_ops = self.semantics.get_setter_for(
            child, set_val, members, dim_mapping, extent_resolver, ptr
        )
        return ptr_ops + child_ops

    def ensure_space(
        self,
        layout: dlt.StructLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        ops: list[Operation] = []
        ptr_ops, ptr, child, child_idx = self.pick_child(
            layout, members, set(dim_mapping.keys()), extent_resolver, input_ptr
        )
        ops.extend(ptr_ops)

        def current_fill_value_getter() -> tuple[list[Operation], SSAValue]:
            f_ops = []
            true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
            false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
            zero_op = arith.Constant(IntegerAttr(0, IndexType()))
            one_op = arith.Constant(IntegerAttr(1, IndexType()))
            ops.extend([true_op, false_op, zero_op, one_op])

            zero_idx_ops, zero_idx_range = _get_packed_zero_for_accepted_type(
                dlt.IndexRangeType()
            )
            f_ops.extend(zero_idx_ops)

            running_known = true_op.result
            running_val = zero_idx_range

            for idx, other_child in reversed(
                list(enumerate(list(layout.children)[:child_idx]))
            ):
                offset = NumericResult.from_const(0)
                for c in list(layout.children)[:idx]:
                    c_offset = self.semantics.get_size(c, extent_resolver).sum()
                    offset = offset + c_offset
                data_ptr_ops, data_ptr = offset.add_to_llvm_pointer(input_ptr)
                f_ops.extend(data_ptr_ops)

                get_index_range_callback = GetFirstValueCallback(
                    self.semantics, zero_idx_range
                )
                lin_iter_ops, callback_results, exited_early = (
                    self.semantics.linear_iterate(
                        other_child,
                        extent_resolver,
                        data_ptr,
                        get_index_range_callback,
                        [val],
                        False,
                        False,
                        reversed_direction=True,
                    )
                )
                bool_ops, exited_early = _make_bool_ssa(exited_early)
                if_op = scf.If(
                    running_known,
                    [IntegerType(1), base_type],
                    [scf.Yield(true_op.result, running_val)],
                    lin_iter_ops
                    + bool_ops
                    + [scf.Yield(exited_early, callback_results[0])],
                )
                # if_op.attributes["debug"] = StringAttr(f"Struct_ensure")
                f_ops.append(if_op)
                running_known = if_op.output[0]
                running_val = if_op.output[1]

            unpack_ops, _, fill_val = _extract_indices_from_index_range(running_val)
            f_ops.extend(unpack_ops)
            pack_ops, fill_value = _pack_indices_in_index_range(fill_val, fill_val)
            f_ops.extend(pack_ops)

            if_escalate, escalated_value = fill_value_getter()
            if_escalate.append(scf.Yield(escalated_value))
            if_op = scf.If(
                running_known, [base_type], [scf.Yield(fill_value)], if_escalate
            )
            # if_op.attributes["debug"] = StringAttr(f"Struct_ensure_2")
            f_ops.append(if_op)

            return f_ops, if_op.output[0]

        child_ops, val = self.semantics.ensure_space(
            child,
            base_type,
            members,
            dim_mapping,
            current_fill_value_getter,
            extent_resolver,
            ptr,
        )
        ops.extend(child_ops)

        return ops, val

    def init_layout(
        self,
        layout: dlt.StructLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        ops: list[Operation] = []
        true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_const)
        current_input_ptr = input_ptr
        current_callback_args = callback_args

        for i, child in enumerate(layout.children):
            child_callback = DerivedCallback(init_callback, set(), {})
            child_is_last_element = (
                is_last_element if i == len(layout.children) - 1 else False
            )
            child_ops, current_callback_args = self.semantics.init_layout(
                child,
                extent_resolver,
                current_input_ptr,
                initial_values,
                child_callback,
                current_callback_args,
                true_const.result,
                child_is_last_element,
            )
            ops.extend(child_ops)
            if (i - 1) < len(layout.children):
                child_offset = self.semantics.get_size(child, extent_resolver).sum()
                ptr_ops, current_input_ptr = child_offset.add_to_llvm_pointer(
                    current_input_ptr
                )
                ops.extend(ptr_ops)
        return ops, current_callback_args

    def dealloc_layout(
        self,
        layout: dlt.StructLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:

        ops = []
        current_ptr = input_ptr
        for i, child in enumerate(layout.children):
            ops.extend(
                self.semantics.dealloc_layout(child, extent_resolver, current_ptr)
            )
            if (i - 1) < len(layout.children):
                ptr_ops, current_ptr = (
                    self.semantics.get_size(child, extent_resolver)
                    .sum()
                    .add_to_llvm_pointer(current_ptr)
                )
                ops.extend(ptr_ops)
        return ops

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        ops: list[Operation] = []
        true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_const)
        false_const = arith.Constant(IntegerAttr(0, IntegerType(1)))
        ops.append(false_const)

        current_exit = false_const.result
        current_ptr = input_ptr
        current_callback_args = callback_args

        if reversed_direction:
            child_list = list(reversed(layout.children))
        else:
            child_list = list(layout.children)

        for i, child in enumerate(child_list):
            if_ops = []
            child_callback = DerivedCallback(callback, set(), {})
            child_is_last_element = (
                is_last_element if i == len(layout.children) - 1 else False
            )
            child_ops, new_callback_args, new_exit = self.semantics.linear_iterate(
                child,
                extent_resolver,
                current_ptr,
                child_callback,
                current_callback_args,
                true_const.result,
                child_is_last_element,
                reversed_direction,
            )
            if_ops.extend(child_ops)

            exit_ops, new_exit = _make_bool_ssa(new_exit)
            if_ops.extend(exit_ops)
            child_offset = self.semantics.get_size(child, extent_resolver).sum()
            ptr_ops, new_ptr = child_offset.add_to_llvm_pointer(current_ptr)
            if_ops.extend(ptr_ops)
            if_ops.append(scf.Yield(new_exit, new_ptr, *new_callback_args))
            if_op = scf.If(
                current_exit,
                [
                    IntegerType(1),
                    llvm.LLVMPointerType.opaque(),
                    *[a.type for a in callback_args],
                ],
                [scf.Yield(true_const.result, current_ptr, *current_callback_args)],
                if_ops,
            )
            # if_op.attributes["debug"] = StringAttr(f"Struct_lin_iter")
            ops.append(if_op)
            current_exit = if_op.output[0]
            current_ptr = if_op.output[1]
            current_callback_args = list(if_op.output[2 : 2 + len(callback_args)])

        return ops, current_callback_args, current_exit

    def make_sparse_loop_for(
        self,
        starting_layout: dlt.StructLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        if starting_layout == ending_layout:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out
        elif (
            child_data := self.pick_child(
                starting_layout,
                members,
                set(dim_mapping.keys()) | dims_to_loop,
                extent_resolver,
                input_ptr,
                none_okay=True,
            )
        ) is not None:
            ptr_ops, ptr, child, child_idx = child_data
            new_callback = DerivedLoopCallback(callback, set(), {})
            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                child,
                ending_layout,
                extent_resolver,
                ptr,
                new_callback,
                callback_args,
                members,
                dim_mapping,
                dims_to_loop,
            )
            return ptr_ops + child_ops, iter_args_out
        else:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out


class ArithDropSemantics(DirectLayoutNodeSemantics[dlt.ArithDropLayoutAttr]):

    def get_size(
        self, layout: dlt.ArithDropLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        return self.semantics.get_size(layout.child, extent_resolver)

    def get_select_for(
        self,
        starting_layout: dlt.ArithDropLayoutAttr,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        dim_mapping.pop(starting_layout.dimension)
        child_ops, child_res = self.semantics.get_select_for(
            starting_layout.child,
            ending_layout,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
        )
        return child_ops, child_res

    def get_getter_for(
        self,
        starting_layout: dlt.ArithDropLayoutAttr,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        dim_mapping.pop(starting_layout.dimension)
        child_ops, child_res, child_found = self.semantics.get_getter_for(
            starting_layout.child,
            get_type,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
        )
        return child_ops, child_res, child_found

    def get_setter_for(
        self,
        starting_layout: dlt.ArithDropLayoutAttr,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        dim_mapping.pop(starting_layout.dimension)
        child_ops = self.semantics.get_setter_for(
            starting_layout.child,
            set_val,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
        )
        return child_ops

    def ensure_space(
        self,
        layout: dlt.ArithDropLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        dim_mapping.pop(layout.dimension)
        return self.semantics.ensure_space(
            layout.child,
            base_type,
            members,
            dim_mapping,
            fill_value_getter,
            extent_resolver,
            input_ptr,
        )

    def init_layout(
        self,
        layout: dlt.ArithDropLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        zero_const: NumericResult1 = NumericResult.from_mixed([], 0)
        ops, (zero,) = zero_const.output()

        new_initialiser = DerivedInitialiser(
            initial_values, set(), {layout.dimension: zero}
        )
        new_callback = DerivedCallback(init_callback, set(), {layout.dimension: zero})

        child_ops, iter_args_out = self.semantics.init_layout(
            layout.child,
            extent_resolver,
            input_ptr,
            new_initialiser,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
        )
        ops.extend(child_ops)
        return ops, iter_args_out

    def dealloc_layout(
        self,
        layout: dlt.ArithDropLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        return self.semantics.dealloc_layout(layout.child, extent_resolver, input_ptr)

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        zero_const: NumericResult1 = NumericResult.from_mixed([], 0)
        ops, (zero,) = zero_const.output()

        new_callback = DerivedCallback(callback, set(), {layout.dimension: zero})

        child_ops, iter_args_out, exited = self.semantics.linear_iterate(
            layout.child,
            extent_resolver,
            input_ptr,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
            reversed_direction,
        )
        ops.extend(child_ops)
        return ops, iter_args_out, exited

    def make_sparse_loop_for(
        self,
        starting_layout: dlt.ArithDropLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        if starting_layout == ending_layout:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out
        elif starting_layout.dimension in dim_mapping:
            idx_getter = dim_mapping.pop(starting_layout.dimension)
            idx_ops, (index) = idx_getter.get().output()
            new_callback = DerivedLoopCallback(
                callback, set(), {starting_layout.dimension: index}
            )
            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                starting_layout.child,
                ending_layout,
                extent_resolver,
                input_ptr,
                new_callback,
                callback_args,
                members,
                dim_mapping,
                dims_to_loop,
            )
            return idx_ops + child_ops, iter_args_out
        elif starting_layout.dimension in dims_to_loop:
            child_size: NumericResult1 = self.semantics.get_size(
                starting_layout.child, extent_resolver
            ).keep(1)
            ops: list[Operation] = []

            true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.append(true_const)
            false_const = arith.Constant(IntegerAttr(0, IntegerType(1)))
            ops.append(false_const)
            one_const = arith.Constant(IntegerAttr(1, IndexType()))
            ops.append(one_const)
            zero_const = arith.Constant(IntegerAttr(0, IndexType()))
            ops.append(zero_const)

            child_size_ops, child_size = child_size.split()
            ops.extend(child_size_ops)

            start_idx = zero_const.result
            step = one_const.result
            end_idx_ops, (end_idx,) = extent_resolver.resolve(
                starting_layout.dimension.extent
            ).output()
            ops.extend(end_idx_ops)

            block = Block()  # loop body
            index = block.insert_arg(
                IndexType(), 0
            )  # index - to run through the dense dimension
            new_callback = DerivedLoopCallback(
                callback, set(), {starting_layout.dimension: index}
            )
            new_callback_args = []
            for arg in callback_args:
                new_callback_args.append(block.insert_arg(arg.type, len(block.args)))

            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                starting_layout.child,
                ending_layout,
                extent_resolver,
                input_ptr,
                new_callback,
                new_callback_args,
                members,
                dim_mapping,
                dims_to_loop - {starting_layout.dimension},
            )
            block.add_ops(child_ops)

            block.add_op(scf.Yield(*iter_args_out))
            for_loop = scf.For(start_idx, end_idx, step, callback_args, block)
            ops.append(for_loop)
            output_callback_iter_args = list(for_loop.res)

            return ops, output_callback_iter_args
        else:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out


class ArithReplaceSemantics(DirectLayoutNodeSemantics[dlt.ArithReplaceLayoutAttr]):

    def get_size(
        self, layout: dlt.ArithReplaceLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        return self.semantics.get_size(layout.child, extent_resolver)

    @staticmethod
    def get_selectors(
        layout: dlt.ArithReplaceLayoutAttr,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> tuple[set[dlt.MemberAttr], dict[dlt.DimensionAttr, IndexGetter]]:
        dim_mapping = dict(dim_mapping)
        dims = set(dim_mapping.keys()) & layout.outer_dimensions()
        if len(dims) != 1:
            raise ValueError("Cannot select dimension to replace")
        dim = dims.pop()
        replacement = layout.replacement_for(dim)

        idx_getter = dim_mapping.pop(replacement.outer_dimension)
        dim_mapping[replacement.inner_dimension] = idx_getter

        return members | {replacement.inner_member}, dim_mapping

    def get_select_for(
        self,
        starting_layout: dlt.ArithReplaceLayoutAttr,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        members, dim_mapping = self.get_selectors(starting_layout, members, dim_mapping)
        child_ops, child_res = self.semantics.get_select_for(
            starting_layout.child,
            ending_layout,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
        )
        return child_ops, child_res

    def get_getter_for(
        self,
        starting_layout: dlt.ArithReplaceLayoutAttr,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        members, dim_mapping = self.get_selectors(starting_layout, members, dim_mapping)
        child_ops, child_res, child_found = self.semantics.get_getter_for(
            starting_layout.child,
            get_type,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
        )
        return child_ops, child_res, child_found

    def get_setter_for(
        self,
        starting_layout: dlt.ArithReplaceLayoutAttr,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        new_members, new_dim_mapping = self.get_selectors(
            starting_layout, members, dim_mapping
        )
        child_ops = self.semantics.get_setter_for(
            starting_layout.child,
            set_val,
            new_members,
            new_dim_mapping,
            extent_resolver,
            input_ptr,
        )
        return child_ops

    def ensure_space(
        self,
        layout: dlt.ArithReplaceLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        members, dim_mapping = self.get_selectors(layout, members, dim_mapping)
        return self.semantics.ensure_space(
            layout.child,
            base_type,
            members,
            dim_mapping,
            fill_value_getter,
            extent_resolver,
            input_ptr,
        )

    class ReplacementCallback(Callback):
        def __init__(self, layout: dlt.ArithReplaceLayoutAttr, original: Callback):
            self.layout = layout
            self.original = original
            super().__init__(original.initial_iter_args(), original.can_exits_early())

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, SSAValue],
            extent_resolver: ExtentResolver,
            base_type: dlt.AcceptedTypes,
            ptr: SSAValue,
            has_extra_space: SSAValue,
            is_last_element: bool | SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            replacements = {
                r
                for r in self.layout.replacements
                if r.inner_member in members and r.inner_dimension in dim_map
            }
            if len(replacements) != 1:
                raise ValueError()
            replacement = replacements.pop()

            new_members = members - {replacement.inner_member}
            new_dim_map = {
                dim: val
                for dim, val in dim_map.items()
                if dim != replacement.inner_dimension
            } | {replacement.outer_dimension: dim_map[replacement.inner_dimension]}
            ops, iter_args_out, exit_early = self.original.callback(
                terminal_layout,
                new_members,
                new_dim_map,
                extent_resolver,
                base_type,
                ptr,
                has_extra_space,
                is_last_element,
                iter_args,
            )
            return ops, iter_args_out, exit_early

    class ReplacementLoopCallback(LoopCallback):
        def __init__(self, layout: dlt.ArithReplaceLayoutAttr, original: LoopCallback):
            self.layout = layout
            self.original = original
            super().__init__(original.initial_iter_args())

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, IndexGetter],
            dims_left_to_loop: set[dlt.DimensionAttr],
            extent_resolver: ExtentResolver,
            ptr: SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue]]:
            replacements = {
                r
                for r in self.layout.replacements
                if r.inner_member in members
                and r.inner_dimension in (set(dim_map.keys()) | dims_left_to_loop)
            }
            if len(replacements) != 1:
                raise ValueError()
            replacement = replacements.pop()

            dim_in_map = False
            dim_in_loop = False

            new_members = members - {replacement.inner_member}
            new_dim_map = {
                dim: val
                for dim, val in dim_map.items()
                if dim != replacement.inner_dimension
            }
            if len(new_dim_map) < len(dim_map):
                dim_in_map = True
                new_dim_map |= {
                    replacement.outer_dimension: dim_map[replacement.inner_dimension]
                }

            new_dims_left_to_loop = {
                dim for dim in dims_left_to_loop if dim != replacement.inner_dimension
            }
            if len(new_dims_left_to_loop) < len(dims_left_to_loop):
                dim_in_loop = True
                new_dims_left_to_loop |= {
                    replacement.outer_dimension: dim_map[replacement.inner_dimension]
                }

            assert not (dim_in_map and dim_in_loop)

            ops, iter_args_out = self.original.callback(
                terminal_layout,
                new_members,
                new_dim_map,
                new_dims_left_to_loop,
                extent_resolver,
                ptr,
                iter_args,
            )
            return ops, iter_args_out

    class ReplacementInitialiser(Initialiser):
        def __init__(self, layout: dlt.ArithReplaceLayoutAttr, original: Initialiser):
            self.layout = layout
            self.original = original
            super().__init__()

        def get_value(
            self,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, IndexGetter],
            base_type: dlt.AcceptedTypes,
        ) -> tuple[list[Operation], None | SSAValue, bool | SSAValue]:
            replacements = {
                r for r in self.layout.replacements if r.inner_member in members
            }
            if len(replacements) != 1:
                raise ValueError()
            replacement = replacements.pop()

            new_members = members - {replacement.inner_member}
            new_dim_map = {
                dim: val
                for dim, val in dim_map.items()
                if dim != replacement.inner_dimension
            } | {replacement.outer_dimension: dim_map[replacement.inner_dimension]}
            return self.original.get_value(new_members, new_dim_map, base_type)

        def get_non_zero(
            self,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, None | IndexGetter],
            type_type: dlt.TypeType,
        ) -> tuple[list[Operation], bool | SSAValue]:
            ops = []
            result = False
            full_type = type_type.add_members(members).add_dimensions(dim_map.keys())
            replacements = {
                r
                for r in self.layout.replacements
                if r.inner_member in full_type.all_member_attributes()
            }
            for replacement in replacements:
                r_type = (
                    full_type.select_member(replacement.inner_member)
                    .select_dimension(replacement.inner_dimension)
                    .add_dimension(replacement.outer_dimension)
                )
                new_members = members - {replacement.inner_member}
                r_type = r_type.select_members(new_members)
                new_dim_map = dict(dim_map)
                if replacement.inner_dimension in new_dim_map:
                    index_val = new_dim_map.pop(replacement.inner_dimension)
                    new_dim_map[replacement.outer_dimension] = index_val
                r_type = r_type.select_dimensions(new_dim_map.keys())
                r_ops, r_result = self.original.get_non_zero(
                    new_members, new_dim_map, r_type
                )

                if r_result is False:
                    ops.extend(r_ops)
                elif r_result is True:
                    for op in reversed(ops):
                        op.erase()
                    return [], True
                else:
                    assert isinstance(r_result, SSAValue)
                    bool_ops, result = _make_bool_ssa(result)
                    ops.extend(bool_ops)
                    ops.extend(r_ops)
                    or_op = arith.OrI(result, r_result)
                    ops.append(or_op)
                    result = or_op.result
            return ops, result

    def init_layout(
        self,
        layout: dlt.ArithReplaceLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:

        new_initialiser = self.ReplacementInitialiser(layout, initial_values)
        new_callback = self.ReplacementCallback(layout, init_callback)

        child_ops, iter_args_out = self.semantics.init_layout(
            layout.child,
            extent_resolver,
            input_ptr,
            new_initialiser,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
        )
        return child_ops, iter_args_out

    def dealloc_layout(
        self,
        layout: dlt.ArithDropLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        return self.semantics.dealloc_layout(layout.child, extent_resolver, input_ptr)

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:

        new_callback = self.ReplacementCallback(layout, callback)

        child_ops, iter_args_out, exited = self.semantics.linear_iterate(
            layout.child,
            extent_resolver,
            input_ptr,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
            reversed_direction,
        )
        return child_ops, iter_args_out, exited

    def make_sparse_loop_for(
        self,
        starting_layout: dlt.ArithReplaceLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        if starting_layout == ending_layout:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out

        overlap_dims = set(dim_mapping.keys()) & starting_layout.outer_dimensions()
        if len(overlap_dims) > 0:
            assert len(overlap_dims) == 1
            dim = overlap_dims.pop()
            replacement = starting_layout.replacement_for(dim)
            new_dim_mapping = dict(dim_mapping)

            idx_getter = new_dim_mapping.pop(replacement.outer_dimension)
            new_dim_mapping[replacement.inner_dimension] = idx_getter
            new_members = members | {replacement.inner_member}

            new_callback = self.ReplacementLoopCallback(starting_layout, callback)
            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                starting_layout.child,
                ending_layout,
                extent_resolver,
                input_ptr,
                new_callback,
                callback_args,
                new_members,
                new_dim_mapping,
                dims_to_loop,
            )
            return child_ops, iter_args_out
        overlap_dims = set(dims_to_loop) & starting_layout.outer_dimensions()
        if len(overlap_dims) > 0:
            assert len(overlap_dims) == 1
            dim = overlap_dims.pop()
            replacement = starting_layout.replacement_for(dim)
            new_dims_to_loop = (dims_to_loop - {replacement.outer_dimension}) | {
                replacement.inner_dimension
            }
            new_members = members | {replacement.inner_member}

            new_callback = self.ReplacementLoopCallback(starting_layout, callback)
            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                starting_layout.child,
                ending_layout,
                extent_resolver,
                input_ptr,
                new_callback,
                callback_args,
                new_members,
                dim_mapping,
                new_dims_to_loop,
            )
            return child_ops, iter_args_out

        callback_ops, iter_args_out = callback.callback(
            starting_layout,
            members,
            dim_mapping,
            dims_to_loop,
            extent_resolver,
            input_ptr,
            callback_args,
        )
        return callback_ops, iter_args_out


class IndexingSemantics(DirectLayoutNodeSemantics[dlt.IndexingLayoutAttr]):

    def _get_direct_parts(
        self,
        layout: dlt.IndexingLayoutAttr,
        extent_resolver: ExtentResolver,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        input_ptr: SSAValue,
    ) -> tuple[
        list[Operation],
        set[dlt.MemberAttr],
        dict[dlt.DimensionAttr, IndexGetter],
        SSAValue,
    ]:
        ops = []
        ptr_space = self.semantics.get_size(layout.indexedChild, extent_resolver).sum()
        direct_ptr_ops, direct_data_ptr = ptr_space.add_to_llvm_pointer(input_ptr)
        ops.extend(direct_ptr_ops)

        direct_content_type = layout.directChild.contents_type
        direct_members = direct_content_type.all_member_attributes().intersection(
            members
        )
        direct_content_all_dims = direct_content_type.all_dimension_attributes()
        direct_dim_mapping = {
            dim: val
            for dim, val in dim_mapping.items()
            if dim in direct_content_all_dims
        }
        return ops, direct_members, direct_dim_mapping, direct_data_ptr

    def get_size(
        self, layout: dlt.IndexingLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        return (
            self.semantics.get_size(layout.indexedChild, extent_resolver).sum()
            + self.semantics.get_size(layout.directChild, extent_resolver).sum()
        ).extend(0)

    def get_select_for(
        self,
        starting_layout: T,
        ending_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:

        # this is not allowed - because the data buffers might change...
        raise NotImplementedError

    def get_getter_for(
        self,
        starting_layout: dlt.IndexingLayoutAttr,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        ops = []

        direct_parts_ops, direct_members, direct_dim_mapping, direct_data_ptr = (
            self._get_direct_parts(
                starting_layout, extent_resolver, members, dim_mapping, input_ptr
            )
        )
        ops.extend(direct_parts_ops)
        direct_get_ops, index_range, index_range_found = self.semantics.get_getter_for(
            starting_layout.directChild,
            starting_layout.indexedChild.indexed_by(),
            set(direct_members),
            dict(direct_dim_mapping),
            extent_resolver,
            direct_data_ptr,
        )
        ops.extend(direct_get_ops)

        child_dim_mapping = {
            d: v for d, v in dim_mapping.items() if d not in direct_dim_mapping
        }
        child_get_ops, get_res, get_found = self.semantics.get_indexed_getter_for(
            starting_layout.indexedChild,
            get_type,
            members - direct_members,
            child_dim_mapping,
            extent_resolver,
            input_ptr,
            index_range,
            index_range_found,
        )
        ops.extend(child_get_ops)
        return ops, get_res, get_found

    def get_setter_for(
        self,
        starting_layout: dlt.IndexingLayoutAttr,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        ops = []
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)

        # 1) Get the members and dim_mapping used in the direct child
        direct_parts_ops, direct_members, direct_dim_mapping, direct_data_ptr = (
            self._get_direct_parts(
                starting_layout, extent_resolver, members, dim_mapping, input_ptr
            )
        )
        ops.extend(direct_parts_ops)

        # 2) Ask the direct child for the element's index range
        direct_get_ops, index_range, index_range_found = self.semantics.get_getter_for(
            starting_layout.directChild,
            starting_layout.indexedChild.indexed_by(),
            set(direct_members),
            dict(direct_dim_mapping),
            extent_resolver,
            direct_data_ptr,
        )
        ops.extend(direct_get_ops)
        bool_ops, index_range_found = _make_bool_ssa(index_range_found)
        ops.extend(bool_ops)

        child_dim_mapping = {
            d: v for d, v in dim_mapping.items() if d not in direct_dim_mapping
        }

        def ensure_space_func(
            fill_value_getter: FillValueGetter,
        ) -> tuple[list[Operation], SSAValue]:
            return self.semantics.ensure_space(
                starting_layout.directChild,
                starting_layout.indexedChild.indexed_by(),
                set(direct_members),
                dict(direct_dim_mapping),
                fill_value_getter,
                extent_resolver,
                direct_data_ptr,
            )

        def direct_iter_func(
            callback: Callback,
            is_last_element: bool | SSAValue,
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            return self.semantics.linear_iterate(
                starting_layout.directChild,
                extent_resolver,
                direct_data_ptr,
                callback,
                callback.initial_iter_args(),
                true_op.result,
                is_last_element,
            )

        child_ops = self.semantics.get_indexed_setter_for(
            starting_layout.indexedChild,
            set_val,
            members - direct_members,
            child_dim_mapping,
            extent_resolver,
            input_ptr,
            index_range,
            index_range_found,
            ensure_space_func,
            direct_iter_func,
            direct_members,
            direct_dim_mapping,
        )
        ops.extend(child_ops)
        return ops

    def ensure_space(
        self,
        layout: dlt.IndexingLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        ops = []
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)
        direct_parts_ops, direct_members, direct_dim_mapping, direct_data_ptr = (
            self._get_direct_parts(
                layout, extent_resolver, members, dim_mapping, input_ptr
            )
        )
        ops.extend(direct_parts_ops)

        direct_get_ops, index_range, index_range_found = self.semantics.get_getter_for(
            layout.directChild,
            layout.indexedChild.indexed_by(),
            set(direct_members),
            dict(direct_dim_mapping),
            extent_resolver,
            direct_data_ptr,
        )
        ops.extend(direct_get_ops)
        bool_ops, index_range_found = _make_bool_ssa(index_range_found)
        ops.extend(bool_ops)

        if_index_range_found_true = [scf.Yield(index_range)]
        if_index_range_found_false = []

        def current_fill_value_getter() -> tuple[list[Operation], SSAValue]:
            return _get_packed_zero_for_accepted_type(dlt.IndexRangeType())

        ensure_space_ops, new_index_range = self.semantics.ensure_space(
            layout.directChild,
            layout.indexedChild.indexed_by(),
            set(direct_members),
            dict(direct_dim_mapping),
            current_fill_value_getter,
            extent_resolver,
            direct_data_ptr,
        )
        if_index_range_found_false.extend(ensure_space_ops)

        if_index_range_found_false.append(scf.Yield(new_index_range))
        if_index_range_found_op = scf.If(
            index_range_found,
            [dlt.IndexRangeType()],
            if_index_range_found_true,
            if_index_range_found_false,
        )
        # if_index_range_found_op.attributes["debug"] = StringAttr(f"Indexing_ensure")
        ops.append(if_index_range_found_op)
        true_index_range = if_index_range_found_op.output[0]

        child_dim_mapping = {
            d: v for d, v in dim_mapping.items() if d not in direct_dim_mapping
        }

        def direct_iter_func(
            callback: Callback, is_last_element: bool | SSAValue
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            return self.semantics.linear_iterate(
                layout.directChild,
                extent_resolver,
                direct_data_ptr,
                callback,
                callback.initial_iter_args(),
                true_op.result,
                is_last_element,
            )

        child_ops, index_result = self.semantics.ensure_indexed_space(
            layout.indexedChild,
            base_type,
            members - direct_members,
            child_dim_mapping,
            fill_value_getter,
            extent_resolver,
            input_ptr,
            true_index_range,
            direct_iter_func,
            direct_members,
            direct_dim_mapping,
        )
        ops.extend(child_ops)
        return ops, index_result

    def init_layout(
        self,
        layout: dlt.IndexingLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:

        ops: list[Operation] = []
        true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
        # true_const.attributes["debug"] = builtin.StringAttr(
        #     "Indexing Semantics Init Layout start"
        # )
        ops.append(true_const)
        ptr_space = self.semantics.get_size(layout.indexedChild, extent_resolver).sum()
        ptr_ops, direct_data_ptr = ptr_space.add_to_llvm_pointer(input_ptr)
        ops.extend(ptr_ops)

        def direct_init_func(
            callback_arg: Callback,
            is_last_element_arg: bool | SSAValue,
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            direct_init_ops, total_size_iter_args = self.semantics.init_layout(
                layout.directChild,
                extent_resolver,
                direct_data_ptr,
                initial_values,
                callback_arg,
                callback_arg.initial_iter_args(),
                true_const.result,
                is_last_element_arg,
            )
            return direct_init_ops, total_size_iter_args, True

        def direct_iter_func(
            callback_arg: Callback,
            is_last_element_arg: bool | SSAValue,
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            return self.semantics.linear_iterate(
                layout.directChild,
                extent_resolver,
                direct_data_ptr,
                callback_arg,
                callback_arg.initial_iter_args(),
                true_const.result,
                is_last_element_arg,
            )

        child_ops, child_iter_args = self.semantics.init_indexed_layout(
            layout.indexedChild,
            extent_resolver,
            input_ptr,
            initial_values,
            init_callback,
            callback_args,
            is_last_element,
            direct_init_func,
            direct_iter_func,
        )
        ops.extend(child_ops)
        return ops, child_iter_args

    def dealloc_layout(
        self,
        layout: dlt.IndexingLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:

        ops = []
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)

        ptr_space = self.semantics.get_size(layout.indexedChild, extent_resolver).sum()
        ptr_ops, direct_data_ptr = ptr_space.add_to_llvm_pointer(input_ptr)
        ops.extend(ptr_ops)

        def direct_iter_func(
            callback: Callback,
            is_last_element: bool | SSAValue,
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            return self.semantics.linear_iterate(
                layout.directChild,
                extent_resolver,
                direct_data_ptr,
                callback,
                callback.initial_iter_args(),
                true_op.result,
                is_last_element,
            )

        child_ops = self.semantics.dealloc_indexed_layout(
            layout.indexedChild, extent_resolver, input_ptr, direct_iter_func
        )
        ops.extend(child_ops)

        dealloc_direct_child = self.semantics.dealloc_layout(
            layout.directChild, extent_resolver, direct_data_ptr
        )
        ops.extend(dealloc_direct_child)

        return ops

    def linear_iterate(
        self,
        layout: dlt.IndexingLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        indexed_side = self.semantics.get_size(
            layout.indexedChild, extent_resolver
        ).sum()
        ops, direct_data_ptr = indexed_side.add_to_llvm_pointer(input_ptr)
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)

        def direct_iter_func(
            callback_arg: Callback,
            is_last_element_arg: bool | SSAValue,
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            return self.semantics.linear_iterate(
                layout.directChild,
                extent_resolver,
                direct_data_ptr,
                callback_arg,
                callback_arg.initial_iter_args(),
                true_op.result,
                is_last_element_arg,
                reversed_direction=reversed_direction,
            )

        child_ops, child_iter_args, child_exited_early = (
            self.semantics.linear_iterate_indexed(
                layout.indexedChild,
                extent_resolver,
                input_ptr,
                callback,
                callback_args,
                is_last_element,
                direct_iter_func,
                reversed_direction,
            )
        )
        ops.extend(child_ops)
        return ops, child_iter_args, child_exited_early

    def make_sparse_loop_for(
        self,
        starting_layout: dlt.IndexingLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], list[SSAValue]]:
        if starting_layout == ending_layout:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out

        dir_type = starting_layout.directChild.contents_type
        direct_members = dir_type.all_member_attributes() & members
        direct_dims = dir_type.all_dimension_attributes() & (
            dims_to_loop | set(dim_mapping.keys())
        )
        dir_select_type = dlt.TypeType(
            [(direct_members, direct_dims, starting_layout.indexedChild.indexed_by())]
        )
        if (SetAttr([]), SetAttr([])) in dir_type.has_selectable_type(dir_select_type):
            ops = []
            ptr_space = self.semantics.get_size(
                starting_layout.indexedChild, extent_resolver
            ).sum()
            direct_ptr_ops, direct_data_ptr = ptr_space.add_to_llvm_pointer(input_ptr)
            ops.extend(direct_ptr_ops)

            direct_dim_mapping = {
                d: dim_mapping[d] for d in direct_dims if d in dim_mapping
            }
            direct_dims_to_loop = {d for d in direct_dims if d in dims_to_loop}

            direct_terminal = Manipulator.reduce_to_terminal(
                starting_layout.directChild,
                direct_members,
                direct_dims,
                starting_layout.indexedChild.indexed_by(),
            )
            assert direct_terminal is not None

            callback_setup_ops, new_callback = (
                self.semantics.make_sparse_loop_callback_for(
                    starting_layout.indexedChild,
                    ending_layout,
                    extent_resolver,
                    input_ptr,
                    callback,
                    callback_args,
                    members - direct_members,
                    {
                        d: g
                        for d, g in dim_mapping.items()
                        if d not in direct_dim_mapping
                    },
                    dims_to_loop - direct_dims_to_loop,
                )
            )
            ops.extend(callback_setup_ops)

            child_ops, iter_args_out = self.semantics.make_sparse_loop_for(
                starting_layout.directChild,
                direct_terminal,
                extent_resolver,
                direct_data_ptr,
                new_callback,
                callback_args,
                direct_members,
                direct_dim_mapping,
                direct_dims_to_loop,
            )
            return ops + child_ops, iter_args_out
        else:
            callback_ops, iter_args_out = callback.callback(
                starting_layout,
                members,
                dim_mapping,
                dims_to_loop,
                extent_resolver,
                input_ptr,
                callback_args,
            )
            return callback_ops, iter_args_out


class COOSemantics(IndexedLayoutNodeSemantics[T], abc.ABC, typing.Generic[T]):

    def _get_fill_value_getter(
        self,
        indexed_child_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        parent_fill_value_getter: FillValueGetter,
        base_type: dlt.AcceptedTypes,
        data_buffer_ptr: SSAValue,
        data_elem_size: SSAValue,
        index: SSAValue,
    ) -> FillValueGetter:

        def getter() -> tuple[list[Operation], SSAValue]:
            ops = []

            true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
            false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
            zero_op = arith.Constant(IntegerAttr(0, IndexType()))
            one_op = arith.Constant(IntegerAttr(1, IndexType()))
            ops.extend([true_op, false_op, zero_op, one_op])

            zero_idx_range_ops, zero_idx_range = _get_packed_zero_for_accepted_type(
                dlt.IndexRangeType()
            )
            ops.extend(zero_idx_range_ops)

            get_index_range_callback = GetFirstValueCallback(
                self.semantics, zero_idx_range
            )

            index_sub_one_op = arith.Subi(index, one_op.result)
            ops.append(index_sub_one_op)
            start_idx = index_sub_one_op.result
            end_idx = zero_op.result
            exit_now = false_op.result
            iter_arg = zero_idx_range

            condition_block = Block()
            c_index = condition_block.insert_arg(IndexType(), 0)
            c_exit = condition_block.insert_arg(IntegerType(1), 1)
            c_out_range = condition_block.insert_arg(dlt.IndexRangeType(), 2)
            condition_block.add_ops(
                [
                    cmp1 := arith.Cmpi(c_index, end_idx, "sge"),
                    cmp2 := arith.Cmpi(false_op.result, c_exit, "eq"),
                    cmp := arith.AndI(cmp1.result, cmp2.result),
                    scf.Condition(cmp, c_index, c_exit, c_out_range),
                ]
            )
            while_block = Block()
            w_index = while_block.insert_arg(IndexType(), 0)
            while_block.insert_arg(IntegerType(1), 1)  # w_exit
            w_index_range = while_block.insert_arg(dlt.IndexRangeType(), 2)

            elem_ops, f_elem_ptr = (
                NumericResult.from_ssa(data_elem_size) * NumericResult.from_ssa(w_index)
            ).add_to_llvm_pointer(data_buffer_ptr)
            while_block.add_ops(elem_ops)

            w_lin_iter_ops, w_callback_results, w_exited_early = (
                self.semantics.linear_iterate(
                    indexed_child_layout,
                    extent_resolver,
                    f_elem_ptr,
                    get_index_range_callback,
                    [w_index_range],
                    False,
                    False,
                    reversed_direction=True,
                )
            )
            while_block.add_ops(w_lin_iter_ops)
            bool_ops, w_exited_early = _make_bool_ssa(w_exited_early)
            while_block.add_ops(bool_ops)

            w_new_index_op = arith.Subi(w_index, one_op)
            while_block.add_op(w_new_index_op)
            while_block.add_op(
                scf.Yield(w_new_index_op.result, w_exited_early, w_callback_results[0])
            )

            while_op = scf.While(
                [start_idx, exit_now, iter_arg],
                [IndexType(), IntegerType(1), dlt.IndexRangeType()],
                [condition_block],
                [while_block],
            )
            ops.append(while_op)
            is_index_range_found = while_op.res[1]
            found_index_range = while_op.res[2]
            unpack_ops, _, fill_val = _extract_indices_from_index_range(
                found_index_range
            )
            ops.extend(unpack_ops)
            pack_ops, fill_value = _pack_indices_in_index_range(fill_val, fill_val)
            ops.extend(pack_ops)

            if_escalate, escalated_value = parent_fill_value_getter()
            if_escalate.append(scf.Yield(escalated_value))
            if_op = scf.If(
                is_index_range_found, [base_type], [scf.Yield(fill_value)], if_escalate
            )
            ops.append(if_op)

            return ops, if_op.output[0]

        return getter

    class SparseIndexWritingCallback(Callback):

        def __init__(
            self,
            indexed_child_typetype: dlt.TypeType,
            init_total: SSAValue,
            sparse_initial_values: Initialiser,
            sparse_dims: list[dlt.DimensionAttr],
        ):
            self.indexed_child_typetype = indexed_child_typetype
            self.sparse_initial_values = sparse_initial_values
            self.sparse_dims = sparse_dims
            super().__init__([init_total])

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, SSAValue],
            extent_resolver: ExtentResolver,
            base_type: dlt.AcceptedTypes,
            ptr: SSAValue,
            has_extra_space: SSAValue,
            is_last_element: bool | SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            if not isinstance(base_type, dlt.IndexRangeType):
                return [], iter_args, False
            # This callback keeps track of a running total of non-zeros, and sets the index value to that running total
            # as we go
            running_total = iter_args[0]
            ops: list[Operation] = []

            # first store the current running total as the index
            idx_type = base_type
            if idx_type == dlt.IndexRangeType():
                idx_type = i64 # dlt.indexRange is [i64,i64]
            assert isinstance(idx_type, IntegerType)
            convert_running_total_ops, running_total_i = index_convert(running_total, idx_type)
            ops.extend(convert_running_total_ops)
            store_idx_op = llvm.StoreOp(running_total_i, ptr)
            # store_idx_op.attributes["debug"] = builtin.StringAttr(
            #     "                 Debug Sparse index writting Callback"
            # )
            ops.append(store_idx_op)

            derived_initialiser = DerivedInitialiser(
                self.sparse_initial_values, members, dim_map
            )

            iter_ops, iter_op = _make_iterate(
                self.sparse_dims, extent_resolver, [], [running_total]
            )
            ops.extend(iter_ops)

            block = iter_op.body.block
            block.erase_op(block.last_op)  # remove yield as we will add our own
            block_running_total = iter_op.get_block_arg_for_iter_arg_idx(0)

            get_non_zeros, found_non_zeros = derived_initialiser.get_non_zero(
                set(),
                {
                    dim: ArgIndexGetter(arg)
                    for dim, arg in zip(
                        self.sparse_dims, iter_op.get_block_args_for_extent_args()
                    )
                },
                self.indexed_child_typetype,
            )
            block.add_ops(get_non_zeros)
            bool_ssa_ops, has_non_zero = _make_bool_ssa(found_non_zeros)
            block.add_ops(bool_ssa_ops)

            # if there is a non-zero in any of the source inner elems then we add 1 to the running total
            if_op = scf.If(
                has_non_zero,
                [IndexType()],
                [
                    one := arith.Constant(IntegerAttr(1, IndexType())),
                    add := arith.Addi(block_running_total, one.result),
                    scf.Yield(add.result),
                ],
                [scf.Yield(block_running_total)],
            )
            block.add_op(if_op)
            block.add_op(dlt.IterateYieldOp(if_op.output[0]))

            new_running_total = iter_op.res[0]
            if_has_extra_space_true = []
            extra_ptr_ops, extra_ptr = (
                _get_accepted_type_size(base_type).keep(1).add_to_llvm_pointer(ptr)
            )
            if_has_extra_space_true.extend(extra_ptr_ops)
            convert_extra_running_total_ops, new_running_total_i = index_convert(new_running_total, idx_type)
            # if there is extra space, this can only be because
            ops.extend(convert_extra_running_total_ops)
            store_idx_op = llvm.StoreOp(new_running_total_i, extra_ptr)
            # store_idx_op.attributes["debug"] = builtin.StringAttr(
            #     "                 Debug Sparse index writing Callback for extra space"
            # )
            if_has_extra_space_true.append(store_idx_op)
            if_has_extra_space_true.append(scf.Yield())
            ops.append(scf.If(has_extra_space, [], if_has_extra_space_true))

            return ops, [new_running_total], False

    class WriteEmptyRangeInitialiser(Initialiser):

        def __init__(
            self,
            fill_value: SSAValue,  # the index value at the point the index was found.
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, IndexGetter],
        ):
            self.fill_value = fill_value
            self.members = members
            self.dim_map = dim_map
            assert isinstance(self.fill_value.type, dlt.IndexRangeType)
            super().__init__()

        def get_value(
            self,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, IndexGetter],
            base_type: dlt.AcceptedTypes,
        ) -> tuple[list[Operation], None | SSAValue, bool | SSAValue]:
            if self.fill_value.type != base_type:
                return [], None, False
            else:
                return [], self.fill_value, True

        def get_non_zero(
            self,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, None | IndexGetter],
            type_type: dlt.TypeType,
        ) -> tuple[list[Operation], bool | SSAValue]:
            full_type = type_type.add_members(members).add_dimensions(dim_map.keys())
            if (
                dlt.ElementAttr(self.members, self.dim_map.keys(), dlt.IndexRangeType())
                not in full_type.elements
            ):
                return [], False
            ops = []
            true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.append(true_op)
            found = true_op.result
            for dim in dim_map:
                if dim_map[dim] is not None:
                    other_get_ops, (other_index,) = dim_map[dim].get().output()
                    ops.extend(other_get_ops)
                    self_get_ops, (self_index,) = self.dim_map[dim].get().output()
                    ops.extend(self_get_ops)
                    compare = arith.Cmpi(other_index, self_index, "eq")
                    ops.append(compare)
                    and_op = arith.AndI(found, compare.result)
                    ops.append(and_op)
                    found = and_op.result
            return ops, found

    class DeallocCallback(Callback):
        def __init__(
            self,
            semantics: SemanticsMapper,
            coo_layout: dlt.UnpackedCOOLayoutAttr | dlt.SeparatedCOOLayoutAttr,
            extent_resolver: ExtentResolver,
            data_buffer_ptr: SSAValue,
            data_elem_size: SSAValue,
        ):
            self.semantics = semantics
            self.coo_layout = coo_layout
            self.extent_resolver = extent_resolver
            self.data_buffer_ptr = data_buffer_ptr
            self.data_elem_size = data_elem_size
            super().__init__([])

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, SSAValue],
            extent_resolver: ExtentResolver,
            base_type: dlt.AcceptedTypes,
            ptr: SSAValue,
            has_extra_space: SSAValue,
            is_last_element: bool | SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            ops = []

            getter_ops, index_range, index_range_found = self.semantics.get_getter_for(
                terminal_layout, dlt.IndexRangeType(), set(), {}, extent_resolver, ptr
            )
            ops.extend(getter_ops)
            extract_ops, start, end = _extract_indices_from_index_range(index_range, as_index=True)
            ops.extend(extract_ops)
            ops.append(one_op := arith.Constant(IntegerAttr(1, IndexType())))

            # Loop over sparse buffers
            start_idx, end_idx, step = start, end, one_op.result

            block = Block()
            index = block.insert_arg(IndexType(), 0)

            data_buffer_ops, current_data_buffer_ptr = (
                NumericResult.from_mixed([], index)
                * NumericResult.from_mixed([], self.data_elem_size)
            ).add_to_llvm_pointer(self.data_buffer_ptr)
            block.add_ops(data_buffer_ops)

            child_dealloc_ops = self.semantics.dealloc_layout(
                self.coo_layout.child,
                extent_resolver,
                current_data_buffer_ptr,
            )
            block.add_ops(child_dealloc_ops)

            inc_index_op = arith.Addi(index, step)
            block.add_op(inc_index_op)

            block.add_op(scf.Yield())

            for_loop = scf.For(start, end, step, [], block)
            ops.append(for_loop)

            return ops, [], False

    class SparseIndexIncrementCallback(Callback):

        def __init__(
            self,
            direct_dim_map: dict[
                dlt.DimensionAttr, IndexGetter
            ],  # the indices of the new element
            direct_members: set[dlt.MemberAttr],  # the members of the new element
            index_found: SSAValue,  # bool; set true once we're at the index of the new element
            last_index: SSAValue,
            # updated with the last index so that by the end it is equal to the top index and
            # so is number of non-zeros
            found_index: SSAValue,  # the index value at the point the index was found.
        ):
            self.direct_dim_map = direct_dim_map
            self.direct_members = direct_members

            super().__init__([index_found, last_index, found_index])

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, SSAValue],
            extent_resolver: ExtentResolver,
            base_type: dlt.AcceptedTypes,
            ptr: SSAValue,
            has_extra_space: SSAValue,
            is_last_element: bool | SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            if not isinstance(base_type, dlt.IndexRangeType):
                return [], iter_args, False
            if members != self.direct_members:
                return [], iter_args, False
            ops = []
            zero_op = arith.Constant(IntegerAttr(0, IndexType()))
            one_i64_op = arith.Constant(IntegerAttr(1, i64))
            false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
            true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.extend([zero_op, one_i64_op, false_op, true_op])

            if is_last_element is True:
                is_last_element = true_op.result
            elif is_last_element is False:
                is_last_element = false_op.result

            is_index_found = iter_args[0]
            # last_index = iter_args[1]  This is re-written each time
            found_index = iter_args[2]

            # If index_found:
            if_found_true = []
            # then increment the stored index by 1
            load_op = llvm.LoadOp(ptr, i64)
            if_found_true.append(load_op)
            add_op = arith.Addi(load_op.dereferenced_value, one_i64_op.result)
            if_found_true.append(add_op)
            store_op = llvm.StoreOp(add_op.result, ptr)
            if_found_true.append(store_op)
            if_found_true.append(scf.Yield())

            if_found_if_op = scf.If(is_index_found, [], if_found_true)
            ops.append(if_found_if_op)

            # now check if this iteration should set is_found
            element_found_op = arith.Cmpi(is_index_found, false_op.result, "eq")
            ops.append(element_found_op)
            element_found = element_found_op.result
            # We start by checking if is_found_index is true, because if it is, then we know it can't be found again.
            # hopefully this means an llvm optimisation can decide not to do the rest of the checks if they might
            # take too much time
            for dim, idx_getter in self.direct_dim_map.items():
                idx_to_find_ops, (idx_to_find,) = idx_getter.get().output()
                ops.extend(idx_to_find_ops)
                idx_to_check = dim_map[dim]
                idx_cmp_op = arith.Cmpi(idx_to_check, idx_to_find, "eq")
                ops.append(idx_cmp_op)
                element_found_op = arith.AndI(element_found, idx_cmp_op.result)
                ops.append(element_found_op)
                element_found = element_found_op.result
            # if element_found, then this iteration of the callback has found the sparse index for element we care about

            # If element_found then we need to return the right iter_args
            if_element_found_true = []  # yield (found_index)
            load_op = llvm.LoadOp(ptr, i64)
            if_element_found_true.append(load_op)
            convert_new_found_index_ops, new_found_index = index_convert(load_op.dereferenced_value, IndexType())
            if_element_found_true.extend(convert_new_found_index_ops)
            if_element_found_true.append(scf.Yield(new_found_index))
            if_element_found_op = scf.If(
                element_found,
                [IndexType()],
                if_element_found_true,
                [scf.Yield(found_index)],
            )
            ops.append(if_element_found_op)
            new_is_index_found_op = arith.OrI(element_found, is_index_found)
            ops.append(new_is_index_found_op)
            new_is_index_found = new_is_index_found_op.result

            # If (extra_space and found) or is_last, then we must also increment the range end index
            extra_space_op = arith.AndI(has_extra_space, new_is_index_found)
            ops.append(extra_space_op)
            increment_extra_op = arith.OrI(is_last_element, extra_space_op)
            ops.append(increment_extra_op)
            if_increment_extra_true = []
            ptr_ops, extra_ptr = (
                _get_accepted_type_size(base_type).keep(1).add_to_llvm_pointer(ptr)
            )
            if_increment_extra_true.extend(ptr_ops)
            extra_load_op = llvm.LoadOp(extra_ptr, i64)
            if_increment_extra_true.append(extra_load_op)
            extra_add_op = arith.Addi(extra_load_op.dereferenced_value, one_i64_op.result)
            if_increment_extra_true.append(extra_add_op)
            extra_store_op = llvm.StoreOp(extra_add_op.result, extra_ptr)
            if_increment_extra_true.append(extra_store_op)
            convert_last_idx_ops, new_last_index = index_convert(extra_add_op.result, IndexType())
            if_increment_extra_true.extend(convert_last_idx_ops)
            last_index_op = arith.Select(
                is_last_element, new_last_index, zero_op.result
            )
            if_increment_extra_true.append(last_index_op)
            if_increment_extra_true.append(scf.Yield(last_index_op.result))

            if_increment_extra_if_op = scf.If(
                increment_extra_op.result,
                [IndexType()],
                if_increment_extra_true,
                [scf.Yield(zero_op.result)],
            )
            ops.append(if_increment_extra_if_op)
            new_last_index = if_increment_extra_if_op.results[0]

            return (
                ops,
                [
                    new_is_index_found,
                    new_last_index,
                    if_element_found_op.results[0],
                ],
                False,
            )


class UnpackCOOSemantics(COOSemantics[dlt.UnpackedCOOLayoutAttr]):

    def get_size(
        self, layout: dlt.UnpackedCOOLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        # [index_buffer, data_buffer]
        if layout.is_buffered():
            return ((_get_ptr_size().sum() * 2) + _get_accepted_type_size(i64).sum()).extend(0)
        else:
            return ((_get_ptr_size().sum() * 2)).extend(0)

    def get_getter_for(
        self,
        starting_layout: dlt.UnpackedCOOLayoutAttr,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index_range: SSAValue,
        index_found: bool | SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        ops = []

        extract_ops, start, end = _extract_indices_from_index_range(index_range, as_index=True)
        ops.extend(extract_ops)

        idx_buffer_ptr_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        ops.append(idx_buffer_ptr_op)

        while_op, index_is_found, index = self.loop_to_find_index(
            starting_layout,
            start,
            end,
            idx_buffer_ptr_op.dereferenced_value,
            dim_mapping,
        )
        ops.append(while_op)
        # If index is found we can use it, else we return an appropriate zero element
        if_found_true = []

        ptr_size = _get_ptr_size().sum()
        data_buffer_ptr_ops, data_buffer_ptr_ptr = ptr_size.add_to_llvm_pointer(
            input_ptr
        )
        if_found_true.extend(data_buffer_ptr_ops)
        data_buffer_ptr_op = llvm.LoadOp(
            data_buffer_ptr_ptr, llvm.LLVMPointerType.opaque()
        )
        if_found_true.append(data_buffer_ptr_op)
        data_ptr_ops, child_elem_ptr = (
            NumericResult.from_ssa(index)
            * self.semantics.get_size(starting_layout.child, extent_resolver).keep(1)
        ).add_to_llvm_pointer(data_buffer_ptr_op.dereferenced_value)
        if_found_true.extend(data_ptr_ops)

        child_dim_mapping = {
            dim: val
            for dim, val in dim_mapping.items()
            if dim not in starting_layout.dimensions
        }
        child_members = members
        child_ops, child_res, child_found = self.semantics.get_getter_for(
            starting_layout.child,
            get_type,
            child_members,
            child_dim_mapping,
            extent_resolver,
            child_elem_ptr,
        )
        if_found_true.extend(child_ops)
        bool_ops, child_found = _make_bool_ssa(child_found)
        if_found_true.extend(bool_ops)

        if_found_true.append(scf.Yield(child_res, child_found))

        if_found_false = []

        zero_result_ops, zero_result = _get_packed_zero_for_accepted_type(get_type)
        if_found_false.extend(zero_result_ops)

        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        if_found_false.append(false_op)
        if_found_false.append(scf.Yield(zero_result, false_op.result))

        if_found_op = scf.If(
            index_is_found, [get_type, IntegerType(1)], if_found_true, if_found_false
        )
        ops.append(if_found_op)
        return ops, if_found_op.output[0], if_found_op.output[1]

    def get_setter_for(
        self,
        starting_layout: dlt.UnpackedCOOLayoutAttr,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index_range: SSAValue,
        index_range_found: bool | SSAValue,
        ensure_space_func: EnsureSpaceFunc,
        direct_iterate_func: LinearIterCallbackFunc,
        direct_members: set[dlt.MemberAttr],
        direct_dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> list[Operation]:

        # get the members and dim_mapping that will be needed by the recursive child get_setter_for
        child_dim_mapping = {
            dim: val
            for dim, val in dim_mapping.items()
            if dim not in starting_layout.dimensions
        }
        child_members = members

        ops = []
        zero_cmp_ops, is_non_zero = _compare_is_non_zero(set_val)
        ops.extend(zero_cmp_ops)
        index_range_found_ops, index_range_found = _make_bool_ssa(index_range_found)
        ops.extend(index_range_found_ops)
        must_start_set_cond_op = arith.OrI(index_range_found, is_non_zero)
        ops.append(must_start_set_cond_op)

        if_must_start_set_ops = []

        # 3) if the direct child doesn't have a range, we must ask it to make one.
        if_index_range_found_true = [scf.Yield(index_range)]
        if_index_range_found_false = []

        def current_fill_value_getter() -> tuple[list[Operation], SSAValue]:
            f_ops, zero_idx_range = _get_packed_zero_for_accepted_type(
                dlt.IndexRangeType()
            )
            return f_ops, zero_idx_range

        ensure_space_ops, new_index_range = ensure_space_func(current_fill_value_getter)

        if_index_range_found_false.extend(ensure_space_ops)
        if_index_range_found_false.append(scf.Yield(new_index_range))

        if_index_range_found_op = scf.If(
            index_range_found,
            [dlt.IndexRangeType()],
            if_index_range_found_true,
            if_index_range_found_false,
        )
        if_must_start_set_ops.append(if_index_range_found_op)
        true_index_range = if_index_range_found_op.output[0]

        # 4) with the index-range, we must look for the matching element in the sparse index buffer
        extract_ops, start, end = _extract_indices_from_index_range(true_index_range, as_index=True)
        if_must_start_set_ops.extend(extract_ops)

        idx_buffer_ptr_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        if_must_start_set_ops.append(idx_buffer_ptr_op)
        idx_buffer_ptr = idx_buffer_ptr_op.dereferenced_value

        while_op, index_is_found, index = self.loop_to_find_index(
            starting_layout,
            start,
            end,
            idx_buffer_ptr,
            dim_mapping,
        )
        if_must_start_set_ops.append(while_op)
        # index_is_found iff we find an index tuple that matches the sparse dimension values from dim_mapping index
        # will be the index offset in the index-buffer (and data-buffer) of the match if there is one, else it will
        # be the index it was expected to be (i.e. the index where the loop stopped because either the range ran out,
        # or the index tuples got larger than the values in dim_mapping)

        must_do_set_cond_op = arith.OrI(index_is_found, is_non_zero)
        if_must_start_set_ops.append(must_do_set_cond_op)
        if_must_do_set_ops = []

        # 5) Now get the data buffer ptr
        ptr_size = _get_ptr_size().sum()
        data_buffer_ptr_ops, data_buffer_ptr_ptr = ptr_size.add_to_llvm_pointer(
            input_ptr
        )
        if_must_do_set_ops.extend(data_buffer_ptr_ops)
        data_buffer_ptr_op = llvm.LoadOp(
            data_buffer_ptr_ptr, llvm.LLVMPointerType.opaque()
        )
        if_must_do_set_ops.append(data_buffer_ptr_op)
        data_buffer_ptr = data_buffer_ptr_op.dereferenced_value

        # 6) If index is found we can get the ptr for the child, else we need to reallocate and move lots of indices
        # around
        if_found_true = []

        data_ptr_ops, child_elem_ptr = (
            NumericResult.from_ssa(index)
            * self.semantics.get_size(starting_layout.child, extent_resolver).keep(1)
        ).add_to_llvm_pointer(data_buffer_ptr)
        if_found_true.extend(data_ptr_ops)

        # and finally we can set the value in this child element
        child_setter_ops = self.semantics.get_setter_for(
            starting_layout.child,
            set_val,
            set(child_members),
            dict(child_dim_mapping),
            extent_resolver,
            child_elem_ptr,
        )
        if_found_true.extend(child_setter_ops)

        if_found_true.append(scf.Yield())

        # If the value is not found we need to increment the indices of everything larger than this in the direct
        # data, then allocate new buffers with extra size for the new element, and mem copy the data across leaving
        # room for the new element. Then we init the new element and can continue setting from there.
        if_found_false = []

        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        if_found_false.extend([true_op, false_op, zero_op])

        increment_indices_callback = self.SparseIndexIncrementCallback(
            dict(direct_dim_mapping),
            set(direct_members),
            false_op.result,
            zero_op.result,
            zero_op.result,
        )

        increment_ops, increment_results, _increment_exited = direct_iterate_func(
            increment_indices_callback, True
        )

        if_found_false.extend(increment_ops)
        # index_found = increment_results[0] - unused as this is for iter-callback communication - not a result
        last_index = increment_results[1]
        # last_index will be the number of non-zero elements now in the sparse buffer
        # found_index = increment_results[2]
        # found_index will be the index of the element we care about - the start of the range in the direct child
        # This is not actually required and should be removed

        if starting_layout.is_buffered():
            ptr_size = _get_ptr_size().sum()
            buffer_size_ptr_ops, buffer_size_ptr = ptr_size.add_to_llvm_pointer(
                data_buffer_ptr_ptr
            )
            if_found_false.extend(buffer_size_ptr_ops)
            buffer_size_op = llvm.LoadOp(buffer_size_ptr, i64)
            if_found_false.append(buffer_size_op)
            buffer_size_i64 = buffer_size_op.dereferenced_value
            convert_buffer_size_ops, buffer_size = index_convert(buffer_size_i64, IndexType())
            if_found_false.extend(convert_buffer_size_ops)

            # if_found_false.append(printf.PrintFormatOp("last_index {}   buffer_size {}", last_index, buffer_size))

            cmp_buffer_size_op = arith.Cmpi(last_index, buffer_size, "ugt")
            if_found_false.append(cmp_buffer_size_op)

            # if we need new buffers - handle it
            if_needs_new_buffers = []
            buffer_scaler = starting_layout.buffer_scaler.data
            buffer_scaler_const = arith.Constant(
                IntegerAttr(abs(buffer_scaler), IndexType())
            )
            if_needs_new_buffers.append(buffer_scaler_const)
            if buffer_scaler > 0:
                # multiply
                mul_op = arith.Muli(buffer_size, buffer_scaler_const)
                if_needs_new_buffers.append(mul_op)
                new_size_candidate = mul_op.result
                # if_needs_new_buffers.append(printf.PrintFormatOp("candidate {}   last index {}", new_size_candidate, last_index))
            else:
                # add
                add_op = arith.Addi(buffer_size, buffer_scaler_const)
                if_needs_new_buffers.append(add_op)
                new_size_candidate = add_op.result
                # if_needs_new_buffers.append(
                #     printf.PrintFormatOp("candidate {}   last index {}", new_size_candidate, last_index))
            new_size_op = arith.MaxUI(new_size_candidate, last_index)
            if_needs_new_buffers.append(new_size_op)
            new_size = new_size_op.result
            (
                do_new_buffers_ops,
                new_idx_buffer_ptr,
                new_data_buffer_ptr,
                idx_elem_ptr,
                data_elem_ptr,
            ) = self.do_new_buffers(
                starting_layout,
                extent_resolver,
                new_size,
                idx_buffer_ptr,
                data_buffer_ptr,
                buffer_size,
                index,
            )
            if_needs_new_buffers.extend(do_new_buffers_ops)

            initialiser = SingletonInitialiser(
                set_val, set(child_members), dict(child_dim_mapping)
            )
            set_up_child_ops = self.set_up_new_child(
                starting_layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                idx_elem_ptr,
                data_elem_ptr,
            )
            if_needs_new_buffers.extend(set_up_child_ops)
            # if_needs_new_buffers.append(
            #     printf.PrintFormatOp("4613 new idx {}", new_idx_buffer_ptr))
            if_needs_new_buffers.append(llvm.StoreOp(new_idx_buffer_ptr, input_ptr))
            # if_needs_new_buffers.append(
            #     printf.PrintFormatOp("4616 new data {}", new_data_buffer_ptr))
            if_needs_new_buffers.append(
                llvm.StoreOp(new_data_buffer_ptr, data_buffer_ptr_ptr)
            )
            convert_new_size_ops, new_size_i64 = index_convert(new_size, i64)
            if_needs_new_buffers.extend(convert_new_size_ops)
            if_needs_new_buffers.append(llvm.StoreOp(new_size_i64, buffer_size_ptr))
            if_needs_new_buffers.append(scf.Yield())

            # alternatively we don't need new buffers - just need to move the data in the buffers we have
            if_needs_moved_buffers = []
            calc_old_elems_ops, (elems_to_copy,) = (
                NumericResult.from_ssa(last_index) - 1
            ).output()
            if_needs_moved_buffers.extend(calc_old_elems_ops)
            (
                do_move_buffers_ops,
                idx_elem_ptr,
                data_elem_ptr,
            ) = self.do_move_buffers(
                starting_layout,
                extent_resolver,
                buffer_size,
                last_index,
                elems_to_copy,
                idx_buffer_ptr,
                data_buffer_ptr,
                index,
            )
            if_needs_moved_buffers.extend(do_move_buffers_ops)
            initialiser = SingletonInitialiser(
                set_val, set(child_members), dict(child_dim_mapping)
            )
            set_up_child_ops = self.set_up_new_child(
                starting_layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                idx_elem_ptr,
                data_elem_ptr,
            )
            if_needs_moved_buffers.extend(set_up_child_ops)
            if_needs_moved_buffers.append(scf.Yield())

            if_new_buffers_op = scf.If(
                cmp_buffer_size_op.result,
                [],
                if_needs_new_buffers,
                if_needs_moved_buffers,
            )
            if_found_false.append(if_new_buffers_op)
        else:
            # if it's not buffered then we know we just need to always do new buffers at the size of last_index
            # and we know the old buffer size was just one element less than the new last_index
            new_size = last_index
            calc_old_size_ops, (old_buffer_size,) = (
                NumericResult.from_ssa(last_index) - 1
            ).output()
            if_found_false.extend(calc_old_size_ops)
            (
                do_new_buffers_ops,
                new_idx_buffer_ptr,
                new_data_buffer_ptr,
                idx_elem_ptr,
                data_elem_ptr,
            ) = self.do_new_buffers(
                starting_layout,
                extent_resolver,
                new_size,
                idx_buffer_ptr,
                data_buffer_ptr,
                old_buffer_size,
                index,
            )
            if_found_false.extend(do_new_buffers_ops)
            # New we fill in the new index tuple and data elem

            initialiser = SingletonInitialiser(
                set_val, set(child_members), dict(child_dim_mapping)
            )
            set_up_child_ops = self.set_up_new_child(
                starting_layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                idx_elem_ptr,
                data_elem_ptr,
            )
            if_found_false.extend(set_up_child_ops)

            # The new buffers need to be put in the indexing struct and the old ones freed
            # if_found_false.append(
            #     printf.PrintFormatOp("4692 new idx {}", new_idx_buffer_ptr))
            if_found_false.append(llvm.StoreOp(new_idx_buffer_ptr, input_ptr))
            # if_found_false.append(
            #     printf.PrintFormatOp("4696 new data {}", new_data_buffer_ptr))
            if_found_false.append(
                llvm.StoreOp(new_data_buffer_ptr, data_buffer_ptr_ptr)
            )

        if_found_false.append(scf.Yield())

        if_found_op = scf.If(
            index_is_found,
            [],
            if_found_true,
            if_found_false,
        )
        if_must_do_set_ops.append(if_found_op)
        if_must_do_set_ops.append(scf.Yield())
        if_must_do_set_op = scf.If(must_do_set_cond_op.result, [], if_must_do_set_ops)
        if_must_start_set_ops.append(if_must_do_set_op)

        if_must_start_set_ops.append(scf.Yield())
        if_must_do_set_op = scf.If(
            must_start_set_cond_op.result, [], if_must_start_set_ops
        )
        ops.append(if_must_do_set_op)
        return ops

    def ensure_space(
        self,
        layout: dlt.UnpackedCOOLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index_range: SSAValue,
        direct_iterate_func: LinearIterCallbackFunc,
        direct_members: set[dlt.MemberAttr],
        direct_dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> tuple[list[Operation], SSAValue]:

        ops = []
        extract_ops, start, end = _extract_indices_from_index_range(index_range, as_index=True)
        ops.extend(extract_ops)

        idx_buffer_ptr_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        ops.append(idx_buffer_ptr_op)
        idx_buffer_ptr = idx_buffer_ptr_op.dereferenced_value

        while_op, index_is_found, index = self.loop_to_find_index(
            layout,
            start,
            end,
            idx_buffer_ptr,
            dim_mapping,
        )
        ops.append(while_op)

        # now get the data buffer ptr
        ptr_size = _get_ptr_size().sum()
        data_buffer_ptr_ops, data_buffer_ptr_ptr = ptr_size.add_to_llvm_pointer(
            input_ptr
        )
        ops.extend(data_buffer_ptr_ops)
        data_buffer_ptr_op = llvm.LoadOp(
            data_buffer_ptr_ptr, llvm.LLVMPointerType.opaque()
        )
        ops.append(data_buffer_ptr_op)
        data_buffer_ptr = data_buffer_ptr_op.dereferenced_value

        # get the members and dim_mapping that will eventually be needed by the recursive child get_setter_for
        child_dim_mapping = {
            dim: val for dim, val in dim_mapping.items() if dim not in layout.dimensions
        }
        child_members = members

        data_elem_size_ops, (data_elem_size, data_elem_size_extra) = (
            self.semantics.get_size(layout.child, extent_resolver).output()
        )
        ops.extend(data_elem_size_ops)

        child_fill_value_getter = self._get_fill_value_getter(
            layout.child,
            extent_resolver,
            fill_value_getter,
            base_type,
            data_buffer_ptr,
            data_elem_size,
            index,
        )

        # If index_is_found we can get the ptr to the child, else we need to reallocate and move lots of indices around
        if_found_true = []

        data_ptr_ops, child_elem_ptr = (
            NumericResult.from_ssa(index) * NumericResult.from_ssa(data_elem_size)
        ).add_to_llvm_pointer(data_buffer_ptr)
        if_found_true.extend(data_ptr_ops)

        # and finally we can set the ensure_space in this new child element
        child_ops, child_result = self.semantics.ensure_space(
            layout.child,
            base_type,
            child_members,
            child_dim_mapping,
            child_fill_value_getter,
            extent_resolver,
            child_elem_ptr,
        )
        if_found_true.extend(child_ops)
        if_found_true.append(scf.Yield(child_result))

        # If the value is not found we need to increment the indices of everything larger than this in the direct
        # data, then allocate new buffers with extra size for the new element, and mem copy the data across leaving
        # room for the new element. Then we can init the new element and can continue setting from there.
        if_found_false = []

        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        if_found_false.extend([true_op, false_op, zero_op, one_op])

        increment_indices_callback = self.SparseIndexIncrementCallback(
            dict(direct_dim_mapping),
            set(direct_members),
            false_op.result,
            zero_op.result,
            zero_op.result,
        )
        increment_ops, increment_results, _increment_exited = direct_iterate_func(
            increment_indices_callback, True
        )

        if_found_false.extend(increment_ops)
        # index_found = increment_results[0]
        # index_found (bool) is not used here as it is an internal check for inside the callback.
        last_index = increment_results[1]
        # last_index will be the number of non-zero elements now in the sparse buffer
        # found_index = increment_results[2]
        # found_index will be the index of the element we care about

        if layout.is_buffered():
            ptr_size = _get_ptr_size().sum()
            buffer_size_ptr_ops, buffer_size_ptr = ptr_size.add_to_llvm_pointer(
                data_buffer_ptr_ptr
            )
            if_found_false.extend(buffer_size_ptr_ops)
            buffer_size_op = llvm.LoadOp(buffer_size_ptr, i64)
            if_found_false.append(buffer_size_op)
            buffer_size_i64 = buffer_size_op.dereferenced_value
            convert_buffer_size_ops, buffer_size = index_convert(buffer_size_i64, IndexType())
            if_found_false.extend(convert_buffer_size_ops)

            cmp_buffer_size_op = arith.Cmpi(last_index, buffer_size, "ugt")
            need_bigger_buffers = cmp_buffer_size_op.result
            if_found_false.append(cmp_buffer_size_op)

            # if we need new buffers - handle it
            if_needs_new_buffers = []
            get_fill_value_ops, child_fill_value = child_fill_value_getter()
            if_needs_new_buffers.extend(get_fill_value_ops)

            buffer_scaler = layout.buffer_scaler.data
            buffer_scaler_const = arith.Constant(
                IntegerAttr(abs(buffer_scaler), IndexType())
            )
            if_needs_new_buffers.append(buffer_scaler_const)
            if buffer_scaler > 0:
                # multiply
                mul_op = arith.Muli(buffer_size, buffer_scaler_const)
                if_needs_new_buffers.append(mul_op)
                new_size_candidate = mul_op.result
                # if_needs_new_buffers.append(printf.PrintFormatOp("candidate {}   last index {}", new_size_candidate, last_index))
            else:
                # add
                add_op = arith.Addi(buffer_size, buffer_scaler_const)
                if_needs_new_buffers.append(add_op)
                new_size_candidate = add_op.result
                # if_needs_new_buffers.append(
                #     printf.PrintFormatOp("candidate {}   last index {}", new_size_candidate, last_index))
            new_size_op = arith.MaxUI(new_size_candidate, last_index)
            if_needs_new_buffers.append(new_size_op)
            new_size = new_size_op.result
            (
                do_new_buffers_ops,
                new_idx_buffer_ptr,
                new_data_buffer_ptr,
                new_idx_elem_ptr,
                new_data_elem_ptr,
            ) = self.do_new_buffers(
                layout,
                extent_resolver,
                new_size,
                idx_buffer_ptr,
                data_buffer_ptr,
                buffer_size,
                index,
            )
            if_needs_new_buffers.extend(do_new_buffers_ops)

            # Now we fill in the new index tuple and data elem
            initialiser = self.WriteEmptyRangeInitialiser(
                child_fill_value, child_members, child_dim_mapping
            )
            set_up_child_ops = self.set_up_new_child(
                layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                new_idx_elem_ptr,
                new_data_elem_ptr,
            )
            if_needs_new_buffers.extend(set_up_child_ops)

            # if_needs_new_buffers.append(
            #     printf.PrintFormatOp("4907 new idx {}", new_idx_buffer_ptr))
            if_needs_new_buffers.append(llvm.StoreOp(new_idx_buffer_ptr, input_ptr))
            # if_needs_new_buffers.append(
            #     printf.PrintFormatOp("4909 new data {}", new_data_buffer_ptr))
            if_needs_new_buffers.append(
                llvm.StoreOp(new_data_buffer_ptr, data_buffer_ptr_ptr)
            )
            # if_needs_new_buffers.append(
            #     printf.PrintFormatOp("4912 new size {}", new_size))
            convert_new_size_ops, new_size_i64 = index_convert(new_size, i64)
            if_needs_new_buffers.extend(convert_new_size_ops)
            if_needs_new_buffers.append(llvm.StoreOp(new_size_i64, buffer_size_ptr))
            if_needs_new_buffers.append(scf.Yield(child_fill_value))

            # alternatively we don't need new buffers - just need to move the data in the buffers we have
            if_needs_moved_buffers = []
            calc_old_elems_ops, (elems_to_copy,) = (
                NumericResult.from_ssa(last_index) - 1
            ).output()
            if_needs_moved_buffers.extend(calc_old_elems_ops)
            (
                do_move_buffers_ops,
                new_idx_elem_ptr,
                new_data_elem_ptr,
            ) = self.do_move_buffers(
                layout,
                extent_resolver,
                buffer_size,
                last_index,
                elems_to_copy,
                idx_buffer_ptr,
                data_buffer_ptr,
                index,
            )
            if_needs_moved_buffers.extend(do_move_buffers_ops)

            # Now we fill in the new index tuple and data elem
            get_fill_value_ops, child_fill_value = child_fill_value_getter()
            if_needs_moved_buffers.extend(get_fill_value_ops)
            initialiser = self.WriteEmptyRangeInitialiser(
                child_fill_value, child_members, child_dim_mapping
            )
            set_up_child_ops = self.set_up_new_child(
                layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                new_idx_elem_ptr,
                new_data_elem_ptr,
            )
            if_needs_moved_buffers.extend(set_up_child_ops)

            if_needs_moved_buffers.append(scf.Yield(child_fill_value))

            if_new_buffers_op = scf.If(
                need_bigger_buffers,
                [base_type],
                if_needs_new_buffers,
                if_needs_moved_buffers,
            )
            if_found_false.append(if_new_buffers_op)
            child_fill_value = if_new_buffers_op.results[0]
        else:
            # if it's not buffered then we know we just need to always do new buffers at the size of last_index
            # and we know the old buffer size was just one element less than the new last_index
            get_fill_value_ops, child_fill_value = child_fill_value_getter()
            if_found_false.extend(get_fill_value_ops)

            new_size = last_index
            calc_old_size_ops, (old_buffer_size,) = (
                NumericResult.from_ssa(last_index) - 1
            ).output()
            if_found_false.extend(calc_old_size_ops)
            (
                do_new_buffers_ops,
                new_idx_buffer_ptr,
                new_data_buffer_ptr,
                new_idx_elem_ptr,
                new_data_elem_ptr,
            ) = self.do_new_buffers(
                layout,
                extent_resolver,
                new_size,
                idx_buffer_ptr,
                data_buffer_ptr,
                old_buffer_size,
                index,
            )
            if_found_false.extend(do_new_buffers_ops)

            # Now we fill in the new index tuple and data elem
            initialiser = self.WriteEmptyRangeInitialiser(
                child_fill_value, child_members, child_dim_mapping
            )
            set_up_child_ops = self.set_up_new_child(
                layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                new_idx_elem_ptr,
                new_data_elem_ptr,
            )
            if_found_false.extend(set_up_child_ops)

            # if_found_false.append(
            #     printf.PrintFormatOp("4987 new idx {}", new_idx_buffer_ptr))
            if_found_false.append(llvm.StoreOp(new_idx_buffer_ptr, input_ptr))
            # if_found_false.append(
            #     printf.PrintFormatOp("4990 new data {}", new_data_buffer_ptr))
            if_found_false.append(
                llvm.StoreOp(new_data_buffer_ptr, data_buffer_ptr_ptr)
            )

        if_found_false.append(scf.Yield(child_fill_value))

        if_found_op = scf.If(
            index_is_found,
            [dlt.IndexRangeType()],
            if_found_true,
            if_found_false,
        )
        ops.append(if_found_op)

        return ops, if_found_op.output[0]

    def init_layout(
        self,
        layout: dlt.UnpackedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        is_last_element: bool | SSAValue,
        direct_init_func: LinearIterCallbackFunc,
        direct_iter_func: LinearIterCallbackFunc,
    ) -> tuple[list[Operation], list[SSAValue]]:
        assert len(callback_args) == len(init_callback.initial_iter_args())
        ops = []

        running_total_initial_zero = arith.Constant(IntegerAttr(0, IndexType()))
        ops.append(running_total_initial_zero)

        set_idx_callback = self.SparseIndexWritingCallback(
            layout.child.contents_type,
            running_total_initial_zero.result,
            initial_values,
            list(layout.dimensions),
        )
        direct_init_ops, total_size_iter_args, _exited_early = direct_init_func(
            set_idx_callback, True
        )
        ops.extend(direct_init_ops)
        total_size = total_size_iter_args[0]

        malloc_ops, idx_buffer_ptr, data_buffer_ptr = self.malloc_buffers(
            layout,
            total_size,
            self.semantics.get_size(layout.child, extent_resolver),
        )
        ops.extend(malloc_ops)

        idx_buffer_ptr_ptr = input_ptr
        ops.append(llvm.StoreOp(idx_buffer_ptr, idx_buffer_ptr_ptr))
        ptr_size = _get_ptr_size().sum()
        ptr_ops, data_buffer_ptr_ptr = ptr_size.add_to_llvm_pointer(input_ptr)
        ops.extend(ptr_ops)
        ops.append(llvm.StoreOp(data_buffer_ptr, data_buffer_ptr_ptr))

        if layout.is_buffered():
            ptr_size = _get_ptr_size().sum()
            ptr_ops, buffer_size_ptr = ptr_size.add_to_llvm_pointer(data_buffer_ptr_ptr)
            ops.extend(ptr_ops)
            convert_total_size_ops, total_size_i64 = index_convert(total_size, i64)
            ops.extend(convert_total_size_ops)
            ops.append(llvm.StoreOp(total_size_i64, buffer_size_ptr))

        data_element_size_ops, (data_element_size,) = (
            self.semantics.get_size(layout.child, extent_resolver).keep(1).output()
        )
        ops.extend(data_element_size_ops)

        set_data_callback = self.SparseDataWritingCallback(
            self.semantics,
            layout,
            idx_buffer_ptr,
            data_buffer_ptr,
            running_total_initial_zero.result,
            total_size,
            initial_values,
            data_element_size,
            init_callback,
            callback_args,
        )
        set_data_ops, iter_args, _exited = direct_iter_func(
            set_data_callback, is_last_element
        )
        ops.extend(set_data_ops)
        callback_iter_args = iter_args[3:]
        assert len(callback_args) == len(callback_iter_args)
        return ops, callback_iter_args

    def dealloc_layout(
        self,
        layout: dlt.UnpackedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        direct_iter_func: LinearIterCallbackFunc,
    ) -> list[Operation]:

        ops = []
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)

        idx_buffer_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        idx_buffer_ptr = idx_buffer_op.dereferenced_value
        ops.append(idx_buffer_op)
        data_buffer_ptr_ptr_ops, data_buffer_ptr_ptr = (
            _get_ptr_size().sum().add_to_llvm_pointer(input_ptr)
        )
        ops.extend(data_buffer_ptr_ptr_ops)
        data_buffer_ptr_op = llvm.LoadOp(
            data_buffer_ptr_ptr, llvm.LLVMPointerType.opaque()
        )
        ops.append(data_buffer_ptr_op)
        data_buffer_ptr = data_buffer_ptr_op.dereferenced_value

        data_elem_size_ops, (data_elem_size,) = (
            self.semantics.get_size(layout.child, extent_resolver).keep(1).output()
        )
        ops.extend(data_elem_size_ops)

        dealloc_callback = self.DeallocCallback(
            self.semantics,
            layout,
            extent_resolver,
            data_buffer_ptr,
            data_elem_size,
        )
        iter_ops, _, _ = direct_iter_func(dealloc_callback, True)
        ops.extend(iter_ops)
        ops.append(llvm.CallOp("free", idx_buffer_ptr))
        if self.semantics.print_memory_calls:
            ops.append(printf.PrintFormatOp("# called free({})", idx_buffer_ptr))
            ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
            ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
            ops.append(llvm.CallOp("fflush", nullptr.output))

        ops.append(llvm.CallOp("free", data_buffer_ptr))
        if self.semantics.print_memory_calls:
            ops.append(printf.PrintFormatOp("# called free({})", data_buffer_ptr))
            ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
            ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
            ops.append(llvm.CallOp("fflush", nullptr.output))

        return ops

    def linear_iterate(
        self,
        layout: dlt.UnpackedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        is_last_element: bool | SSAValue,
        direct_iter_func: LinearIterCallbackFunc,
        reversed_direction: bool = False,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        ops = []
        idx_buffer_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        idx_buffer_ptr = idx_buffer_op.dereferenced_value
        ops.append(idx_buffer_op)
        data_buffer_ops, data_buffer_ptr_ptr = (
            _get_ptr_size().sum().add_to_llvm_pointer(input_ptr)
        )
        ops.extend(data_buffer_ops)
        data_buffer_op = llvm.LoadOp(data_buffer_ptr_ptr, llvm.LLVMPointerType.opaque())
        ops.append(data_buffer_op)
        data_buffer_ptr = data_buffer_op.dereferenced_value

        data_elem_size_ops, (data_elem_size,) = (
            self.semantics.get_size(layout.child, extent_resolver).keep(1).output()
        )
        ops.extend(data_elem_size_ops)

        linear_iter_callback: Callback = self.SparseLinearIterCallback(
            self.semantics,
            layout,
            idx_buffer_ptr,
            data_buffer_ptr,
            data_elem_size,
            callback,
            callback_args,
            is_last_element,
            reversed_direction,
        )
        child_ops, child_iter_args, child_exited_early = direct_iter_func(
            linear_iter_callback, True
        )
        ops.extend(child_ops)
        return ops, child_iter_args, child_exited_early

    def make_sparse_loop_callback_for(
        self,
        starting_layout: dlt.UnpackedCOOLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], LoopCallback]:
        ops = []
        idx_buffer_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        idx_buffer_ptr = idx_buffer_op.dereferenced_value
        ops.append(idx_buffer_op)
        data_buffer_ops, data_buffer_ptr_ptr = (
            _get_ptr_size().sum().add_to_llvm_pointer(input_ptr)
        )
        ops.extend(data_buffer_ops)
        data_buffer_op = llvm.LoadOp(data_buffer_ptr_ptr, llvm.LLVMPointerType.opaque())
        ops.append(data_buffer_op)
        data_buffer_ptr = data_buffer_op.dereferenced_value

        data_elem_size_ops, (data_elem_size,) = (
            self.semantics.get_size(starting_layout.child, extent_resolver)
            .keep(1)
            .output()
        )
        ops.extend(data_elem_size_ops)

        new_callback = self.SparseMakeLoopCallback(
            self.semantics,
            starting_layout,
            ending_layout,
            extent_resolver,
            members,
            dim_mapping,
            dims_to_loop,
            idx_buffer_ptr,
            data_buffer_ptr,
            data_elem_size,
            callback,
            callback_args,
        )
        return ops, new_callback

    @staticmethod
    def loop_to_find_index(
        upcoo_layout: dlt.UnpackedCOOLayoutAttr,
        start: SSAValue,
        end: SSAValue,
        index_buffer_ptr: SSAValue,
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ):
        num_dims = len(upcoo_layout.dimensions)

        # Do {
        #   If in bounds {
        #       check if index is found,
        #       or if we've found an index larger,
        #       return (found or over), found # exit-loop?, index-found?
        #   } else {
        #       return true , false # exit-loop, index-not-found
        # } while (not exit-loop) {
        #   increment index
        # }

        before = Block()
        before_index = before.insert_arg(IndexType(), 0)

        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        before.add_ops([false_op, true_op])

        before_index_in_range = arith.Cmpi(before_index, end, "ult")
        before.add_op(before_index_in_range)

        # IF index_in_range: go through the sparse dimensions' indices at current index to check for: 1) if they are
        # equal to the indices we're looking for; 2) if they are lexicographically larger than the indices we're
        # looking for, i.e. we can stop looking because we store sparse values sorted by ascending lexicographic sparse
        # indices
        if_in_range_true = []
        running_is_over = false_op.result
        running_is_found = true_op.result
        index_offset_ops, (index_offset,) = (
            _get_accepted_type_size(i64).sum()
            * num_dims
            * NumericResult.from_ssa(before_index)
        ).output()
        if_in_range_true.extend(index_offset_ops)

        for i, dim in enumerate(upcoo_layout.dimensions):
            stored_idx_ptr_ops, stored_idx_ptr = (
                NumericResult.from_ssa(index_offset)
                + (
                    NumericResult.from_const(i)
                    * (_get_accepted_type_size(i64).sum())
                )
            ).add_to_llvm_pointer(index_buffer_ptr)
            if_in_range_true.extend(stored_idx_ptr_ops)
            inner_load_stored_index = llvm.LoadOp(stored_idx_ptr, i64)
            if_in_range_true.append(inner_load_stored_index)
            convert_inner_load_index_ops, inner_loaded_index = index_convert(inner_load_stored_index.dereferenced_value, IndexType())
            if_in_range_true.extend(convert_inner_load_index_ops)
            get_index_ops, (get_index,) = dim_mapping[dim].get().output()
            if_in_range_true.extend(get_index_ops)

            inner_is_over_cmp = arith.Cmpi(
                inner_loaded_index, get_index, "ugt"
            )
            if_in_range_true.append(inner_is_over_cmp)
            inner_is_found_cmp = arith.Cmpi(
                inner_loaded_index, get_index, "eq"
            )
            if_in_range_true.append(inner_is_found_cmp)

            inner_is_now_over = arith.AndI(running_is_found, inner_is_over_cmp)
            if_in_range_true.append(inner_is_now_over)
            inner_is_greater_reduce = arith.OrI(running_is_over, inner_is_now_over)
            if_in_range_true.append(inner_is_greater_reduce)

            inner_is_found_reduce = arith.AndI(running_is_found, inner_is_found_cmp)
            if_in_range_true.append(inner_is_found_reduce)

            running_is_found = inner_is_found_reduce.result
            running_is_over = inner_is_greater_reduce.result

        exit_loop_cond = arith.OrI(running_is_found, running_is_over)
        if_in_range_true.append(exit_loop_cond)

        if_in_range_true.append(scf.Yield(exit_loop_cond.result, running_is_found))

        if_in_range_false = [scf.Yield(true_op.result, false_op.result)]

        if_in_range_op = scf.If(
            before_index_in_range,
            [IntegerType(1), IntegerType(1)],
            if_in_range_true,
            if_in_range_false,
        )
        before.add_op(if_in_range_op)
        before_exit_loop = if_in_range_op.output[0]
        before_is_found = if_in_range_op.output[1]

        before_continue_loop_op = arith.Cmpi(before_exit_loop, false_op, "eq")
        before.add_op(before_continue_loop_op)

        before.add_op(
            scf.Condition(before_continue_loop_op.result, before_is_found, before_index)
        )

        after = Block()
        after.insert_arg(IntegerType(1), 0)  # after_is_found =
        after_index = after.insert_arg(IndexType(), 1)
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        after.add_op(one_op)

        new_index_op = arith.Addi(after_index, one_op.result)
        after.add_op(new_index_op)
        after.add_op(scf.Yield(new_index_op.result))

        # While op, index <= start, returns (found, index)
        while_op = scf.While([start], [IntegerType(1), IndexType()], [before], [after])

        index_is_found = while_op.res[0]
        index = while_op.res[1]

        return while_op, index_is_found, index

    def malloc_buffers(
        self,
        upcoo_layout: dlt.UnpackedCOOLayoutAttr,
        num_non_zero: SSAValue,
        child_size: NumericResult2,
        old_buffers: tuple[SSAValue, SSAValue] | None = None,
    ) -> tuple[list[Operation], SSAValue, SSAValue]:
        ops = []
        # Malloc the Unpacked COO index Buffer:
        alloc_idx_buffer_bytes_numeric = (
            _get_accepted_type_size(i64).sum()
            * len(upcoo_layout.dimensions)
            * NumericResult.from_ssa( num_non_zero)
        )
        alloc_idx_buffer_bytes_ops, (alloc_idx_buffer_bytes,) = (
            alloc_idx_buffer_bytes_numeric.output()
        )
        ops.extend(alloc_idx_buffer_bytes_ops)
        convert_idx_bytes_ops, alloc_idx_buffer_bytes_i64 = index_convert(alloc_idx_buffer_bytes, i64)
        ops.extend(convert_idx_bytes_ops)

        if old_buffers is None:
            malloc_idx_buffer = llvm.CallOp(
                "malloc",
                alloc_idx_buffer_bytes_i64,
                return_type=llvm.LLVMPointerType.opaque(),
            )
            ops.append(malloc_idx_buffer)
            idx_buffer_ptr = malloc_idx_buffer.returned
            if self.semantics.print_memory_calls:
                ops.append(printf.PrintFormatOp("# called malloc({}) -> {}", alloc_idx_buffer_bytes_i64, idx_buffer_ptr))
                ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
                ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
                ops.append(llvm.CallOp("fflush", nullptr.output))
        else:

            realloc_idx_buffer = llvm.CallOp(
                "realloc",
                old_buffers[0],
                alloc_idx_buffer_bytes_i64,
                return_type=llvm.LLVMPointerType.opaque(),
            )
            ops.append(realloc_idx_buffer)
            idx_buffer_ptr = realloc_idx_buffer.returned
            if self.semantics.print_memory_calls:
                ops.append(printf.PrintFormatOp("# called realloc({}, {}) -> {}", old_buffers[0], alloc_idx_buffer_bytes_i64, idx_buffer_ptr))
                ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
                ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
                ops.append(llvm.CallOp("fflush", nullptr.output))

        # c_db_ops, (child_size, child_size_debug) = child_size.split_n(2)
        # ops.extend(c_db_ops)

        # Malloc the Unpacked COO data Buffer:
        alloc_data_buffer_bytes_numeric = (
            child_size * NumericResult.from_mixed([], num_non_zero)
        ).sum()
        alloc_data_buffer_ops, (alloc_data_buffer_bytes,) = (
            alloc_data_buffer_bytes_numeric.output()
        )
        ops.extend(alloc_data_buffer_ops)
        convert_data_bytes_ops,  alloc_data_buffer_bytes_i64 = index_convert(alloc_data_buffer_bytes, i64)
        ops.extend(convert_data_bytes_ops)

        if old_buffers is None:
            malloc_data_buffer = llvm.CallOp(
                "malloc",
                alloc_data_buffer_bytes_i64,
                return_type=llvm.LLVMPointerType.opaque(),
            )
            ops.append(malloc_data_buffer)
            data_buffer_ptr = malloc_data_buffer.returned
            if self.semantics.print_memory_calls:
                ops.append(printf.PrintFormatOp("# called malloc({}) -> {}", alloc_data_buffer_bytes_i64, data_buffer_ptr))
                ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
                ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
                ops.append(llvm.CallOp("fflush", nullptr.output))
        else:
            realloc_data_buffer = llvm.CallOp(
                "realloc",
                old_buffers[1],
                alloc_data_buffer_bytes_i64,
                return_type=llvm.LLVMPointerType.opaque(),
            )
            ops.append(realloc_data_buffer)
            data_buffer_ptr = realloc_data_buffer.returned
            if self.semantics.print_memory_calls:
                ops.append(printf.PrintFormatOp("# called realloc({}, {}) -> {}", old_buffers[1], alloc_data_buffer_bytes_i64, data_buffer_ptr))
                ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
                ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
                ops.append(llvm.CallOp("fflush", nullptr.output))

        # db_ops, (child_size_db_base, child_size_db_extra) = child_size_debug.output()
        # ops.extend(db_ops)
        # ops.append(printf.PrintFormatOp("Malloc  capacity:{}  child_size:{} + {}", num_non_zero, child_size_db_base, child_size_db_extra))
        # ops.append(printf.PrintFormatOp("Malloc  index buffer:{}  data buffer:{}", malloc_idx_buffer.returned, malloc_data_buffer.returned))
        # ops.append(printf.PrintFormatOp("Malloc  index bytes:{}  data bytes:{} for "+str(upcoo_layout), alloc_idx_buffer_bytes,  alloc_data_buffer_bytes))

        return ops, idx_buffer_ptr, data_buffer_ptr

    @staticmethod
    def copy_to_new_buffers(
        old_buffer_ptr: SSAValue,
        new_buffer_ptr: SSAValue,
        element_size: SSAValue,
        new_elem_index: SSAValue,
        new_capacity: SSAValue,
        elements_to_copy: SSAValue,
        extra_space: SSAValue,
        shuffle_instead: bool = False,
    ) -> tuple[list[Operation], SSAValue]:
        ops = []

        # ops.append(printf.PrintFormatOp("# Copy to new buffers.  elem_size:{}  new_elem_index:{}  new_capacity:{}  elements_to_copy: {}  extra_space:{} shuffle_instead:" + str(shuffle_instead), element_size, new_elem_index, new_capacity, elements_to_copy, extra_space))
        # ops.append(printf.PrintFormatOp("# old buffer: {}  new buffer: {}", old_buffer_ptr, new_buffer_ptr))
        # calculate how many bytes to copy before the new element
        pre_elem_buffer_size_ops, (pre_elem_buffer_size,) = (
            NumericResult.from_ssa(element_size)
            * NumericResult.from_ssa(new_elem_index)
        ).output()
        ops.extend(pre_elem_buffer_size_ops)

        if not shuffle_instead:
            # Copy the data before the new element
            # ops.append(printf.PrintFormatOp("# Call memcpy (before elem) dst: {}  src: {}  size: {}", new_buffer_ptr, old_buffer_ptr, pre_elem_buffer_size))
            convert_pre_elem_buffer_size_ops, pre_elem_buffer_size_i64 = index_convert(pre_elem_buffer_size, i64)
            ops.extend(convert_pre_elem_buffer_size_ops)
            pre_elem_idx_copy_op = llvm.CallOp(
                "memcpy",
                new_buffer_ptr,
                old_buffer_ptr,
                pre_elem_buffer_size_i64,
                return_type=None,
            )
            ops.append(pre_elem_idx_copy_op)

        # calculate the ptr in the old buffer - the rest that wasn't copied before
        post_elem_buffer_ptr_ops, post_elem_buffer_ptr = NumericResult.from_ssa(
            pre_elem_buffer_size
        ).add_to_llvm_pointer(old_buffer_ptr)
        ops.extend(post_elem_buffer_ptr_ops)
        # calculate the ptr in the new buffer, adding an extra element_size to give room for the new element
        new_post_elem_buffer_ptr_ops, new_post_elem_buffer_ptr = (
            NumericResult.from_ssa(pre_elem_buffer_size)
            + NumericResult.from_ssa(element_size)
        ).add_to_llvm_pointer(new_buffer_ptr)
        ops.extend(new_post_elem_buffer_ptr_ops)
        # calculate how much data to move, this will be the new last_index (new number of elements in the buffer)
        # minus what was copied already (new_elem_index), minus 1 to account for 1 new element
        post_elem_buffer_size_ops, (post_elem_buffer_size,) = (
            (
                NumericResult.from_ssa(element_size)
                * (
                    (
                        NumericResult.from_ssa(elements_to_copy)
                        - NumericResult.from_ssa(new_elem_index)
                    )
                )
            )
            + NumericResult.from_ssa(extra_space)
        ).output()
        ops.extend(post_elem_buffer_size_ops)
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        ops.append(zero_op)
        cmp_op = arith.Cmpi(zero_op, post_elem_buffer_size, "ne")
        ops.append(cmp_op)
        # ops.append(printf.PrintFormatOp("# Post elem copy size: {}, !=0?: {}", post_elem_buffer_size, cmp_op.result))
        if_needs_post_copy = []
        convert_post_elem_buffer_size_ops, post_elem_buffer_size_i64 = index_convert(post_elem_buffer_size, i64)
        if_needs_post_copy.extend(convert_post_elem_buffer_size_ops)

        if shuffle_instead:
            # Copy the elements after the new element - but expect an overlap
            # if_needs_post_copy.append(printf.PrintFormatOp("# Call memmove (after elem) dst: {}  src: {}  size: {}", new_post_elem_buffer_ptr,
            #                                 post_elem_buffer_ptr, post_elem_buffer_size))
            pre_elem_idx_copy_op = llvm.CallOp(
                "memmove",
                new_post_elem_buffer_ptr,
                post_elem_buffer_ptr,
                post_elem_buffer_size_i64,
                return_type=None,
            )
            if_needs_post_copy.append(pre_elem_idx_copy_op)
        else:
            # Copy the elements after the new element
            # if_needs_post_copy.append(printf.PrintFormatOp("# Call memcpy (after elem) dst: {}  src: {}  size: {}", new_post_elem_buffer_ptr,
            #                                 post_elem_buffer_ptr, post_elem_buffer_size))
            pre_elem_idx_copy_op = llvm.CallOp(
                "memcpy",
                new_post_elem_buffer_ptr,
                post_elem_buffer_ptr,
                post_elem_buffer_size_i64,
                return_type=None,
            )
            if_needs_post_copy.append(pre_elem_idx_copy_op)
        if_needs_post_copy.append(scf.Yield())
        ops.append(scf.If(cmp_op.result, [], if_needs_post_copy))

        # calculate the ptr to the new_element to return
        new_elem_ptr_ops, new_elem_ptr = NumericResult.from_ssa(
            pre_elem_buffer_size
        ).add_to_llvm_pointer(new_buffer_ptr)
        ops.extend(new_elem_ptr_ops)
        # ops.append(
        #     printf.PrintFormatOp("# new elem ptr {}", new_elem_ptr))
        return ops, new_elem_ptr

    class SparseDataWritingCallback(Callback):
        def __init__(
            self,
            semantics: SemanticsMapper,
            unpack_coo_layout: dlt.UnpackedCOOLayoutAttr,
            running_idx_buffer_ptr: SSAValue,
            running_data_buffer_ptr: SSAValue,
            running_total: SSAValue,
            total_non_zeros: SSAValue,
            sparse_initial_values: Initialiser,
            data_element_size: SSAValue,
            inner_callback: Callback,
            inner_callback_args: list[SSAValue],
        ):
            self.semantics = semantics
            self.unpack_coo_layout = unpack_coo_layout

            self.idx_buffer_ptr = running_idx_buffer_ptr
            assert isinstance(self.idx_buffer_ptr.type, llvm.LLVMPointerType)
            self.data_buffer_ptr = running_data_buffer_ptr
            assert isinstance(self.data_buffer_ptr.type, llvm.LLVMPointerType)
            self.running_total = running_total
            assert isinstance(self.running_total.type, IndexType)

            self.total_non_zeros = total_non_zeros
            assert isinstance(self.total_non_zeros.type, IndexType)
            self.sparse_initial_values = sparse_initial_values
            # assert all(
            #     isinstance(v.type, dlt.PtrType) for v in sparse_initial_values.values()
            # )
            self.data_element_size = data_element_size
            assert isinstance(self.data_element_size.type, IndexType)

            self.inner_callback = inner_callback
            self.inner_callback_args = inner_callback_args

            super().__init__(
                [running_idx_buffer_ptr, running_data_buffer_ptr, running_total]
                + self.inner_callback_args
            )

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, SSAValue],
            extent_resolver: ExtentResolver,
            base_type: dlt.AcceptedTypes,
            ptr: SSAValue,
            has_extra_space: SSAValue,
            is_last_element: bool | SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:

            ops: list[Operation] = []

            if is_last_element is True:
                true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
                ops.append(true_const)
                is_last_element = true_const.result

            extents = [d.extent for d in self.unpack_coo_layout.dimensions]
            extent_args = []
            for e in extents:
                if e.get_stage() >= dlt.Stage.INIT:
                    nr = extent_resolver.resolve(e)
                    ext_ops, ext = nr.output()
                    ops.extend(ext_ops)
                    extent_args.extend(ext)

            derived_initialiser = DerivedInitialiser(
                self.sparse_initial_values, members, dim_map
            )

            # Create the loop(s) over sparse dimensions
            iter_op = dlt.IterateOp(
                extents,
                extent_args,
                [],  # dim_specifiers,
                [],  # dlt_ptrs,
                iter_args,
                dlt.NestedIterationOrderAttr.generate_for(list(range(len(extents)))),
                None,
                None,
            )
            ops.append(iter_op)

            sparse_iter_block = iter_op.body.block
            sparse_iter_block.erase_op(
                sparse_iter_block.last_op
            )  # remove yield as we will add our own
            block_running_idx_buffer_ptr = iter_op.get_block_arg_for_iter_arg_idx(0)
            block_running_data_buffer_ptr = iter_op.get_block_arg_for_iter_arg_idx(1)
            block_running_total = iter_op.get_block_arg_for_iter_arg_idx(2)

            init_data_callback_args = [
                iter_op.get_block_arg_for_iter_arg_idx(3 + i)
                for i in range(len(self.inner_callback_args))
            ]

            block_dim_map = {
                d: a
                for d, a in zip(
                    self.unpack_coo_layout.dimensions,
                    iter_op.get_block_args_for_extent_args(),
                )
            }

            block_callback = DerivedCallback(
                self.inner_callback, members, dim_map | block_dim_map
            )

            get_non_zeros_ops, found_non_zeros = derived_initialiser.get_non_zero(
                set(),
                {k: ArgIndexGetter(v) for k, v in block_dim_map.items()},
                self.unpack_coo_layout.child.contents_type,
            )
            sparse_iter_block.add_ops(get_non_zeros_ops)
            bool_ssa_ops, has_non_zero = _make_bool_ssa(found_non_zeros)
            sparse_iter_block.add_ops(bool_ssa_ops)

            # setup If statement - if has_non_zeros:
            true_ops: list[Operation] = []
            one = NumericResult.from_mixed([], 1)
            running_total_numeric = NumericResult.from_mixed([], block_running_total)
            new_total_ops, (new_running_total,) = (running_total_numeric + one).output()
            true_ops.extend(new_total_ops)

            current_block_running_idx_buffer_ptr = block_running_idx_buffer_ptr
            for idx_arg in iter_op.get_block_args_for_extent_args():
                convert_idx_arg_ops, idx_arg_i64 = index_convert(idx_arg, i64)
                true_ops.extend(convert_idx_arg_ops)
                true_ops.append(
                    llvm.StoreOp(idx_arg_i64, current_block_running_idx_buffer_ptr)
                )
                idx_size = _get_accepted_type_size(i64).sum()
                new_idx_ptr_ops, current_block_running_idx_buffer_ptr = (
                    idx_size.add_to_llvm_pointer(current_block_running_idx_buffer_ptr)
                )
                true_ops.extend(new_idx_ptr_ops)
            new_block_running_idx_buffer_ptr = current_block_running_idx_buffer_ptr

            is_last_iter_op = arith.Cmpi(new_running_total, self.total_non_zeros, "eq")
            true_ops.append(is_last_iter_op)
            has_extra_space_op = arith.AndI(is_last_iter_op, has_extra_space)
            true_ops.append(has_extra_space_op)
            if isinstance(is_last_element, SSAValue):
                is_last_element_op = arith.AndI(is_last_iter_op, is_last_element)
                true_ops.append(is_last_element_op)
                is_last_element = is_last_iter_op.result

            init_data_ops, new_init_data_callback_args = self.semantics.init_layout(
                self.unpack_coo_layout.child,
                extent_resolver,
                block_running_data_buffer_ptr,
                DerivedInitialiser(derived_initialiser, set(), block_dim_map),
                block_callback,
                init_data_callback_args,
                has_extra_space_op.result,
                is_last_element,
            )
            true_ops.extend(init_data_ops)

            size_numeric = NumericResult.from_mixed([], self.data_element_size)
            new_data_ptr_ops, new_block_running_data_buffer_ptr = (
                size_numeric.add_to_llvm_pointer(block_running_data_buffer_ptr)
            )
            true_ops.extend(new_data_ptr_ops)

            true_ops.append(
                scf.Yield(
                    new_block_running_idx_buffer_ptr,
                    new_block_running_data_buffer_ptr,
                    new_running_total,
                    *new_init_data_callback_args,
                )
            )

            if_op = scf.If(
                has_non_zero,
                [
                    llvm.LLVMPointerType.opaque(),
                    llvm.LLVMPointerType.opaque(),
                    IndexType(),
                ]
                + [arg.type for arg in self.inner_callback.initial_iter_args()],
                true_ops,
                [
                    scf.Yield(
                        block_running_idx_buffer_ptr,
                        block_running_data_buffer_ptr,
                        block_running_total,
                        *init_data_callback_args,
                    )
                ],
            )
            sparse_iter_block.add_op(if_op)
            sparse_iter_block.add_op(dlt.IterateYieldOp(*if_op.output))

            new_iter_args = iter_op.results
            return ops, new_iter_args, False

    class SparseLinearIterCallback(Callback):
        def __init__(
            self,
            semantics: SemanticsMapper,
            unpack_coo_layout: dlt.UnpackedCOOLayoutAttr,
            idx_buffer_ptr: SSAValue,
            data_buffer_ptr: SSAValue,
            data_element_size: SSAValue,
            inner_callback: Callback,
            inner_callback_args: list[SSAValue],
            inner_is_last_element: bool | SSAValue,
            reversed_direction: bool,
        ):
            self.semantics = semantics
            self.unpack_coo_layout = unpack_coo_layout

            self.idx_buffer_ptr = idx_buffer_ptr
            assert isinstance(self.idx_buffer_ptr.type, llvm.LLVMPointerType)
            self.data_buffer_ptr = data_buffer_ptr
            assert isinstance(self.data_buffer_ptr.type, llvm.LLVMPointerType)

            self.data_element_size = data_element_size
            assert isinstance(self.data_element_size.type, IndexType)

            self.inner_callback = inner_callback
            # self.inner_callback_args = inner_callback_args

            self.inner_is_last_element = inner_is_last_element
            self.reversed_direction = reversed_direction

            super().__init__(inner_callback_args, inner_callback.can_exits_early())

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, SSAValue],
            extent_resolver: ExtentResolver,
            base_type: dlt.AcceptedTypes,
            ptr: SSAValue,
            has_extra_space: SSAValue,
            is_last_element: bool | SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            ops: list[Operation] = []
            true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.append(true_const)
            false_const = arith.Constant(IntegerAttr(0, IntegerType(1)))
            ops.append(false_const)

            if is_last_element is True:
                is_last_element = true_const.result
            if self.inner_is_last_element is True:
                true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
                ops.append(true_const)
                inner_is_last_element = true_const.result
            else:
                inner_is_last_element = self.inner_is_last_element

            getter_ops, index_range, index_range_found = self.semantics.get_getter_for(
                terminal_layout, dlt.IndexRangeType(), set(), {}, extent_resolver, ptr
            )
            ops.extend(getter_ops)
            extract_ops, start, end = _extract_indices_from_index_range(index_range, as_index=True)
            ops.extend(extract_ops)
            ops.append(one_op := arith.Constant(IntegerAttr(1, IndexType())))

            idxs_step_ops, (idxs_step,) = (
                _get_accepted_type_size(i64).sum()
                * len(self.unpack_coo_layout.dimensions)
            ).output()
            ops.extend(idxs_step_ops)
            idx_step_ops, (idx_step,) = (
                _get_accepted_type_size(i64).sum().output()
            )
            ops.extend(idx_step_ops)

            end_minus_one_op = arith.Subi(end, one_op.result)
            ops.append(end_minus_one_op)

            if self.reversed_direction:
                start_idx = end_minus_one_op.result
                end_idx = start
                ops.append(step_op := arith.Constant(IntegerAttr(-1, IndexType())))
                step = step_op.result
                ops.append(zero_op := arith.Constant(IntegerAttr(0, IndexType())))
                last_iter_idx = zero_op.result
            else:
                start_idx, end_idx, step = start, end, one_op.result
                last_iter_idx = end_minus_one_op.result

            block = Block()
            index = block.insert_arg(IndexType(), 0)
            block.insert_arg(IntegerType(1), 1)  # exit early
            block_inner_callback_args = [
                block.insert_arg(arg.type, len(block.args)) for arg in iter_args
            ]

            # sort out has_extra_space and is_last_elem the data elem has extra space if this is the last elem of
            # iteration over the direct data source of the indexing node and the last iteration within this loop
            # across the sparse dims this is the last elem if it's the last iteration of the loop across the sparse
            # data (is_last_iter_op) and it's the last elem of the index source iteration (is_last_elem) and it was
            # the last elem for the indexing node (self.inner_is_last_elem)
            is_last_iter_op = arith.Cmpi(last_iter_idx, index, "eq")
            block.add_op(is_last_iter_op)
            if is_last_element is False:
                has_extra_space_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
                block.add_op(has_extra_space_op)
                inner_has_extra_space = has_extra_space_op.result
            else:
                has_extra_space_op = arith.AndI(is_last_iter_op, is_last_element)
                block.add_op(has_extra_space_op)
                inner_has_extra_space = has_extra_space_op.result

            if isinstance(inner_is_last_element, SSAValue):
                is_last_element_op = arith.AndI(is_last_iter_op, is_last_element)
                block.add_op(is_last_element_op)
                inner_last_elem_op = arith.AndI(
                    is_last_element_op.result, inner_is_last_element
                )
                block.add_op(inner_last_elem_op)
                inner_is_last_element = inner_last_elem_op.result

            idx_buffer_ops, current_idx_buffer_ptr = (
                NumericResult.from_mixed([], index)
                * NumericResult.from_mixed([], idxs_step)
            ).add_to_llvm_pointer(self.idx_buffer_ptr)
            block.add_ops(idx_buffer_ops)
            data_buffer_ops, current_data_buffer_ptr = (
                NumericResult.from_mixed([], index)
                * NumericResult.from_mixed([], self.data_element_size)
            ).add_to_llvm_pointer(self.data_buffer_ptr)
            block.add_ops(data_buffer_ops)

            sparse_dims = {}
            for dim in self.unpack_coo_layout.dimensions:
                sparse_index_load = llvm.LoadOp(current_idx_buffer_ptr, i64)
                block.add_op(sparse_index_load)
                convert_sparse_index_ops, sparse_index = index_convert(
                    sparse_index_load.dereferenced_value, IndexType())
                block.add_ops(convert_sparse_index_ops)
                sparse_dims[dim] = sparse_index
                increment_idx_ops, current_idx_buffer_ptr = NumericResult.from_mixed(
                    [], idx_step
                ).add_to_llvm_pointer(current_idx_buffer_ptr)
                block.add_ops(increment_idx_ops)

            block_callback = DerivedCallback(
                self.inner_callback, members, dim_map | sparse_dims
            )

            child_ops, new_inner_callback_arg, exited_early = (
                self.semantics.linear_iterate(
                    self.unpack_coo_layout.child,
                    extent_resolver,
                    current_data_buffer_ptr,
                    block_callback,
                    block_inner_callback_args,
                    inner_has_extra_space,
                    inner_is_last_element,
                    reversed_direction=self.reversed_direction,
                )
            )
            block.add_ops(child_ops)

            exited_ops, exit_early = _make_bool_ssa(exited_early)
            block.add_ops(exited_ops)

            inc_index_op = arith.Addi(index, step)
            block.add_op(inc_index_op)

            block.add_op(
                scf.Yield(
                    inc_index_op.result,
                    exit_early,
                    *new_inner_callback_arg,
                )
            )

            condition_block = Block()
            c_index = condition_block.insert_arg(IndexType(), 0)
            c_exit = condition_block.insert_arg(IntegerType(1), 1)
            c_block_inner_callback_args = [
                condition_block.insert_arg(arg.type, len(condition_block.args))
                for arg in iter_args
            ]
            if self.reversed_direction:
                cmp1 = arith.Cmpi(c_index, end_idx, "sge")
            else:
                cmp1 = arith.Cmpi(c_index, end_idx, "slt")
            condition_block.add_ops(
                [
                    cmp1,
                    cmp2 := arith.Cmpi(false_const.result, c_exit, "eq"),
                    cmp := arith.AndI(cmp1.result, cmp2.result),
                    scf.Condition(cmp, c_index, c_exit, *c_block_inner_callback_args),
                ]
            )
            while_loop = scf.While(
                [start_idx, false_const.result, *iter_args],
                [IndexType(), IntegerType(1), *[a.type for a in iter_args]],
                [condition_block],
                [block],
            )
            ops.append(while_loop)
            output_inner_callback_iter_args = list(
                while_loop.results[2 : 2 + len(iter_args)]
            )
            did_exit_early = while_loop.results[1]

            return ops, output_inner_callback_iter_args, did_exit_early

    class SparseMakeLoopCallback(LoopCallback):
        def __init__(
            self,
            semantics: SemanticsMapper,
            unpack_coo_layout: dlt.UnpackedCOOLayoutAttr,
            end_layout: dlt.DirectLayout,
            extent_resolver: ExtentResolver,
            members: set[dlt.MemberAttr],
            dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
            dims_to_loop: set[dlt.DimensionAttr],
            idx_buffer_ptr: SSAValue,
            data_buffer_ptr: SSAValue,
            data_element_size: SSAValue,
            inner_callback: LoopCallback,
            inner_callback_args: list[SSAValue],
        ):
            self.semantics = semantics
            self.unpack_coo_layout = unpack_coo_layout

            self.end_layout = end_layout
            self.extent_resolver = extent_resolver
            self.members = members
            self.dim_mapping = dim_mapping
            self.dims_to_loop = dims_to_loop

            self.idx_buffer_ptr = idx_buffer_ptr
            assert isinstance(self.idx_buffer_ptr.type, llvm.LLVMPointerType)
            self.data_buffer_ptr = data_buffer_ptr
            assert isinstance(self.data_buffer_ptr.type, llvm.LLVMPointerType)

            self.data_element_size = data_element_size
            assert isinstance(self.data_element_size.type, IndexType)

            self.inner_callback = inner_callback

            super().__init__(inner_callback_args)

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, IndexGetter],
            dims_left_to_loop: set[dlt.DimensionAttr],
            extent_resolver: ExtentResolver,
            ptr: SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue]]:
            assert len(dims_left_to_loop) == 0
            ops: list[Operation] = []
            true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.append(true_const)
            false_const = arith.Constant(IntegerAttr(0, IntegerType(1)))
            ops.append(false_const)
            one_op = arith.Constant(IntegerAttr(1, IndexType()))
            ops.append(one_op)

            getter_ops, index_range, index_range_found = self.semantics.get_getter_for(
                terminal_layout, dlt.IndexRangeType(), set(), {}, extent_resolver, ptr
            )
            ops.extend(getter_ops)
            extract_ops, start, end = _extract_indices_from_index_range(index_range, as_index=True)
            ops.extend(extract_ops)

            idxs_step_ops, (idxs_step,) = (
                _get_accepted_type_size(i64).sum()
                * NumericResult.from_mixed([], len(self.unpack_coo_layout.dimensions))
            ).output()
            ops.extend(idxs_step_ops)
            idx_step_ops, (idx_step,) = (
                _get_accepted_type_size(i64).sum().output()
            )
            ops.extend(idx_step_ops)

            start_idx, end_idx, step = start, end, one_op.result

            block = Block()
            index = block.insert_arg(IndexType(), 0)
            block_inner_callback_args = [
                block.insert_arg(arg.type, len(block.args)) for arg in iter_args
            ]

            idx_buffer_ops, current_idx_buffer_ptr = (
                NumericResult.from_mixed([], index)
                * NumericResult.from_mixed([], idxs_step)
            ).add_to_llvm_pointer(self.idx_buffer_ptr)
            block.add_ops(idx_buffer_ops)
            data_buffer_ops, current_data_buffer_ptr = (
                NumericResult.from_mixed([], index)
                * NumericResult.from_mixed([], self.data_element_size)
            ).add_to_llvm_pointer(self.data_buffer_ptr)
            block.add_ops(data_buffer_ops)

            looped_sparse_dims = {}
            checked_sparse_dims = {}
            checked = true_const.result
            for dim in self.unpack_coo_layout.dimensions:
                sparse_index_load = llvm.LoadOp(current_idx_buffer_ptr, i64)
                block.add_op(sparse_index_load)
                convert_sparse_index_ops, sparse_index = index_convert(sparse_index_load.dereferenced_value, IndexType())
                block.add_ops(convert_sparse_index_ops)
                if dim in self.dims_to_loop:
                    looped_sparse_dims[dim] = sparse_index
                elif dim in self.dim_mapping:
                    checked_sparse_dims[dim] = sparse_index
                    val_wanted_ops, (val_wanted,) = self.dim_mapping[dim].get().output()
                    block.add_ops(val_wanted_ops)
                    cmp_op = arith.Cmpi(checked_sparse_dims[dim], val_wanted, "eq")
                    block.add_op(cmp_op)
                    checked_op = arith.AndI(cmp_op.result, checked)
                    block.add_op(checked_op)
                    checked = checked_op.result

                increment_idx_ops, current_idx_buffer_ptr = NumericResult.from_mixed(
                    [], idx_step
                ).add_to_llvm_pointer(current_idx_buffer_ptr)
                block.add_ops(increment_idx_ops)

            if_checked_true = []
            block_callback = DerivedLoopCallback(
                self.inner_callback,
                members,
                dim_map | looped_sparse_dims | checked_sparse_dims,
            )

            child_ops, new_inner_callback_arg = self.semantics.make_sparse_loop_for(
                self.unpack_coo_layout.child,
                self.end_layout,
                self.extent_resolver,
                current_data_buffer_ptr,
                block_callback,
                block_inner_callback_args,
                self.members,
                {
                    d: g
                    for d, g in self.dim_mapping.items()
                    if d not in checked_sparse_dims
                },
                self.dims_to_loop - set(looped_sparse_dims.keys()),
            )
            if_checked_true.extend(child_ops)
            if_checked_true.append(scf.Yield(*new_inner_callback_arg))
            if_checked_false = []
            if_checked_false.append(scf.Yield(*block_inner_callback_args))
            if_checked_op = scf.If(
                checked,
                [arg.type for arg in iter_args],
                if_checked_true,
                if_checked_false,
            )
            block.add_op(if_checked_op)

            block.add_op(scf.Yield(*if_checked_op.results))
            for_loop = scf.For(start_idx, end_idx, step, iter_args, block)
            ops.append(for_loop)
            output_callback_iter_args = list(for_loop.res)

            return ops, output_callback_iter_args

    def set_up_new_child(
        self,
        starting_layout: dlt.UnpackedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        initialiser: Initialiser,
        idx_elem_ptr: SSAValue,
        data_elem_ptr: SSAValue,
    ) -> list[Operation]:
        assert isinstance(idx_elem_ptr.type, llvm.LLVMPointerType)
        assert isinstance(data_elem_ptr.type, llvm.LLVMPointerType)
        ops = []
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        ops.append(false_op)

        running_idx_buffer_ptr = idx_elem_ptr
        for dim in starting_layout.dimensions:
            idx_arg_ops, (idx_arg,) = dim_mapping[dim].get().output()
            ops.extend(idx_arg_ops)
            convert_idx_arg_ops, idx_arg_i64 = index_convert(idx_arg, i64)
            ops.extend(convert_idx_arg_ops)
            ops.append(llvm.StoreOp(idx_arg_i64, running_idx_buffer_ptr))
            idx_size = _get_accepted_type_size(i64).sum()
            new_idx_ptr_ops, running_idx_buffer_ptr = idx_size.add_to_llvm_pointer(
                running_idx_buffer_ptr
            )
            ops.extend(new_idx_ptr_ops)
        # this is a new data element and so it needs to be init-ed
        child_init_ops, _ = self.semantics.init_layout(
            starting_layout.child,
            extent_resolver,
            data_elem_ptr,
            initialiser,
            NoCallback(),
            [],
            false_op.result,
            True,
        )
        ops.extend(child_init_ops)
        return ops

    def do_new_buffers(
        self,
        starting_layout: dlt.UnpackedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        new_size: SSAValue,
        idx_buffer_ptr: SSAValue,
        data_buffer_ptr: SSAValue,
        old_size: SSAValue,
        index: SSAValue,
    ) -> tuple[list[Operation], SSAValue, SSAValue, SSAValue, SSAValue]:
        assert isinstance(new_size.type, IndexType)
        assert isinstance(idx_buffer_ptr.type, llvm.LLVMPointerType)
        assert isinstance(data_buffer_ptr.type, llvm.LLVMPointerType)
        assert isinstance(index.type, IndexType)
        ops = []
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        ops.append(zero_op)

        # Malloc the new Unpacked COO Buffer:
        malloc_ops, new_idx_buffer_ptr, new_data_buffer_ptr = self.malloc_buffers(
            starting_layout,
            new_size,
            self.semantics.get_size(starting_layout.child, extent_resolver),
            old_buffers=(idx_buffer_ptr, data_buffer_ptr),
        )
        ops.extend(malloc_ops)

        # Set up for lots of memory copies to move the sparse index tuples and child data into their new larger buffers
        # Starting with Index tuples
        index_tuple_size_ops, (index_tuple_size,) = (
            _get_accepted_type_size(i64).keep(1)
            * len(starting_layout.dimensions)
        ).output()
        ops.extend(index_tuple_size_ops)

        idx_copy_ops, idx_elem_ptr = self.copy_to_new_buffers(
            new_idx_buffer_ptr,
            new_idx_buffer_ptr,
            index_tuple_size,
            index,
            new_size,
            old_size,
            zero_op.result,
            shuffle_instead=True,
        )
        ops.extend(idx_copy_ops)

        data_elem_size_ops, (data_elem_size, data_elem_size_extra) = (
            self.semantics.get_size(starting_layout.child, extent_resolver).output()
        )
        ops.extend(data_elem_size_ops)

        data_copy_ops, data_elem_ptr = self.copy_to_new_buffers(
            new_data_buffer_ptr,
            new_data_buffer_ptr,
            data_elem_size,
            index,
            new_size,
            old_size,
            data_elem_size_extra,
            shuffle_instead=True,
        )
        ops.extend(data_copy_ops)

        # ops.append(printf.PrintFormatOp("# free idx {}", idx_buffer_ptr))
        # not needed since using realloc: ops.append(llvm.CallOp("free", idx_buffer_ptr))
        # ops.append(printf.PrintFormatOp("# free data {}", data_buffer_ptr))
        # not needed since using realloc: ops.append(llvm.CallOp("free", data_buffer_ptr))

        return ops, new_idx_buffer_ptr, new_data_buffer_ptr, idx_elem_ptr, data_elem_ptr

    def do_move_buffers(
        self,
        starting_layout: dlt.UnpackedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        buffer_size: SSAValue,
        new_number_of_elems: SSAValue,
        old_number_of_elems: SSAValue,
        idx_buffer_ptr: SSAValue,
        data_buffer_ptr: SSAValue,
        index: SSAValue,
    ) -> tuple[list[Operation], SSAValue, SSAValue]:
        assert isinstance(buffer_size.type, IndexType)
        assert isinstance(idx_buffer_ptr.type, llvm.LLVMPointerType)
        assert isinstance(data_buffer_ptr.type, llvm.LLVMPointerType)
        assert isinstance(index.type, IndexType)
        ops = []
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        ops.append(zero_op)

        # Set up for lots of memory copies to move the sparse index tuples and child data into their new larger buffers
        # Starting with Index tuples
        index_tuple_size_ops, (index_tuple_size,) = (
            _get_accepted_type_size(i64).keep(1)
            * len(starting_layout.dimensions)
        ).output()
        ops.extend(index_tuple_size_ops)

        idx_copy_ops, idx_elem_ptr = self.copy_to_new_buffers(
            idx_buffer_ptr,
            idx_buffer_ptr,
            index_tuple_size,
            index,
            buffer_size,
            old_number_of_elems,
            zero_op.result,
            shuffle_instead=True,
        )
        ops.extend(idx_copy_ops)

        data_elem_size_ops, (data_elem_size, data_elem_size_extra) = (
            self.semantics.get_size(starting_layout.child, extent_resolver).output()
        )
        ops.extend(data_elem_size_ops)

        data_copy_ops, data_elem_ptr = self.copy_to_new_buffers(
            data_buffer_ptr,
            data_buffer_ptr,
            data_elem_size,
            index,
            buffer_size,
            old_number_of_elems,
            data_elem_size_extra,
            shuffle_instead=True,
        )
        ops.extend(data_copy_ops)

        return ops, idx_elem_ptr, data_elem_ptr


class SeparatedCOOSemantics(COOSemantics[dlt.SeparatedCOOLayoutAttr]):

    def _get_buffer_ptrs(
        self,
        layout: dlt.SeparatedCOOLayoutAttr,
        input_ptr: SSAValue,
        get_capacity: bool = False,
    ) -> tuple[
        list[Operation],
        tuple[tuple[SSAValue, ...], SSAValue, SSAValue | None]
        | tuple[tuple[SSAValue, ...], SSAValue],
    ]:
        ops = []
        idxs = []
        for i, _ in enumerate(layout.dimensions):
            idx_buffer_ptr_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
            ops.append(idx_buffer_ptr_op)
            idxs.append(idx_buffer_ptr_op.dereferenced_value)
            ptr_size = _get_ptr_size().sum()
            ptr_inc_ops, input_ptr = ptr_size.add_to_llvm_pointer(input_ptr)
            ops.extend(ptr_inc_ops)

        data_buffer_ptr_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        data_buffer_ptr = data_buffer_ptr_op.dereferenced_value
        ops.append(data_buffer_ptr_op)

        if not get_capacity:
            return ops, (tuple(idxs), data_buffer_ptr)
        if not layout.is_buffered():
            return ops, (tuple(idxs), data_buffer_ptr, None)

        ptr_size = _get_ptr_size().sum()
        ptr_inc_ops, capacity_ptr = ptr_size.add_to_llvm_pointer(input_ptr)
        ops.extend(ptr_inc_ops)
        capacity_op = llvm.LoadOp(capacity_ptr, i64)
        capacity_i64 = capacity_op.dereferenced_value
        ops.append(capacity_op)
        convert_capacity_ops, capacity = index_convert(capacity_i64, IndexType())
        ops.extend(convert_capacity_ops)
        return ops, (tuple(idxs), data_buffer_ptr, capacity)

    def _set_buffer_ptrs(
        self,
        layout: dlt.SeparatedCOOLayoutAttr,
        input_ptr: SSAValue,
        new_idx_buffer_ptrs: tuple[SSAValue, ...],
        new_data_buffer_ptr: SSAValue,
        capacity: SSAValue = None,
    ) -> list[Operation]:
        ops = []
        if layout.is_buffered():
            llvm_type = llvm.LLVMStructType.from_type_list(
                ([llvm.LLVMPointerType.opaque()] * (len(layout.dimensions) + 1))+[i64])
        else:
            llvm_type = llvm.LLVMStructType.from_type_list(
                [llvm.LLVMPointerType.opaque()]*(len(layout.dimensions)+1))
        for i, idx_buffer_ptr in enumerate(new_idx_buffer_ptrs):
            gep_op = llvm.GEPOp(input_ptr, [0,i], pointee_type=llvm_type)
            ops.append(gep_op)
            idx_buffer_ptr_op = llvm.StoreOp(idx_buffer_ptr, gep_op.result)
            ops.append(idx_buffer_ptr_op)

        gep_op = llvm.GEPOp(input_ptr, [0,len(layout.dimensions) + 0], pointee_type=llvm_type)
        ops.append(gep_op)
        data_buffer_ptr_op = llvm.StoreOp(new_data_buffer_ptr, gep_op.result)
        ops.append(data_buffer_ptr_op)

        if capacity is None:
            assert not layout.is_buffered()
            return ops
        assert layout.is_buffered()
        gep_op = llvm.GEPOp(input_ptr, [0, len(layout.dimensions) + 1], pointee_type=llvm_type)
        ops.append(gep_op)
        convert_capacity_ops, capacity_i64 = index_convert(capacity, i64)
        ops.extend(convert_capacity_ops)
        capacity_op = llvm.StoreOp(capacity_i64, gep_op.result)
        ops.append(capacity_op)
        return ops

    def get_size(
        self, layout: dlt.SeparatedCOOLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        # [*index_buffer per dimension, data_buffer]
        if layout.is_buffered():
            return ((_get_ptr_size().sum() * (len(layout.dimensions)+1)) + _get_accepted_type_size(i64).sum()).extend(0)
        else:
            return (_get_ptr_size().sum() * (len(layout.dimensions)+1)).extend(0)

    def get_getter_for(
        self,
        starting_layout: dlt.SeparatedCOOLayoutAttr,
        get_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index_range: SSAValue,
        index_found: bool | SSAValue,
    ) -> tuple[list[Operation], SSAValue, bool | SSAValue]:
        ops = []

        extract_ops, start, end = _extract_indices_from_index_range(index_range, as_index=True)
        ops.extend(extract_ops)

        extract_ptr_ops, (idx_buffers, data_buffer_ptr) = self._get_buffer_ptrs(
            starting_layout, input_ptr
        )
        ops.extend(extract_ptr_ops)

        search_loop_ops, index_is_found, index = self.search_to_find_index(
            starting_layout,
            start,
            end,
            idx_buffers,
            dim_mapping,
        )
        ops.extend(search_loop_ops)
        # If index is found we can use it, else we return an appropriate zero element
        if_found_true = []
        data_ptr_ops, child_elem_ptr = (
            NumericResult.from_ssa(index)
            * self.semantics.get_size(starting_layout.child, extent_resolver).keep(1)
        ).add_to_llvm_pointer(data_buffer_ptr)
        if_found_true.extend(data_ptr_ops)

        child_dim_mapping = {
            dim: val
            for dim, val in dim_mapping.items()
            if dim not in starting_layout.dimensions
        }
        child_members = members
        child_ops, child_res, child_found = self.semantics.get_getter_for(
            starting_layout.child,
            get_type,
            child_members,
            child_dim_mapping,
            extent_resolver,
            child_elem_ptr,
        )
        if_found_true.extend(child_ops)
        bool_ops, child_found = _make_bool_ssa(child_found)
        if_found_true.extend(bool_ops)

        if_found_true.append(scf.Yield(child_res, child_found))

        if_found_false = []

        zero_result_ops, zero_result = _get_packed_zero_for_accepted_type(get_type)
        if_found_false.extend(zero_result_ops)

        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        if_found_false.append(false_op)
        if_found_false.append(scf.Yield(zero_result, false_op.result))

        if_found_op = scf.If(
            index_is_found, [get_type, IntegerType(1)], if_found_true, if_found_false
        )
        ops.append(if_found_op)
        return ops, if_found_op.output[0], if_found_op.output[1]

    def get_setter_for(
        self,
        starting_layout: dlt.SeparatedCOOLayoutAttr,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index_range: SSAValue,
        index_range_found: bool | SSAValue,
        ensure_space_func: EnsureSpaceFunc,
        direct_iterate_func: LinearIterCallbackFunc,
        direct_members: set[dlt.MemberAttr],
        direct_dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> list[Operation]:

        # get the members and dim_mapping that will be needed by the recursive child get_setter_for
        child_dim_mapping = {
            dim: val
            for dim, val in dim_mapping.items()
            if dim not in starting_layout.dimensions
        }
        child_members = members

        ops = []
        zero_cmp_ops, is_non_zero = _compare_is_non_zero(set_val)
        ops.extend(zero_cmp_ops)
        index_range_found_ops, index_range_found = _make_bool_ssa(index_range_found)
        ops.extend(index_range_found_ops)
        must_start_set_cond_op = arith.OrI(index_range_found, is_non_zero)
        ops.append(must_start_set_cond_op)

        if_must_start_set_ops = []

        # 3) if the direct child doesn't have a range, we must ask it to make one.
        if_index_range_found_true = [scf.Yield(index_range)]
        if_index_range_found_false = []

        def current_fill_value_getter() -> tuple[list[Operation], SSAValue]:
            f_ops, zero_idx_range = _get_packed_zero_for_accepted_type(
                dlt.IndexRangeType()
            )
            return f_ops, zero_idx_range

        ensure_space_ops, new_index_range = ensure_space_func(current_fill_value_getter)

        if_index_range_found_false.extend(ensure_space_ops)
        if_index_range_found_false.append(scf.Yield(new_index_range))

        if_index_range_found_op = scf.If(
            index_range_found,
            [dlt.IndexRangeType()],
            if_index_range_found_true,
            if_index_range_found_false,
        )
        if_must_start_set_ops.append(if_index_range_found_op)
        true_index_range = if_index_range_found_op.output[0]

        # 4) with the index-range, we must look for the matching element in the sparse index buffer
        extract_ops, start, end = _extract_indices_from_index_range(true_index_range, as_index=True)
        if_must_start_set_ops.extend(extract_ops)

        extract_ptr_ops, (idx_buffer_ptrs, data_buffer_ptr, buffer_size) = (
            self._get_buffer_ptrs(starting_layout, input_ptr, get_capacity=True)
        )
        if_must_start_set_ops.extend(extract_ptr_ops)

        search_loop_ops, index_is_found, index = self.search_to_find_index(
            starting_layout,
            start,
            end,
            idx_buffer_ptrs,
            dim_mapping,
        )
        if_must_start_set_ops.extend(search_loop_ops)
        # index_is_found iff we find an index tuple that matches the sparse dimension values from dim_mapping index
        # will be the index offset in the index-buffer (and data-buffer) of the match if there is one, else it will
        # be the index it was expected to be (i.e. the index where the loop stopped because either the range ran out,
        # or the index tuples got larger than the values in dim_mapping)

        must_do_set_cond_op = arith.OrI(index_is_found, is_non_zero)
        if_must_start_set_ops.append(must_do_set_cond_op)
        if_must_do_set_ops = []

        # 6) If index is found we can get the ptr for the child, else we need to reallocate and move lots of indices
        # around
        if_found_true = []

        data_ptr_ops, child_elem_ptr = (
            NumericResult.from_ssa(index)
            * self.semantics.get_size(starting_layout.child, extent_resolver).keep(1)
        ).add_to_llvm_pointer(data_buffer_ptr)
        if_found_true.extend(data_ptr_ops)

        # and finally we can set the value in this child element
        child_setter_ops = self.semantics.get_setter_for(
            starting_layout.child,
            set_val,
            set(child_members),
            dict(child_dim_mapping),
            extent_resolver,
            child_elem_ptr,
        )
        if_found_true.extend(child_setter_ops)

        if_found_true.append(scf.Yield())

        # If the value is not found we need to increment the indices of everything larger than this in the direct
        # data, then allocate new buffers with extra size for the new element, and mem copy the data across leaving
        # room for the new element. Then we init the new element and can continue setting from there.
        if_found_false = []

        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        if_found_false.extend([true_op, false_op, zero_op])

        increment_indices_callback = self.SparseIndexIncrementCallback(
            dict(direct_dim_mapping),
            set(direct_members),
            false_op.result,
            zero_op.result,
            zero_op.result,
        )

        increment_ops, increment_results, _increment_exited = direct_iterate_func(
            increment_indices_callback, True
        )

        if_found_false.extend(increment_ops)
        # index_found = increment_results[0] - unused as this is for iter-callback communication - not a result
        last_index = increment_results[1]
        # last_index will be the number of non-zero elements now in the sparse buffer
        # found_index = increment_results[2]
        # found_index will be the index of the element we care about - the start of the range in the direct child
        # This is not actually required and should be removed

        if buffer_size is not None:
            # if_found_false.append(printf.PrintFormatOp("last_index {}   buffer_size {}", last_index, buffer_size))

            cmp_buffer_size_op = arith.Cmpi(last_index, buffer_size, "ugt")
            if_found_false.append(cmp_buffer_size_op)

            # if we need new buffers - handle it
            if_needs_new_buffers = []
            buffer_scaler = starting_layout.buffer_scaler.data
            buffer_scaler_const = arith.Constant(
                IntegerAttr(abs(buffer_scaler), IndexType())
            )
            if_needs_new_buffers.append(buffer_scaler_const)
            if buffer_scaler > 0:
                # multiply
                mul_op = arith.Muli(buffer_size, buffer_scaler_const)
                if_needs_new_buffers.append(mul_op)
                new_size_candidate = mul_op.result
                # if_needs_new_buffers.append(printf.PrintFormatOp("candidate {}   last index {}", new_size_candidate, last_index))
            else:
                # add
                add_op = arith.Addi(buffer_size, buffer_scaler_const)
                if_needs_new_buffers.append(add_op)
                new_size_candidate = add_op.result
                # if_needs_new_buffers.append(
                #     printf.PrintFormatOp("candidate {}   last index {}", new_size_candidate, last_index))
            new_size_op = arith.MaxUI(new_size_candidate, last_index)
            if_needs_new_buffers.append(new_size_op)
            new_size = new_size_op.result
            (
                do_new_buffers_ops,
                new_idx_buffer_ptrs,
                new_data_buffer_ptr,
                idx_elem_ptrs,
                data_elem_ptr,
            ) = self.do_new_buffers(
                starting_layout,
                extent_resolver,
                new_size,
                idx_buffer_ptrs,
                data_buffer_ptr,
                buffer_size,
                index,
            )
            if_needs_new_buffers.extend(do_new_buffers_ops)

            initialiser = SingletonInitialiser(
                set_val, set(child_members), dict(child_dim_mapping)
            )
            set_up_child_ops = self.set_up_new_child(
                starting_layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                idx_elem_ptrs,
                data_elem_ptr,
            )
            if_needs_new_buffers.extend(set_up_child_ops)

            if_needs_new_buffers.extend(
                self._set_buffer_ptrs(
                    starting_layout,
                    input_ptr,
                    new_idx_buffer_ptrs,
                    new_data_buffer_ptr,
                    new_size,
                )
            )
            if_needs_new_buffers.append(scf.Yield())

            # alternatively we don't need new buffers - just need to move the data in the buffers we have
            if_needs_moved_buffers = []
            calc_old_elems_ops, (elems_to_copy,) = (
                NumericResult.from_ssa(last_index) - 1
            ).output()
            if_needs_moved_buffers.extend(calc_old_elems_ops)
            (
                do_move_buffers_ops,
                idx_elem_ptrs,
                data_elem_ptr,
            ) = self.do_move_buffers(
                starting_layout,
                extent_resolver,
                buffer_size,
                last_index,
                elems_to_copy,
                idx_buffer_ptrs,
                data_buffer_ptr,
                index,
            )
            if_needs_moved_buffers.extend(do_move_buffers_ops)
            initialiser = SingletonInitialiser(
                set_val, set(child_members), dict(child_dim_mapping)
            )
            set_up_child_ops = self.set_up_new_child(
                starting_layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                idx_elem_ptrs,
                data_elem_ptr,
            )
            if_needs_moved_buffers.extend(set_up_child_ops)
            if_needs_moved_buffers.append(scf.Yield())

            if_new_buffers_op = scf.If(
                cmp_buffer_size_op.result,
                [],
                if_needs_new_buffers,
                if_needs_moved_buffers,
            )
            if_found_false.append(if_new_buffers_op)
        else:
            # if it's not buffered then we know we just need to always do new buffers at the size of last_index
            # and we know the old buffer size was just one element less than the new last_index
            new_size = last_index
            calc_old_size_ops, (old_buffer_size,) = (
                NumericResult.from_ssa(last_index) - 1
            ).output()
            if_found_false.extend(calc_old_size_ops)
            (
                do_new_buffers_ops,
                new_idx_buffer_ptrs,
                new_data_buffer_ptr,
                idx_elem_ptrs,
                data_elem_ptr,
            ) = self.do_new_buffers(
                starting_layout,
                extent_resolver,
                new_size,
                idx_buffer_ptrs,
                data_buffer_ptr,
                old_buffer_size,
                index,
            )
            if_found_false.extend(do_new_buffers_ops)
            # New we fill in the new index tuple and data elem

            initialiser = SingletonInitialiser(
                set_val, set(child_members), dict(child_dim_mapping)
            )
            set_up_child_ops = self.set_up_new_child(
                starting_layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                idx_elem_ptrs,
                data_elem_ptr,
            )
            if_found_false.extend(set_up_child_ops)

            # The new buffers need to be put in the indexing struct and the old ones freed
            if_found_false.extend(
                self._set_buffer_ptrs(
                    starting_layout, input_ptr, new_idx_buffer_ptrs, new_data_buffer_ptr
                )
            )

        if_found_false.append(scf.Yield())

        if_found_op = scf.If(
            index_is_found,
            [],
            if_found_true,
            if_found_false,
        )
        if_must_do_set_ops.append(if_found_op)
        if_must_do_set_ops.append(scf.Yield())
        if_must_do_set_op = scf.If(must_do_set_cond_op.result, [], if_must_do_set_ops)
        if_must_start_set_ops.append(if_must_do_set_op)

        if_must_start_set_ops.append(scf.Yield())
        if_must_do_set_op = scf.If(
            must_start_set_cond_op.result, [], if_must_start_set_ops
        )
        ops.append(if_must_do_set_op)
        return ops

    def ensure_space(
        self,
        layout: dlt.SeparatedCOOLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: FillValueGetter,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        index_range: SSAValue,
        direct_iterate_func: LinearIterCallbackFunc,
        direct_members: set[dlt.MemberAttr],
        direct_dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> tuple[list[Operation], SSAValue]:

        ops = []
        extract_ops, start, end = _extract_indices_from_index_range(index_range, as_index=True)
        ops.extend(extract_ops)

        extract_ptr_ops, (idx_buffer_ptrs, data_buffer_ptr, buffer_size) = (
            self._get_buffer_ptrs(layout, input_ptr, get_capacity=True)
        )
        ops.extend(extract_ptr_ops)

        search_loop_ops, index_is_found, index = self.search_to_find_index(
            layout,
            start,
            end,
            idx_buffer_ptrs,
            dim_mapping,
        )
        ops.extend(search_loop_ops)

        # get the members and dim_mapping that will eventually be needed by the recursive child get_setter_for
        child_dim_mapping = {
            dim: val for dim, val in dim_mapping.items() if dim not in layout.dimensions
        }
        child_members = members

        data_elem_size_ops, (data_elem_size, data_elem_size_extra) = (
            self.semantics.get_size(layout.child, extent_resolver).output()
        )
        ops.extend(data_elem_size_ops)

        child_fill_value_getter = self._get_fill_value_getter(
            layout.child,
            extent_resolver,
            fill_value_getter,
            base_type,
            data_buffer_ptr,
            data_elem_size,
            index,
        )

        # If index_is_found we can get the ptr to the child, else we need to reallocate and move lots of indices around
        if_found_true = []

        data_ptr_ops, child_elem_ptr = (
            NumericResult.from_ssa(index) * NumericResult.from_ssa(data_elem_size)
        ).add_to_llvm_pointer(data_buffer_ptr)
        if_found_true.extend(data_ptr_ops)

        # and finally we can set the ensure_space in this new child element
        child_ops, child_result = self.semantics.ensure_space(
            layout.child,
            base_type,
            child_members,
            child_dim_mapping,
            child_fill_value_getter,
            extent_resolver,
            child_elem_ptr,
        )
        if_found_true.extend(child_ops)
        if_found_true.append(scf.Yield(child_result))

        # If the value is not found we need to increment the indices of everything larger than this in the direct
        # data, then allocate new buffers with extra size for the new element, and mem copy the data across leaving
        # room for the new element. Then we can init the new element and can continue setting from there.
        if_found_false = []

        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        if_found_false.extend([true_op, false_op, zero_op, one_op])

        increment_indices_callback = self.SparseIndexIncrementCallback(
            dict(direct_dim_mapping),
            set(direct_members),
            false_op.result,
            zero_op.result,
            zero_op.result,
        )
        increment_ops, increment_results, _increment_exited = direct_iterate_func(
            increment_indices_callback, True
        )

        if_found_false.extend(increment_ops)
        # index_found = increment_results[0]
        # index_found (bool) is not used here as it is an internal check for inside the callback.
        last_index = increment_results[1]
        # last_index will be the number of non-zero elements now in the sparse buffer
        # found_index = increment_results[2]
        # found_index will be the index of the element we care about

        if buffer_size is not None:

            cmp_buffer_size_op = arith.Cmpi(last_index, buffer_size, "ugt")
            need_bigger_buffers = cmp_buffer_size_op.result
            if_found_false.append(cmp_buffer_size_op)

            # if we need new buffers - handle it
            if_needs_new_buffers = []
            get_fill_value_ops, child_fill_value = child_fill_value_getter()
            if_needs_new_buffers.extend(get_fill_value_ops)

            buffer_scaler = layout.buffer_scaler.data
            buffer_scaler_const = arith.Constant(
                IntegerAttr(abs(buffer_scaler), IndexType())
            )
            if_needs_new_buffers.append(buffer_scaler_const)
            if buffer_scaler > 0:
                # multiply
                mul_op = arith.Muli(buffer_size, buffer_scaler_const)
                if_needs_new_buffers.append(mul_op)
                new_size_candidate = mul_op.result
                # if_needs_new_buffers.append(printf.PrintFormatOp("candidate {}   last index {}", new_size_candidate, last_index))
            else:
                # add
                add_op = arith.Addi(buffer_size, buffer_scaler_const)
                if_needs_new_buffers.append(add_op)
                new_size_candidate = add_op.result
                # if_needs_new_buffers.append(
                #     printf.PrintFormatOp("candidate {}   last index {}", new_size_candidate, last_index))
            new_size_op = arith.MaxUI(new_size_candidate, last_index)
            if_needs_new_buffers.append(new_size_op)
            new_size = new_size_op.result
            (
                do_new_buffers_ops,
                new_idx_buffer_ptrs,
                new_data_buffer_ptr,
                new_idx_elem_ptrs,
                new_data_elem_ptr,
            ) = self.do_new_buffers(
                layout,
                extent_resolver,
                new_size,
                idx_buffer_ptrs,
                data_buffer_ptr,
                buffer_size,
                index,
            )
            if_needs_new_buffers.extend(do_new_buffers_ops)

            # Now we fill in the new index tuple and data elem

            initialiser = self.WriteEmptyRangeInitialiser(
                child_fill_value, child_members, child_dim_mapping
            )
            set_up_child_ops = self.set_up_new_child(
                layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                new_idx_elem_ptrs,
                new_data_elem_ptr,
            )
            if_needs_new_buffers.extend(set_up_child_ops)

            if_needs_new_buffers.extend(
                self._set_buffer_ptrs(
                    layout,
                    input_ptr,
                    new_idx_buffer_ptrs,
                    new_data_buffer_ptr,
                    new_size,
                )
            )
            if_needs_new_buffers.append(scf.Yield(child_fill_value))

            # alternatively we don't need new buffers - just need to move the data in the buffers we have
            if_needs_moved_buffers = []
            calc_old_elems_ops, (elems_to_copy,) = (
                NumericResult.from_ssa(last_index) - 1
            ).output()
            if_needs_moved_buffers.extend(calc_old_elems_ops)
            (
                do_move_buffers_ops,
                new_idx_elem_ptrs,
                new_data_elem_ptr,
            ) = self.do_move_buffers(
                layout,
                extent_resolver,
                buffer_size,
                last_index,
                elems_to_copy,
                idx_buffer_ptrs,
                data_buffer_ptr,
                index,
            )
            if_needs_moved_buffers.extend(do_move_buffers_ops)

            # Now we fill in the new index tuple and data elem
            get_fill_value_ops, child_fill_value = child_fill_value_getter()
            if_needs_moved_buffers.extend(get_fill_value_ops)
            initialiser = self.WriteEmptyRangeInitialiser(
                child_fill_value, child_members, child_dim_mapping
            )
            set_up_child_ops = self.set_up_new_child(
                layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                new_idx_elem_ptrs,
                new_data_elem_ptr,
            )
            if_needs_moved_buffers.extend(set_up_child_ops)

            if_needs_moved_buffers.append(scf.Yield(child_fill_value))

            if_new_buffers_op = scf.If(
                need_bigger_buffers,
                [base_type],
                if_needs_new_buffers,
                if_needs_moved_buffers,
            )
            if_found_false.append(if_new_buffers_op)
            child_fill_value = if_new_buffers_op.results[0]
        else:
            # if it's not buffered then we know we just need to always do new buffers at the size of last_index
            # and we know the old buffer size was just one element less than the new last_index
            get_fill_value_ops, child_fill_value = child_fill_value_getter()
            if_found_false.extend(get_fill_value_ops)

            new_size = last_index
            calc_old_size_ops, (old_buffer_size,) = (
                NumericResult.from_ssa(last_index) - 1
            ).output()
            if_found_false.extend(calc_old_size_ops)
            (
                do_new_buffers_ops,
                new_idx_buffer_ptrs,
                new_data_buffer_ptr,
                new_idx_elem_ptrs,
                new_data_elem_ptr,
            ) = self.do_new_buffers(
                layout,
                extent_resolver,
                new_size,
                idx_buffer_ptrs,
                data_buffer_ptr,
                old_buffer_size,
                index,
            )
            if_found_false.extend(do_new_buffers_ops)

            # Now we fill in the new index tuple and data elem
            initialiser = self.WriteEmptyRangeInitialiser(
                child_fill_value, child_members, child_dim_mapping
            )
            set_up_child_ops = self.set_up_new_child(
                layout,
                extent_resolver,
                dim_mapping,
                initialiser,
                new_idx_elem_ptrs,
                new_data_elem_ptr,
            )
            if_found_false.extend(set_up_child_ops)

            if_found_false.extend(
                self._set_buffer_ptrs(
                    layout, input_ptr, new_idx_buffer_ptrs, new_data_buffer_ptr
                )
            )

        if_found_false.append(scf.Yield(child_fill_value))

        if_found_op = scf.If(
            index_is_found,
            [dlt.IndexRangeType()],
            if_found_true,
            if_found_false,
        )
        ops.append(if_found_op)

        return ops, if_found_op.output[0]

    def init_layout(
        self,
        layout: dlt.SeparatedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: Initialiser,
        init_callback: Callback,
        callback_args: list[SSAValue],
        is_last_element: bool | SSAValue,
        direct_init_func: LinearIterCallbackFunc,
        direct_iter_func: LinearIterCallbackFunc,
    ) -> tuple[list[Operation], list[SSAValue]]:
        assert len(callback_args) == len(init_callback.initial_iter_args())
        ops = []

        running_total_initial_zero = arith.Constant(IntegerAttr(0, IndexType()))
        ops.append(running_total_initial_zero)

        set_idx_callback = self.SparseIndexWritingCallback(
            layout.child.contents_type,
            running_total_initial_zero.result,
            initial_values,
            list(layout.dimensions),
        )
        direct_init_ops, total_size_iter_args, _exited_early = direct_init_func(
            set_idx_callback, True
        )
        ops.extend(direct_init_ops)
        total_size = total_size_iter_args[0]

        malloc_ops, idx_buffer_ptrs, data_buffer_ptr = self.malloc_buffers(
            layout,
            total_size,
            self.semantics.get_size(layout.child, extent_resolver),
        )
        ops.extend(malloc_ops)

        if layout.is_buffered():
            buffer_size = total_size
        else:
            buffer_size = None

        ops.extend(
            self._set_buffer_ptrs(
                layout, input_ptr, idx_buffer_ptrs, data_buffer_ptr, buffer_size
            )
        )

        data_element_size_ops, (data_element_size,) = (
            self.semantics.get_size(layout.child, extent_resolver).keep(1).output()
        )
        ops.extend(data_element_size_ops)

        set_data_callback = self.SparseDataWritingCallback(
            self.semantics,
            layout,
            idx_buffer_ptrs,
            data_buffer_ptr,
            running_total_initial_zero.result,
            total_size,
            initial_values,
            data_element_size,
            init_callback,
            callback_args,
        )
        set_data_ops, iter_args, _exited = direct_iter_func(
            set_data_callback, is_last_element
        )
        ops.extend(set_data_ops)
        callback_iter_args = iter_args[len(idx_buffer_ptrs)+2:]
        assert len(callback_args) == len(callback_iter_args)
        return ops, callback_iter_args

    def dealloc_layout(
        self,
        layout: dlt.SeparatedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        direct_iter_func: LinearIterCallbackFunc,
    ) -> list[Operation]:

        ops = []
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)

        extract_ptr_ops, (idx_buffer_ptrs, data_buffer_ptr) = self._get_buffer_ptrs(
            layout, input_ptr
        )
        ops.extend(extract_ptr_ops)

        data_elem_size_ops, (data_elem_size,) = (
            self.semantics.get_size(layout.child, extent_resolver).keep(1).output()
        )
        ops.extend(data_elem_size_ops)

        dealloc_callback = COOSemantics.DeallocCallback(
            self.semantics,
            layout,
            extent_resolver,
            data_buffer_ptr,
            data_elem_size,
        )
        iter_ops, _, _ = direct_iter_func(dealloc_callback, True)
        ops.extend(iter_ops)
        for idx_buffer_ptr in idx_buffer_ptrs:
            # ops.append(
            #     printf.PrintFormatOp("5204 free idx {}", idx_buffer_ptr))
            ops.append(llvm.CallOp("free", idx_buffer_ptr))
            if self.semantics.print_memory_calls:
                ops.append(printf.PrintFormatOp("# called free({})", idx_buffer_ptr))
                ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
                ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
                ops.append(llvm.CallOp("fflush", nullptr.output))
        # ops.append(
        #     printf.PrintFormatOp("5207 free data {}", data_buffer_ptr))
        ops.append(llvm.CallOp("free", data_buffer_ptr))
        if self.semantics.print_memory_calls:
            ops.append(printf.PrintFormatOp("# called free({})", data_buffer_ptr))
            ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
            ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
            ops.append(llvm.CallOp("fflush", nullptr.output))

        return ops

    def linear_iterate(
        self,
        layout: dlt.SeparatedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: Callback,
        callback_args: list[SSAValue],
        is_last_element: bool | SSAValue,
        direct_iter_func: LinearIterCallbackFunc,
        reversed_direction: bool = False,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        ops = []
        extract_ptr_ops, (idx_buffer_ptrs, data_buffer_ptr) = self._get_buffer_ptrs(
            layout, input_ptr
        )
        ops.extend(extract_ptr_ops)

        data_elem_size_ops, (data_elem_size,) = (
            self.semantics.get_size(layout.child, extent_resolver).keep(1).output()
        )
        ops.extend(data_elem_size_ops)

        linear_iter_callback: Callback = self.SparseLinearIterCallback(
            self.semantics,
            layout,
            idx_buffer_ptrs,
            data_buffer_ptr,
            data_elem_size,
            callback,
            callback_args,
            is_last_element,
            reversed_direction,
        )
        child_ops, child_iter_args, child_exited_early = direct_iter_func(
            linear_iter_callback, True
        )
        ops.extend(child_ops)
        return ops, child_iter_args, child_exited_early

    def make_sparse_loop_callback_for(
        self,
        starting_layout: dlt.SeparatedCOOLayoutAttr,
        ending_layout: dlt.DirectLayout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        callback: LoopCallback,
        callback_args: list[SSAValue],
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        dims_to_loop: set[dlt.DimensionAttr],
    ) -> tuple[list[Operation], LoopCallback]:
        ops = []
        extract_ptr_ops, (idx_buffer_ptrs, data_buffer_ptr) = self._get_buffer_ptrs(
            starting_layout, input_ptr
        )
        ops.extend(extract_ptr_ops)

        data_elem_size_ops, (data_elem_size,) = (
            self.semantics.get_size(starting_layout.child, extent_resolver)
            .keep(1)
            .output()
        )
        ops.extend(data_elem_size_ops)

        new_callback = self.SparseMakeLoopCallback(
            self.semantics,
            starting_layout,
            ending_layout,
            extent_resolver,
            members,
            dim_mapping,
            dims_to_loop,
            idx_buffer_ptrs,
            data_buffer_ptr,
            data_elem_size,
            callback,
            callback_args,
        )
        return ops, new_callback

    @staticmethod
    def loop_to_find_index(
        scoo_layout: dlt.SeparatedCOOLayoutAttr,
        start: SSAValue,
        end: SSAValue,
        index_buffer_ptrs: tuple[SSAValue],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> tuple[
        list[Operation], SSAValue, SSAValue
    ]:  # ops, index_is_found(bool), index(IndexType)
        num_dims = len(scoo_layout.dimensions)
        assert len(index_buffer_ptrs) == num_dims

        # Do {
        #   If in bounds {
        #       check if index is found,
        #       or if we've found an index larger,
        #       return (found or over), found # exit-loop?, index-found?
        #   } else {
        #       return true , false # exit-loop, index-not-found
        # } while (not exit-loop) {
        #   increment index
        # }

        before = Block()
        before_index = before.insert_arg(IndexType(), 0)

        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        before.add_ops([false_op, true_op])

        before_index_in_range = arith.Cmpi(before_index, end, "ult")
        before.add_op(before_index_in_range)

        # IF index_in_range: go through the sparse dimensions' indices at current index to check for: 1) if they are
        # equal to the indices we're looking for; 2) if they are lexicographically larger than the indices we're
        # looking for, i.e. we can stop looking because we store sparse values sorted by ascending lexicographic sparse
        # indices
        if_in_range_true = []
        running_is_over = false_op.result
        running_is_found = true_op.result

        for i, (dim, idx_buffer_ptr, idx_buffer_type) in enumerate(
            zip(
                scoo_layout.dimensions,
                index_buffer_ptrs,
                scoo_layout.index_buffer_types,
            )
        ):
            before_index_convert_ops, before_index_i64 = index_convert(before_index, i64)
            if_in_range_true.extend(before_index_convert_ops)
            idx_ptr_arith_op = llvm.GEPOp(
                idx_buffer_ptr,
                [llvm.GEP_USE_SSA_VAL],
                [before_index_i64],
                pointee_type=idx_buffer_type,
            )
            if_in_range_true.append(idx_ptr_arith_op)
            inner_load_stored_index_op = llvm.LoadOp(
                idx_ptr_arith_op.result, idx_buffer_type
            )
            if_in_range_true.append(inner_load_stored_index_op)
            convert_inner_idx_ops, inner_idx = index_convert(inner_load_stored_index_op.dereferenced_value, IndexType())
            if_in_range_true.extend(convert_inner_idx_ops)
            if idx_buffer_type.width.data > builtin.i64.width.data:
                raise NotImplementedError(
                    f"{idx_buffer_type} not supported as Separated COO index buffer type"
                )

            get_index_ops, (get_index,) = dim_mapping[dim].get().output()
            if_in_range_true.extend(get_index_ops)

            inner_is_over_cmp = arith.Cmpi(inner_idx, get_index, "ugt")
            if_in_range_true.append(inner_is_over_cmp)
            inner_is_found_cmp = arith.Cmpi(inner_idx, get_index, "eq")
            if_in_range_true.append(inner_is_found_cmp)

            inner_is_now_over = arith.AndI(running_is_found, inner_is_over_cmp)
            if_in_range_true.append(inner_is_now_over)
            inner_is_greater_reduce = arith.OrI(running_is_over, inner_is_now_over)
            if_in_range_true.append(inner_is_greater_reduce)

            inner_is_found_reduce = arith.AndI(running_is_found, inner_is_found_cmp)
            if_in_range_true.append(inner_is_found_reduce)

            running_is_found = inner_is_found_reduce.result
            running_is_over = inner_is_greater_reduce.result

        exit_loop_cond = arith.OrI(running_is_found, running_is_over)
        if_in_range_true.append(exit_loop_cond)

        if_in_range_true.append(scf.Yield(exit_loop_cond.result, running_is_found))

        if_in_range_false = [scf.Yield(true_op.result, false_op.result)]

        if_in_range_op = scf.If(
            before_index_in_range,
            [IntegerType(1), IntegerType(1)],
            if_in_range_true,
            if_in_range_false,
        )
        before.add_op(if_in_range_op)
        before_exit_loop = if_in_range_op.output[0]
        before_is_found = if_in_range_op.output[1]

        before_continue_loop_op = arith.Cmpi(before_exit_loop, false_op, "eq")
        before.add_op(before_continue_loop_op)

        before.add_op(
            scf.Condition(before_continue_loop_op.result, before_is_found, before_index)
        )

        after = Block()
        after.insert_arg(IntegerType(1), 0)  # after_is_found =
        after_index = after.insert_arg(IndexType(), 1)
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        after.add_op(one_op)

        new_index_op = arith.Addi(after_index, one_op.result)
        after.add_op(new_index_op)
        after.add_op(scf.Yield(new_index_op.result))

        # While op, index <= start, returns (found, index)
        while_op = scf.While([start], [IntegerType(1), IndexType()], [before], [after])

        index_is_found = while_op.res[0]
        index = while_op.res[1]

        return [while_op], index_is_found, index

    @staticmethod
    def loop_back_to_find_index(
            scoo_layout: dlt.SeparatedCOOLayoutAttr,
            start: SSAValue,
            end: SSAValue,
            index_buffer_ptrs: tuple[SSAValue],
            dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> tuple[
        list[Operation], SSAValue, SSAValue
    ]:  # ops, index_is_found(bool), index(IndexType)
        num_dims = len(scoo_layout.dimensions)
        assert len(index_buffer_ptrs) == num_dims
        # found = False
        # under = False
        # idx = end
        # while(!found and !under and start < idx) {
        #   next_idx = idx - 1
        #   check if arr[next_index] is found or under
        #   out_idex = idx if under else next_idx
        #   yield (found, under, next_idx)
        # }
        ops = []
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.extend([one_op, false_op, true_op])

        init_found = false_op.result
        init_under = false_op.result
        init_index = end

        before_block = Block(arg_types=(IntegerType(1), IntegerType(1), IndexType()))
        before_found, before_under, before_index = before_block.args
        exit_cond_op = arith.OrI(before_found, before_under)
        idx_cmp_op = arith.Cmpi(start, before_index, "ult")
        continue_op = arith.Select(exit_cond_op.result, false_op.result, idx_cmp_op.result)
        before_block.add_ops([exit_cond_op, idx_cmp_op, continue_op])
        before_condition_op = scf.Condition(continue_op.result, before_found, before_index)
        before_block.add_op(before_condition_op)

        after_block = Block(arg_types=[a.type for a in before_condition_op.arguments])
        _, after_index = after_block.args

        after_next_index_op = arith.Subi(after_index, one_op.result)
        after_block.add_op(after_next_index_op)

        running_is_under = false_op.result
        running_is_found = true_op.result

        for i, (dim, idx_buffer_ptr, idx_buffer_type) in enumerate(
                zip(
                    scoo_layout.dimensions,
                    index_buffer_ptrs,
                    scoo_layout.index_buffer_types,
                )
        ):
            convert_after_next_index_ops, after_next_index_i64 = index_convert(after_next_index_op.result, i64)
            after_block.add_ops(convert_after_next_index_ops)
            idx_ptr_arith_op = llvm.GEPOp(
                idx_buffer_ptr,
                [llvm.GEP_USE_SSA_VAL],
                [after_next_index_i64],
                pointee_type=idx_buffer_type,
            )
            inner_load_stored_index_op = llvm.LoadOp(
                idx_ptr_arith_op.result, idx_buffer_type
            )
            after_block.add_ops([idx_ptr_arith_op, inner_load_stored_index_op])
            current_value = inner_load_stored_index_op.dereferenced_value

            get_index_ops, (target,) = dim_mapping[dim].get().output()
            after_block.add_ops(get_index_ops)
            convert_target_ops, target = index_convert(target, idx_buffer_type)
            after_block.add_ops(convert_target_ops)

            inner_is_under_cmp = arith.Cmpi(current_value, target, "ult")
            inner_is_found_cmp = arith.Cmpi(current_value, target, "eq")
            after_block.add_ops([inner_is_under_cmp, inner_is_found_cmp])

            inner_is_now_under = arith.AndI(running_is_found, inner_is_under_cmp)
            inner_is_lesser_reduce = arith.OrI(running_is_under, inner_is_now_under)
            inner_is_found_reduce = arith.AndI(running_is_found, inner_is_found_cmp)
            after_block.add_ops([inner_is_now_under, inner_is_lesser_reduce, inner_is_found_reduce])

            running_is_found = inner_is_found_reduce.result
            running_is_under = inner_is_lesser_reduce.result

        after_is_found = running_is_found
        after_is_under = running_is_under
        after_out_index_op = arith.Select(after_is_under, after_index, after_next_index_op.result)
        after_block.add_op(after_out_index_op)
        after_block.add_op(scf.Yield(after_is_found, after_is_under, after_out_index_op.result))

        # While op, index <= start, returns (found, index)
        while_op = scf.While([init_found, init_under, init_index], [IntegerType(1), IndexType()], [before_block], [after_block])
        ops.append(while_op)
        index_is_found, index = while_op.res
        return ops, index_is_found, index

    @staticmethod
    def search_to_find_index(
        scoo_layout: dlt.SeparatedCOOLayoutAttr,
        start: SSAValue,
        end: SSAValue,
        index_buffer_ptrs: tuple[SSAValue],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> tuple[
        list[Operation], SSAValue, SSAValue
    ]:  # ops, index_is_found(bool), index(IndexType)

        # return SeparatedCOOSemantics.loop_to_find_index(scoo_layout, start, end, index_buffer_ptrs, dim_mapping)
        # return SeparatedCOOSemantics.loop_back_to_find_index(scoo_layout, start, end, index_buffer_ptrs, dim_mapping)
        return SeparatedCOOSemantics.binary_search_to_find_index(scoo_layout, start, end, index_buffer_ptrs, dim_mapping)


    @staticmethod
    def binary_search_to_find_index(
            scoo_layout: dlt.SeparatedCOOLayoutAttr,
            start: SSAValue,
            end: SSAValue,
            index_buffer_ptrs: tuple[SSAValue],
            dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
    ) -> tuple[
        list[Operation], SSAValue, SSAValue
    ]:  # ops, index_is_found(bool), index(IndexType)
        ops = []
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)

        last_dim = len(scoo_layout.dimensions) - 1
        range_start = start
        range_end = end
        found = true_op.result
        found_index = None
        for i, (dim, index_buffer_ptr, index_buffer_type) in enumerate(zip(scoo_layout.dimensions, index_buffer_ptrs, scoo_layout.index_buffer_types)):
            get_target_ops, (target,) = dim_mapping[dim].get().output()
            ops.extend(get_target_ops)
            if i < last_dim:
                bin_search_ops, result_start, result_end, result_found = SeparatedCOOSemantics.binary_search_buffer_to_find_index_range(range_start, range_end, index_buffer_ptr, index_buffer_type, target)
                ops.extend(bin_search_ops)
                range_start = result_start
                range_end = result_end
                found = result_found
            else:
                bin_search_ops, result_index, result_found = SeparatedCOOSemantics.binary_search_buffer_to_find_index(
                    range_start, range_end, index_buffer_ptr, index_buffer_type, target)
                ops.extend(bin_search_ops)
                found_index = result_index
                found = result_found
        return ops, found, found_index

    @staticmethod
    def binary_search_buffer_to_find_index_range(
        start: SSAValue,  # search range: [start, end)
        end: SSAValue,  # IndexType()
        index_buffer_ptr: SSAValue,  # ptr to array of index_buffer_type
        index_buffer_type: IntegerType,
        target: SSAValue,  # IndexType()
    ) -> tuple[
        list[Operation], SSAValue, SSAValue, SSAValue
    ]:  # ops, index_start(IndexType), index_end(IndexType), found(IntegerType(1))
        # the range [index_start, index_end) holds where the target is in the sorted array.
        # start <= index_start <= index_end <= end
        # found iff index_start != index_end
        # if !found, index_start is where target would be placed
        # We do 2 binary searches, one that loops to find the lower index, and one that finds the upper index
        ops = []
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        two_op = arith.Constant(IntegerAttr(2, IndexType()))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.extend([zero_op, one_op, two_op, false_op, true_op])

        # ops.append(printf.PrintFormatOp(
        #     "# Bin_search_for_idx_range: start: {}, end: {}, ptr: {}, type: "+str(index_buffer_type)+", target: {}",
        #     start, end, index_buffer_ptr, target))

        convert_ops, target = index_convert(
            target, index_buffer_type
        )
        ops.extend(convert_ops)

        lower_init_start = start
        lower_init_end = end
        init_start_index = start

        lower_init_max_index = start
        lower_init_min_gt_index = end

        lower_init_found = false_op.result

        lower_init_middle_op = arith.Subi(lower_init_end, one_op.result)
        ops.append(lower_init_middle_op)
        lower_init_middle = lower_init_middle_op

        lower_before_block = Block(
            arg_types=[
                IndexType(),  # lower_start
                IndexType(),  # lower_end
                IndexType(),  # start_index
                IndexType(),  # max_index
                IndexType(),  # min_gt_index
                IntegerType(1),  # found
                IndexType(),  # lower_middle
            ]
        )
        lower_before_start, lower_before_end, _, _, _, _, _ = lower_before_block.args
        # lower_before_block.add_op(printf.PrintFormatOp("# Bin_search_for_idx_range-Lower_before_block: start: {}, end: {}, s_idx: {}, max_idx: {}, min_gt_index: {}, found: {}, middle: {}", *lower_before_block.args))

        lower_loop_cmp_op = arith.Cmpi(lower_before_start, lower_before_end, "ult")
        lower_before_block.add_op(lower_loop_cmp_op)
        lower_before_block.add_op(
            scf.Condition(lower_loop_cmp_op.result, *lower_before_block.args)
        )

        lower_after_block = Block(arg_types=[a.type for a in lower_before_block.args])
        (
            lower_after_start,
            lower_after_end,
            lower_after_start_index,
            lower_after_max_index,
            lower_after_min_gt_index,
            lower_after_found,
            lower_after_middle,
        ) = lower_after_block.args
        # lower_after_block.add_op(printf.PrintFormatOp(
        #     "# Bin_search_for_idx_range-Lower_after_block-begin: start: {}, end: {}, s_idx: {}, max_idx: {}, min_gt_index: {}, found: {}, middle: {}",
        #     *lower_after_block.args))

        convert_lower_after_middle_ops, lower_after_middle_i64 = index_convert(lower_after_middle, i64)
        lower_after_block.add_ops(convert_lower_after_middle_ops)
        lower_index_ptr_arith_op = llvm.GEPOp(
            index_buffer_ptr,
            [llvm.GEP_USE_SSA_VAL],
            [lower_after_middle_i64],
            pointee_type=index_buffer_type,
        )
        lower_load_stored_index_op = llvm.LoadOp(
            lower_index_ptr_arith_op.result, index_buffer_type
        )
        lower_after_block.add_ops(
            [lower_index_ptr_arith_op, lower_load_stored_index_op]
        )
        lower_current_value = lower_load_stored_index_op.dereferenced_value

        # comparison ops
        lower_value_gt_op = arith.Cmpi(lower_current_value, target, "ugt")
        lower_value_eq_op = arith.Cmpi(lower_current_value, target, "eq")
        lower_value_lt_op = arith.Cmpi(lower_current_value, target, "ult")
        lower_after_block.add_ops(
            [lower_value_gt_op, lower_value_eq_op, lower_value_lt_op]
        )

        # middle plus 1
        lower_after_middle_plus_1_op = arith.Addi(lower_after_middle, one_op.result)
        lower_after_block.add_op(lower_after_middle_plus_1_op)

        # select new start and end
        lower_select_start_op = arith.Select(
            lower_value_lt_op.result,
            lower_after_middle_plus_1_op.result,
            lower_after_start,
        )
        lower_select_end_op = arith.Select(
            lower_value_lt_op.result, lower_after_end, lower_after_middle
        )

        # select start index output
        lower_select_start_index_op1 = arith.Select(
            lower_after_found,
            lower_after_start_index,
            lower_after_middle_plus_1_op.result,
        )
        lower_select_start_index_op2 = arith.Select(
            lower_value_gt_op.result, lower_after_start_index, lower_after_middle
        )
        lower_select_start_index_op3 = arith.Select(
            lower_value_lt_op.result,
            lower_select_start_index_op1.result,
            lower_select_start_index_op2.result,
        )

        # select max index
        lower_select_max_index_op1 = arith.Select(
            lower_value_eq_op.result,
            lower_after_middle,
            lower_after_max_index,
        )
        lower_select_max_index_op2 = arith.Select(
            lower_after_found,
            lower_after_max_index,
            lower_select_max_index_op1.result,
        )

        # select min gt index
        lower_select_min_gt_index_op = arith.Select(
            lower_value_gt_op.result, lower_after_middle, lower_after_min_gt_index
        )

        # select if found flag
        lower_select_found_op = arith.Select(
            lower_value_eq_op.result, true_op.result, lower_after_found
        )

        lower_after_block.add_ops(
            [
                lower_select_start_op,
                lower_select_end_op,
                lower_select_start_index_op1,
                lower_select_start_index_op2,
                lower_select_start_index_op3,
                lower_select_max_index_op1,
                lower_select_max_index_op2,
                lower_select_min_gt_index_op,
                lower_select_found_op,
            ]
        )

        # make new middle (a+b)/2  --->  a + (b-a)/2  --- this mitigates possible overflows in the a+b part.
        lower_end_sub_start_op = arith.Subi(lower_select_end_op.result, lower_select_start_op.result)
        lower_div_2_op = arith.DivUI(lower_end_sub_start_op.result, two_op.result)
        lower_make_new_middle_op = arith.Addi(lower_select_start_op.result, lower_div_2_op)
        lower_after_block.add_ops(
            [
                lower_end_sub_start_op,
                lower_div_2_op,
                lower_make_new_middle_op
            ]
        )
        lower_after_yield_op = scf.Yield(
            lower_select_start_op.result,
            lower_select_end_op.result,
            lower_select_start_index_op3.result,
            lower_select_max_index_op2.result,
            lower_select_min_gt_index_op.result,
            lower_select_found_op.result,
            lower_make_new_middle_op.result,
        )
        # lower_after_block.add_op(printf.PrintFormatOp(
        #     "# Bin_search_for_idx_range-Lower_after_block-exit: start: {}, end: {}, s_idx: {}, max_idx: {}, min_gt_index: {}, found: {}, middle: {}",
        #     *lower_after_yield_op.operands))

        lower_after_block.add_op(lower_after_yield_op)

        lower_while_op = scf.While([lower_init_start, lower_init_end, init_start_index, lower_init_max_index, lower_init_min_gt_index, lower_init_found, lower_init_middle],
                                   [a.type for a in lower_after_block.args],
                                   [lower_before_block], [lower_after_block])
        ops.append(lower_while_op)
        (
            lower_result_start,
            lower_result_end,
            start_index,
            lower_result_max_index,
            lower_result_min_gt_index,
            found,
            lower_result_middle,
        ) = lower_while_op.res


        max_index_plus_1_op = arith.Addi(lower_result_max_index, one_op.result)
        init_end_index_op = arith.Select(found, max_index_plus_1_op.result, start_index)
        ops.extend([max_index_plus_1_op, init_end_index_op])

        upper_init_start = init_end_index_op.result
        upper_init_end = lower_result_min_gt_index
        init_end_index = init_end_index_op.result

        upper_before_block = Block(arg_types=[IndexType(), IndexType(), IndexType()])
        upper_before_start, upper_before_end, _ = upper_before_block.args
        # upper_before_block.add_op(printf.PrintFormatOp(
        #     "# Bin_search_for_idx_range-Upper_before_block: start: {}, end: {}, index: {}",
        #     *upper_before_block.args))

        upper_loop_cmp_op = arith.Cmpi(upper_before_start, upper_before_end, "ult")
        upper_before_block.add_op(upper_loop_cmp_op)
        upper_before_block.add_op(
            scf.Condition(upper_loop_cmp_op.result, *upper_before_block.args)
        )

        upper_after_block = Block(arg_types=[a.type for a in upper_before_block.args])
        upper_after_start, upper_after_end, upper_after_end_index = upper_after_block.args
        # upper_after_block.add_op(printf.PrintFormatOp(
        #     "# Bin_search_for_idx_range-Upper_after_block-begin: start: {}, end: {}, index: {}",
        #     *upper_after_block.args))

        upper_end_sub_start_op = arith.Subi(upper_after_end, upper_after_start)
        upper_middle_div_2_op = arith.DivUI(upper_end_sub_start_op.result, two_op.result)
        upper_middle_op = arith.Addi(upper_after_start, upper_middle_div_2_op.result)
        upper_after_block.add_ops([upper_end_sub_start_op, upper_middle_div_2_op, upper_middle_op])
        upper_middle = upper_middle_op.result

        convert_upper_middle_cast_ops, upper_middle_i64 = index_convert(upper_middle, i64)
        upper_after_block.add_ops(convert_upper_middle_cast_ops)
        upper_index_ptr_arith_op = llvm.GEPOp(
            index_buffer_ptr,
            [llvm.GEP_USE_SSA_VAL],
            [upper_middle_i64],
            pointee_type=index_buffer_type,
        )
        upper_load_stored_index_op = llvm.LoadOp(
            upper_index_ptr_arith_op.result, index_buffer_type
        )
        upper_after_block.add_ops(
            [upper_index_ptr_arith_op, upper_load_stored_index_op]
        )
        upper_current_value = upper_load_stored_index_op.dereferenced_value

        # comparison ops
        upper_value_eq_op = arith.Cmpi(upper_current_value, target, "eq")
        upper_after_block.add_op(upper_value_eq_op)

        # middle plus 1
        upper_middle_plus_1_op = arith.Addi(upper_middle, one_op.result)
        upper_after_block.add_op(upper_middle_plus_1_op)

        #select upper start
        upper_select_start_op = arith.Select(upper_value_eq_op.result, upper_middle_plus_1_op.result, upper_after_start)
        # select upper end
        upper_select_end_op = arith.Select(upper_value_eq_op.result, upper_after_end, upper_middle)
        # select upper end index
        upper_select_end_index_op = arith.Select(upper_value_eq_op.result, upper_middle_plus_1_op.result, upper_after_end_index)

        upper_after_block.add_ops([upper_select_start_op, upper_select_end_op, upper_select_end_index_op])
        upper_after_yield_op = scf.Yield(upper_select_start_op.result, upper_select_end_op.result, upper_select_end_index_op.result)

        # upper_after_block.add_op(printf.PrintFormatOp(
        #     "# Bin_search_for_idx_range-Upper_after_block-exit: start: {}, end: {}, index: {}",
        #     *upper_after_yield_op.operands))
        upper_after_block.add_op(upper_after_yield_op)

        upper_while_op = scf.While([upper_init_start, upper_init_end, init_end_index], [a.type for a in upper_after_block.args], [upper_before_block], [upper_after_block])
        ops.append(upper_while_op)
        (
            upper_result_start,
            upper_result_end,
            end_index
        ) = upper_while_op.res

        # ops.append(printf.PrintFormatOp(
        #     "# Bin_search_for_idx_range finished: res_start: {}, res_end: {}, found: {}",
        #     start_index, end_index, found))

        return ops, start_index, end_index, found


    @staticmethod
    def binary_search_buffer_to_find_index(
        start: SSAValue,
        end: SSAValue,
        index_buffer_ptr: SSAValue,
        index_buffer_type: IntegerType,
        target: SSAValue,
    ) -> tuple[
        list[Operation], SSAValue, SSAValue
    ]:  # ops, index(IndexType) found(IntegerType(1))

        # index = start
        # middle = end -1
        #
        # while start < end {
        #   if idx[middle] > target {
        #       yield (start, middle, middle, (s+e)/2 ) # start end index new_middle
        #   } else {
        #   if idx[middle] < target {
        #       yield (middle+1, end, middle+1, (s+e)/2) # start end index new_middle
        #   } else {
        #       yield (middle+1, middle+1, middle, middle?) # start end index middle(don't care)
        #   }
        #   }
        #
        # }
        # found = start != index

        ops = []
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        one_op = arith.Constant(IntegerAttr(1, IndexType()))
        two_op = arith.Constant(IntegerAttr(2, IndexType()))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.extend([zero_op, one_op, two_op, false_op, true_op])

        # ops.append(printf.PrintFormatOp(
        #     "# Bin_search_for_idx: start: {}, end: {}, ptr: {}, type: "+str(index_buffer_type)+", target: {}",
        #     start, end, index_buffer_ptr, target))


        convert_ops, target = index_convert(
            target, index_buffer_type
        )
        ops.extend(convert_ops)

        init_middle_op = arith.Subi(end, one_op.result)
        ops.append(init_middle_op)

        init_start = start
        init_end = end
        init_index = start
        init_middle = init_middle_op.result

        before_block = Block(
            arg_types=[
                IndexType(),
                IndexType(),
                IndexType(),
                IndexType(),
            ]
        )
        before_start, before_end, before_index, before_middle = (
            before_block.args
        )

        loop_cmp_op = arith.Cmpi(before_start, before_end, "ult")
        before_block.add_op(loop_cmp_op)
        before_block.add_op(
            scf.Condition(loop_cmp_op.result, *before_block.args)
        )

        after_block = Block(
            arg_types=[a.type for a in before_block.args]
        )  # start, end, index, middle
        after_start, after_end, after_index, after_middle = after_block.args

        convert_after_middle_cast_ops, after_middle_i64 = index_convert(after_middle, i64)
        after_block.add_ops(convert_after_middle_cast_ops)
        index_ptr_arith_op = llvm.GEPOp(
            index_buffer_ptr,
            [llvm.GEP_USE_SSA_VAL],
            [after_middle_i64],
            pointee_type=index_buffer_type,
        )
        load_stored_index_op = llvm.LoadOp(
            index_ptr_arith_op.result, index_buffer_type
        )
        after_block.add_ops(
            [index_ptr_arith_op, load_stored_index_op]
        )
        current_value = load_stored_index_op.dereferenced_value

        # comparison ops
        value_gt_op = arith.Cmpi(current_value, target, "ugt")
        value_eq_op = arith.Cmpi(current_value, target, "eq")
        value_lt_op = arith.Cmpi(current_value, target, "ult")
        after_block.add_ops(
            [value_gt_op, value_eq_op, value_lt_op]
        )
        # middle plus 1
        middle_plus_1_op = arith.Addi(after_middle, one_op.result)
        after_block.add_op(middle_plus_1_op)

        # select upper start
        select_start_op = arith.Select(value_gt_op.result, after_start, middle_plus_1_op.result)
        # select upper end
        select_end_op1 = arith.Select(value_gt_op.result, after_middle, middle_plus_1_op)
        select_end_op2 = arith.Select(value_lt_op.result, after_end, select_end_op1.result)
        # select  index
        select_index_op = arith.Select(value_lt_op.result, middle_plus_1_op.result, after_middle)
        after_block.add_ops([select_start_op, select_end_op1, select_end_op2, select_index_op])

        # make new middle (a+b)/2  --->  a + (b-a)/2  --- this mitigates possible overflows in the a+b part.
        end_sub_start_op = arith.Subi(select_end_op2.result, select_start_op.result)
        div_2_op = arith.DivUI(end_sub_start_op.result, two_op.result)
        make_new_middle_op = arith.Addi(select_start_op.result, div_2_op)
        after_block.add_ops(
            [
                end_sub_start_op,
                div_2_op,
                make_new_middle_op
            ]
        )

        after_block.add_op(scf.Yield(select_start_op.result, select_end_op2.result, select_index_op.result, make_new_middle_op.result))

        while_op = scf.While([init_start, init_end, init_index, init_middle], [a.type for a in after_block.args], [before_block], [after_block])
        ops.append(while_op)
        result_start, result_end, result_index, result_middle = while_op.res

        found_op = arith.Cmpi(result_start, result_index, "ne")
        ops.append(found_op)

        # ops.append(printf.PrintFormatOp(
        #     "# Bin_search_for_idx: res index: {}, found: {}",
        #     result_index, found_op.result))

        return ops, result_index, found_op.result

    def malloc_buffers(
        self,
        scoo_layout: dlt.SeparatedCOOLayoutAttr,
        num_non_zero: SSAValue,
        child_size: NumericResult2,
        old_buffers: tuple[tuple[SSAValue, ...], SSAValue] | None = None,
    ) -> tuple[list[Operation], tuple[SSAValue, ...], SSAValue]:
        ops = []
        idx_buffers = []
        # Malloc the Unpacked COO index Buffers:
        for i, idx_buffer_type in enumerate(scoo_layout.index_buffer_types):
            alloc_idx_buffer_bytes_ops, (alloc_idx_buffer_bytes,) = (
                _get_accepted_type_size(idx_buffer_type).sum()
                * NumericResult.from_ssa(num_non_zero)
            ).output()
            ops.extend(alloc_idx_buffer_bytes_ops)
            convert_idx_bytes_ops, alloc_idx_buffer_bytes_i64 = index_convert(alloc_idx_buffer_bytes, i64)
            ops.extend(convert_idx_bytes_ops)

            if old_buffers is None:
                malloc_idx_buffer = llvm.CallOp(
                    "malloc",
                    alloc_idx_buffer_bytes_i64,
                    return_type=llvm.LLVMPointerType.opaque(),
                )
                ops.append(malloc_idx_buffer)
                idx_buffers.append(malloc_idx_buffer.returned)
                if self.semantics.print_memory_calls:
                    ops.append(printf.PrintFormatOp("# called malloc({}) -> {}", alloc_idx_buffer_bytes_i64, malloc_idx_buffer.returned))
                    ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
                    ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
                    ops.append(llvm.CallOp("fflush", nullptr.output))
                # ops.append(printf.PrintFormatOp("# malloc idx size: {} -> {}", alloc_idx_buffer_bytes, malloc_idx_buffer.returned))
            else:
                # ops.append(printf.PrintFormatOp("# Realloc idx: {} to size: {}", old_buffers[0][i],
                #                                 alloc_idx_buffer_bytes))
                realloc_idx_buffer = llvm.CallOp(
                    "realloc",
                    old_buffers[0][i],
                    alloc_idx_buffer_bytes_i64,
                    return_type=llvm.LLVMPointerType.opaque(),
                )
                ops.append(realloc_idx_buffer)
                idx_buffers.append(realloc_idx_buffer.returned)
                if self.semantics.print_memory_calls:
                    ops.append(printf.PrintFormatOp("# called realloc({}, {}) -> {}", old_buffers[0][i], alloc_idx_buffer_bytes_i64, realloc_idx_buffer.returned))
                    ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
                    ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
                    ops.append(llvm.CallOp("fflush", nullptr.output))
                # ops.append(printf.PrintFormatOp("# Realloc idx: {} to size: {} -> {}", old_buffers[0][i],
                #                                 alloc_idx_buffer_bytes, realloc_idx_buffer.returned))


        # c_db_ops, (child_size, child_size_debug) = child_size.split_n(2)
        # ops.extend(c_db_ops)

        # Malloc the Unpacked COO data Buffer:
        alloc_data_buffer_bytes_numeric = (
            child_size * NumericResult.from_ssa(num_non_zero)
        ).sum()
        alloc_data_buffer_ops, (alloc_data_buffer_bytes,) = (
            alloc_data_buffer_bytes_numeric.output()
        )
        ops.extend(alloc_data_buffer_ops)
        convert_data_bytes_ops,  alloc_data_buffer_bytes_i64 = index_convert(alloc_data_buffer_bytes, i64)
        ops.extend(convert_data_bytes_ops)

        if old_buffers is None:
            malloc_data_buffer = llvm.CallOp(
                "malloc",
                alloc_data_buffer_bytes_i64,
                return_type=llvm.LLVMPointerType.opaque(),
            )
            ops.append(malloc_data_buffer)
            data_buffer_ptr = malloc_data_buffer.returned
            if self.semantics.print_memory_calls:
                ops.append(printf.PrintFormatOp("# called malloc({}) -> {}", alloc_data_buffer_bytes_i64, data_buffer_ptr))
                ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
                ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
                ops.append(llvm.CallOp("fflush", nullptr.output))
            # ops.append(printf.PrintFormatOp("# malloc data size: {} -> {}", alloc_data_buffer_bytes, data_buffer_ptr))
        else:
            # ops.append(
            #     printf.PrintFormatOp("# Realloc data: {} to size: {}", old_buffers[1], alloc_data_buffer_bytes,))
            realloc_data_buffer = llvm.CallOp(
                "realloc",
                old_buffers[1],
                alloc_data_buffer_bytes_i64,
                return_type=llvm.LLVMPointerType.opaque(),
            )
            ops.append(realloc_data_buffer)
            data_buffer_ptr = realloc_data_buffer.returned
            if self.semantics.print_memory_calls:
                ops.append(printf.PrintFormatOp("# called realloc({}, {}) -> {}", old_buffers[1], alloc_data_buffer_bytes_i64, data_buffer_ptr))
                ops.append(nullptr_zero := arith.Constant(IntegerAttr(0, i64)))
                ops.append(nullptr := llvm.IntToPtrOp(nullptr_zero.result))
                ops.append(llvm.CallOp("fflush", nullptr.output))
            # ops.append(printf.PrintFormatOp("# Realloc data: {} to size: {} -> {}", old_buffers[1], alloc_data_buffer_bytes, data_buffer_ptr))

        # db_ops, (child_size_db_base, child_size_db_extra) = child_size_debug.output()
        # ops.extend(db_ops)
        # ops.append(printf.PrintFormatOp("Malloc  capacity:{}  child_size:{} + {}", num_non_zero, child_size_db_base, child_size_db_extra))
        # ops.append(printf.PrintFormatOp("Malloc  index buffer:{}  data buffer:{}", malloc_idx_buffer.returned, malloc_data_buffer.returned))
        # ops.append(printf.PrintFormatOp("Malloc  index bytes:{}  data bytes:{} for "+str(upcoo_layout), alloc_idx_buffer_bytes,  alloc_data_buffer_bytes))

        return ops, tuple(idx_buffers), data_buffer_ptr

    @staticmethod
    def copy_to_new_buffers(
        old_buffer_ptr: SSAValue,
        new_buffer_ptr: SSAValue,
        element_size: SSAValue,
        new_elem_index: SSAValue,
        new_capacity: SSAValue,
        elements_to_copy: SSAValue,
        extra_space: SSAValue,
        shuffle_instead: bool = False,
    ) -> tuple[list[Operation], SSAValue]:
        ops = []

        # ops.append(printf.PrintFormatOp("# Copy to new buffers.  elem_size:{}  new_elem_index:{}  new_capacity:{}  elements_to_copy: {}  extra_space:{} shuffle_instead:" + str(shuffle_instead), element_size, new_elem_index, new_capacity, elements_to_copy, extra_space))
        # ops.append(printf.PrintFormatOp("# old buffer: {}  new buffer: {}", old_buffer_ptr, new_buffer_ptr))
        # calculate how many bytes to copy before the new element
        pre_elem_buffer_size_ops, (pre_elem_buffer_size,) = (
            NumericResult.from_ssa(element_size)
            * NumericResult.from_ssa(new_elem_index)
        ).output()
        ops.extend(pre_elem_buffer_size_ops)

        if not shuffle_instead:
            # Copy the data before the new element
            # ops.append(printf.PrintFormatOp("# Call memcpy (before elem) dst: {}  src: {}  size: {}", new_buffer_ptr, old_buffer_ptr, pre_elem_buffer_size))
            convert_pre_elem_buffer_size_ops, pre_elem_buffer_size_i64 = index_convert(pre_elem_buffer_size, i64)
            ops.extend(convert_pre_elem_buffer_size_ops)
            pre_elem_idx_copy_op = llvm.CallOp(
                "memcpy",
                new_buffer_ptr,
                old_buffer_ptr,
                pre_elem_buffer_size_i64,
                return_type=None,
            )
            ops.append(pre_elem_idx_copy_op)

        # calculate the ptr in the old buffer - the rest that wasn't copied before
        post_elem_buffer_ptr_ops, post_elem_buffer_ptr = NumericResult.from_ssa(
            pre_elem_buffer_size
        ).add_to_llvm_pointer(old_buffer_ptr)
        ops.extend(post_elem_buffer_ptr_ops)
        # calculate the ptr in the new buffer, adding an extra element_size to give room for the new element
        new_post_elem_buffer_ptr_ops, new_post_elem_buffer_ptr = (
            NumericResult.from_ssa(pre_elem_buffer_size)
            + NumericResult.from_ssa(element_size)
        ).add_to_llvm_pointer(new_buffer_ptr)
        ops.extend(new_post_elem_buffer_ptr_ops)
        # calculate how much data to move, this will be the new last_index (new number of elements in the buffer)
        # minus what was copied already (new_elem_index), minus 1 to account for 1 new element
        post_elem_buffer_size_ops, (post_elem_buffer_size,) = (
            (
                NumericResult.from_ssa(element_size)
                * (
                    (
                        NumericResult.from_ssa(elements_to_copy)
                        - NumericResult.from_ssa(new_elem_index)
                    )
                )
            )
            + NumericResult.from_ssa(extra_space)
        ).output()
        ops.extend(post_elem_buffer_size_ops)
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        ops.append(zero_op)
        cmp_op = arith.Cmpi(zero_op, post_elem_buffer_size, "ne")
        ops.append(cmp_op)
        # ops.append(printf.PrintFormatOp("# Post elem copy size: {}, !=0?: {}", post_elem_buffer_size, cmp_op.result))
        if_needs_post_copy = []
        convert_post_elem_buffer_size_ops, post_elem_buffer_size_i64 = index_convert(post_elem_buffer_size, i64)
        if_needs_post_copy.extend(convert_post_elem_buffer_size_ops)
        if shuffle_instead:
            # Copy the elements after the new element - but expect an overlap
            # if_needs_post_copy.append(printf.PrintFormatOp("# Call memmove (after elem) dst: {}  src: {}  size: {}", new_post_elem_buffer_ptr,
            #                                 post_elem_buffer_ptr, post_elem_buffer_size))
            pre_elem_idx_copy_op = llvm.CallOp(
                "memmove",
                new_post_elem_buffer_ptr,
                post_elem_buffer_ptr,
                post_elem_buffer_size_i64,
                return_type=None,
            )
            if_needs_post_copy.append(pre_elem_idx_copy_op)
        else:
            # Copy the elements after the new element
            # if_needs_post_copy.append(printf.PrintFormatOp("# Call memcpy (after elem) dst: {}  src: {}  size: {}", new_post_elem_buffer_ptr,
            #                                 post_elem_buffer_ptr, post_elem_buffer_size))
            pre_elem_idx_copy_op = llvm.CallOp(
                "memcpy",
                new_post_elem_buffer_ptr,
                post_elem_buffer_ptr,
                post_elem_buffer_size_i64,
                return_type=None,
            )
            if_needs_post_copy.append(pre_elem_idx_copy_op)
        if_needs_post_copy.append(scf.Yield())
        ops.append(scf.If(cmp_op.result, [], if_needs_post_copy))

        # calculate the ptr to the new_element to return
        new_elem_ptr_ops, new_elem_ptr = NumericResult.from_ssa(
            pre_elem_buffer_size
        ).add_to_llvm_pointer(new_buffer_ptr)
        ops.extend(new_elem_ptr_ops)
        # ops.append(
        #     printf.PrintFormatOp("# new elem ptr {}", new_elem_ptr))
        return ops, new_elem_ptr

    class SparseDataWritingCallback(Callback):
        def __init__(
            self,
            semantics: SemanticsMapper,
            s_coo_layout: dlt.SeparatedCOOLayoutAttr,
            running_idx_buffer_ptrs: tuple[SSAValue, ...],
            running_data_buffer_ptr: SSAValue,
            running_total: SSAValue,
            total_non_zeros: SSAValue,
            sparse_initial_values: Initialiser,
            data_element_size: SSAValue,
            inner_callback: Callback,
            inner_callback_args: list[SSAValue],
        ):
            self.semantics = semantics
            self.s_coo_layout = s_coo_layout

            self.idx_buffer_ptrs = running_idx_buffer_ptrs
            assert all(
                isinstance(i.type, llvm.LLVMPointerType) for i in self.idx_buffer_ptrs
            )
            self.data_buffer_ptr = running_data_buffer_ptr
            assert isinstance(self.data_buffer_ptr.type, llvm.LLVMPointerType)
            self.running_total = running_total
            assert isinstance(self.running_total.type, IndexType)

            self.total_non_zeros = total_non_zeros
            assert isinstance(self.total_non_zeros.type, IndexType)
            self.sparse_initial_values = sparse_initial_values
            # assert all(
            #     isinstance(v.type, dlt.PtrType) for v in sparse_initial_values.values()
            # )
            self.data_element_size = data_element_size
            assert isinstance(self.data_element_size.type, IndexType)

            self.inner_callback = inner_callback
            self.inner_callback_args = inner_callback_args

            super().__init__(
                [*running_idx_buffer_ptrs, running_data_buffer_ptr, running_total]
                + self.inner_callback_args
            )

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, SSAValue],
            extent_resolver: ExtentResolver,
            base_type: dlt.AcceptedTypes,
            ptr: SSAValue,
            has_extra_space: SSAValue,
            is_last_element: bool | SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:

            ops: list[Operation] = []

            if is_last_element is True:
                true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
                ops.append(true_const)
                is_last_element = true_const.result

            extents = [d.extent for d in self.s_coo_layout.dimensions]
            extent_args = []
            for e in extents:
                if e.get_stage() >= dlt.Stage.INIT:
                    nr = extent_resolver.resolve(e)
                    ext_ops, ext = nr.output()
                    ops.extend(ext_ops)
                    extent_args.extend(ext)

            derived_initialiser = DerivedInitialiser(
                self.sparse_initial_values, members, dim_map
            )

            # Create the loop(s) over sparse dimensions
            iter_op = dlt.IterateOp(
                extents,
                extent_args,
                [],  # dim_specifiers,
                [],  # dlt_ptrs,
                iter_args,
                dlt.NestedIterationOrderAttr.generate_for(list(range(len(extents)))),
                None,
                None,
            )
            ops.append(iter_op)

            sparse_iter_block = iter_op.body.block
            sparse_iter_block.erase_op(
                sparse_iter_block.last_op
            )  # remove yield as we will add our own
            block_running_idx_buffer_ptrs = [
                iter_op.get_block_arg_for_iter_arg_idx(i)
                for i in range(len(self.idx_buffer_ptrs))
            ]
            block_running_data_buffer_ptr = iter_op.get_block_arg_for_iter_arg_idx(
                len(self.idx_buffer_ptrs)
            )
            block_running_total = iter_op.get_block_arg_for_iter_arg_idx(
                len(self.idx_buffer_ptrs) + 1
            )

            init_data_callback_args = [
                iter_op.get_block_arg_for_iter_arg_idx(3 + i)
                for i in range(len(self.inner_callback_args))
            ]

            block_dim_map = {
                d: a
                for d, a in zip(
                    self.s_coo_layout.dimensions,
                    iter_op.get_block_args_for_extent_args(),
                )
            }

            block_callback = DerivedCallback(
                self.inner_callback, members, dim_map | block_dim_map
            )

            get_non_zeros_ops, found_non_zeros = derived_initialiser.get_non_zero(
                set(),
                {k: ArgIndexGetter(v) for k, v in block_dim_map.items()},
                self.s_coo_layout.child.contents_type,
            )
            sparse_iter_block.add_ops(get_non_zeros_ops)
            bool_ssa_ops, has_non_zero = _make_bool_ssa(found_non_zeros)
            sparse_iter_block.add_ops(bool_ssa_ops)

            # setup If statement - if has_non_zeros:
            true_ops: list[Operation] = []
            one = NumericResult.from_mixed([], 1)
            running_total_numeric = NumericResult.from_mixed([], block_running_total)
            new_total_ops, (new_running_total,) = (running_total_numeric + one).output()
            true_ops.extend(new_total_ops)

            new_block_running_idx_buffer_ptrs = []
            for idx_arg, running_idx_buffer_ptr, idx_buffer_type in zip(
                iter_op.get_block_args_for_extent_args(),
                block_running_idx_buffer_ptrs,
                self.s_coo_layout.index_buffer_types,
            ):
                convert_idx_arg_ops, idx_val = index_convert(idx_arg, idx_buffer_type)
                true_ops.extend(convert_idx_arg_ops)
                if idx_buffer_type.width.data > builtin.i64.width.data:
                    raise NotImplementedError(
                        f"{idx_buffer_type} not supported as COO index buffer type"
                    )
                true_ops.append(llvm.StoreOp(idx_val, running_idx_buffer_ptr))
                idx_size = _get_accepted_type_size(idx_buffer_type).sum()
                new_idx_ptr_ops, new_running_idx_buffer_ptr = (
                    idx_size.add_to_llvm_pointer(running_idx_buffer_ptr)
                )
                true_ops.extend(new_idx_ptr_ops)
                new_block_running_idx_buffer_ptrs.append(new_running_idx_buffer_ptr)

            is_last_iter_op = arith.Cmpi(new_running_total, self.total_non_zeros, "eq")
            true_ops.append(is_last_iter_op)
            has_extra_space_op = arith.AndI(is_last_iter_op, has_extra_space)
            true_ops.append(has_extra_space_op)
            if isinstance(is_last_element, SSAValue):
                is_last_element_op = arith.AndI(is_last_iter_op, is_last_element)
                true_ops.append(is_last_element_op)
                is_last_element = is_last_iter_op.result

            init_data_ops, new_init_data_callback_args = self.semantics.init_layout(
                self.s_coo_layout.child,
                extent_resolver,
                block_running_data_buffer_ptr,
                DerivedInitialiser(derived_initialiser, set(), block_dim_map),
                block_callback,
                init_data_callback_args,
                has_extra_space_op.result,
                is_last_element,
            )
            true_ops.extend(init_data_ops)

            size_numeric = NumericResult.from_mixed([], self.data_element_size)
            new_data_ptr_ops, new_block_running_data_buffer_ptr = (
                size_numeric.add_to_llvm_pointer(block_running_data_buffer_ptr)
            )
            true_ops.extend(new_data_ptr_ops)

            true_ops.append(
                scf.Yield(
                    *new_block_running_idx_buffer_ptrs,
                    new_block_running_data_buffer_ptr,
                    new_running_total,
                    *new_init_data_callback_args,
                )
            )

            if_op = scf.If(
                has_non_zero,
                [llvm.LLVMPointerType.opaque()] * len(self.idx_buffer_ptrs)
                + [
                    llvm.LLVMPointerType.opaque(),
                    IndexType(),
                ]
                + [arg.type for arg in self.inner_callback.initial_iter_args()],
                true_ops,
                [
                    scf.Yield(
                        *block_running_idx_buffer_ptrs,
                        block_running_data_buffer_ptr,
                        block_running_total,
                        *init_data_callback_args,
                    )
                ],
            )
            sparse_iter_block.add_op(if_op)
            sparse_iter_block.add_op(dlt.IterateYieldOp(*if_op.output))

            new_iter_args = iter_op.results
            return ops, new_iter_args, False

    class SparseLinearIterCallback(Callback):
        def __init__(
            self,
            semantics: SemanticsMapper,
            s_coo_layout: dlt.SeparatedCOOLayoutAttr,
            idx_buffer_ptrs: tuple[SSAValue, ...],
            data_buffer_ptr: SSAValue,
            data_element_size: SSAValue,
            inner_callback: Callback,
            inner_callback_args: list[SSAValue],
            inner_is_last_element: bool | SSAValue,
            reversed_direction: bool,
        ):
            self.semantics = semantics
            self.s_coo_layout = s_coo_layout

            self.idx_buffer_ptrs = idx_buffer_ptrs
            assert all(
                isinstance(i.type, llvm.LLVMPointerType) for i in self.idx_buffer_ptrs
            )
            self.data_buffer_ptr = data_buffer_ptr
            assert isinstance(self.data_buffer_ptr.type, llvm.LLVMPointerType)

            self.data_element_size = data_element_size
            assert isinstance(self.data_element_size.type, IndexType)

            self.inner_callback = inner_callback
            # self.inner_callback_args = inner_callback_args

            self.inner_is_last_element = inner_is_last_element
            self.reversed_direction = reversed_direction

            super().__init__(inner_callback_args, inner_callback.can_exits_early())

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, SSAValue],
            extent_resolver: ExtentResolver,
            base_type: dlt.AcceptedTypes,
            ptr: SSAValue,
            has_extra_space: SSAValue,
            is_last_element: bool | SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            ops: list[Operation] = []
            true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.append(true_const)
            false_const = arith.Constant(IntegerAttr(0, IntegerType(1)))
            ops.append(false_const)

            if is_last_element is True:
                is_last_element = true_const.result
            if self.inner_is_last_element is True:
                true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
                ops.append(true_const)
                inner_is_last_element = true_const.result
            else:
                inner_is_last_element = self.inner_is_last_element

            getter_ops, index_range, index_range_found = self.semantics.get_getter_for(
                terminal_layout, dlt.IndexRangeType(), set(), {}, extent_resolver, ptr
            )
            ops.extend(getter_ops)
            extract_ops, start, end = _extract_indices_from_index_range(index_range, as_index=True)
            ops.extend(extract_ops)
            ops.append(one_op := arith.Constant(IntegerAttr(1, IndexType())))

            # idx_step_ops, (idx_step,) = (
            #     _get_accepted_type_size(IndexType()).sum().output()
            # )
            # ops.extend(idx_step_ops)

            end_minus_one_op = arith.Subi(end, one_op.result)
            ops.append(end_minus_one_op)

            if self.reversed_direction:
                start_idx = end_minus_one_op.result
                end_idx = start
                ops.append(step_op := arith.Constant(IntegerAttr(-1, IndexType())))
                step = step_op.result
                ops.append(zero_op := arith.Constant(IntegerAttr(0, IndexType())))
                last_iter_idx = zero_op.result
            else:
                start_idx, end_idx, step = start, end, one_op.result
                last_iter_idx = end_minus_one_op.result

            block = Block()
            index = block.insert_arg(IndexType(), 0)
            block.insert_arg(IntegerType(1), 1)  # exit early
            block_inner_callback_args = [
                block.insert_arg(arg.type, len(block.args)) for arg in iter_args
            ]

            # sort out has_extra_space and is_last_elem the data elem has extra space if this is the last elem of
            # iteration over the direct data source of the indexing node and the last iteration within this loop
            # across the sparse dims this is the last elem if it's the last iteration of the loop across the sparse
            # data (is_last_iter_op) and it's the last elem of the index source iteration (is_last_elem) and it was
            # the last elem for the indexing node (self.inner_is_last_elem)
            is_last_iter_op = arith.Cmpi(last_iter_idx, index, "eq")
            block.add_op(is_last_iter_op)
            if is_last_element is False:
                has_extra_space_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
                block.add_op(has_extra_space_op)
                inner_has_extra_space = has_extra_space_op.result
            else:
                has_extra_space_op = arith.AndI(is_last_iter_op, is_last_element)
                block.add_op(has_extra_space_op)
                inner_has_extra_space = has_extra_space_op.result

            if isinstance(inner_is_last_element, SSAValue):
                is_last_element_op = arith.AndI(is_last_iter_op, is_last_element)
                block.add_op(is_last_element_op)
                inner_last_elem_op = arith.AndI(
                    is_last_element_op.result, inner_is_last_element
                )
                block.add_op(inner_last_elem_op)
                inner_is_last_element = inner_last_elem_op.result

            current_idx_buffer_ptrs = []
            for idx_buffer_ptr, idx_buffer_type in zip(
                self.idx_buffer_ptrs, self.s_coo_layout.index_buffer_types
            ):
                idx_buffer_ops, current_idx_buffer_ptr = (
                    _get_accepted_type_size(idx_buffer_type).sum()
                    * NumericResult.from_ssa(index)
                ).add_to_llvm_pointer(idx_buffer_ptr)
                block.add_ops(idx_buffer_ops)
                current_idx_buffer_ptrs.append(current_idx_buffer_ptr)
            data_buffer_ops, current_data_buffer_ptr = (
                NumericResult.from_mixed([], index)
                * NumericResult.from_mixed([], self.data_element_size)
            ).add_to_llvm_pointer(self.data_buffer_ptr)
            block.add_ops(data_buffer_ops)

            sparse_dims = {}
            for dim, idx_buffer_ptr, idx_buffer_type in zip(
                self.s_coo_layout.dimensions,
                current_idx_buffer_ptrs,
                self.s_coo_layout.index_buffer_types,
            ):
                sparse_index_load = llvm.LoadOp(idx_buffer_ptr, idx_buffer_type)
                block.add_op(sparse_index_load)

                convert_inner_idx_ops, inner_idx = index_convert(sparse_index_load.dereferenced_value, IndexType())
                block.add_ops(convert_inner_idx_ops)
                if idx_buffer_type.width.data > builtin.i64.width.data:
                    raise NotImplementedError(
                        f"{idx_buffer_type} not supported as Separated COO index buffer type"
                    )

                sparse_dims[dim] = inner_idx

            block_callback = DerivedCallback(
                self.inner_callback, members, dim_map | sparse_dims
            )

            child_ops, new_inner_callback_arg, exited_early = (
                self.semantics.linear_iterate(
                    self.s_coo_layout.child,
                    extent_resolver,
                    current_data_buffer_ptr,
                    block_callback,
                    block_inner_callback_args,
                    inner_has_extra_space,
                    inner_is_last_element,
                    reversed_direction=self.reversed_direction,
                )
            )
            block.add_ops(child_ops)

            exited_ops, exit_early = _make_bool_ssa(exited_early)
            block.add_ops(exited_ops)

            inc_index_op = arith.Addi(index, step)
            block.add_op(inc_index_op)

            block.add_op(
                scf.Yield(
                    inc_index_op.result,
                    exit_early,
                    *new_inner_callback_arg,
                )
            )

            condition_block = Block()
            c_index = condition_block.insert_arg(IndexType(), 0)
            c_exit = condition_block.insert_arg(IntegerType(1), 1)
            c_block_inner_callback_args = [
                condition_block.insert_arg(arg.type, len(condition_block.args))
                for arg in iter_args
            ]
            if self.reversed_direction:
                cmp1 = arith.Cmpi(c_index, end_idx, "sge")
            else:
                cmp1 = arith.Cmpi(c_index, end_idx, "slt")
            condition_block.add_ops(
                [
                    cmp1,
                    cmp2 := arith.Cmpi(false_const.result, c_exit, "eq"),
                    cmp := arith.AndI(cmp1.result, cmp2.result),
                    scf.Condition(cmp, c_index, c_exit, *c_block_inner_callback_args),
                ]
            )
            while_loop = scf.While(
                [start_idx, false_const.result, *iter_args],
                [IndexType(), IntegerType(1), *[a.type for a in iter_args]],
                [condition_block],
                [block],
            )
            ops.append(while_loop)
            output_inner_callback_iter_args = list(
                while_loop.results[2 : 2 + len(iter_args)]
            )
            did_exit_early = while_loop.results[1]

            return ops, output_inner_callback_iter_args, did_exit_early

    class SparseMakeLoopCallback(LoopCallback):
        def __init__(
            self,
            semantics: SemanticsMapper,
            s_coo_layout: dlt.SeparatedCOOLayoutAttr,
            end_layout: dlt.DirectLayout,
            extent_resolver: ExtentResolver,
            members: set[dlt.MemberAttr],
            dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
            dims_to_loop: set[dlt.DimensionAttr],
            idx_buffer_ptrs: tuple[SSAValue, ...],
            data_buffer_ptr: SSAValue,
            data_element_size: SSAValue,
            inner_callback: LoopCallback,
            inner_callback_args: list[SSAValue],
        ):
            self.semantics = semantics
            self.s_coo_layout = s_coo_layout

            self.end_layout = end_layout
            self.extent_resolver = extent_resolver
            self.members = members
            self.dim_mapping = dim_mapping
            self.dims_to_loop = dims_to_loop

            self.idx_buffer_ptrs = idx_buffer_ptrs
            assert all(
                isinstance(i.type, llvm.LLVMPointerType) for i in self.idx_buffer_ptrs
            )
            self.data_buffer_ptr = data_buffer_ptr
            assert isinstance(self.data_buffer_ptr.type, llvm.LLVMPointerType)

            self.data_element_size = data_element_size
            assert isinstance(self.data_element_size.type, IndexType)

            self.inner_callback = inner_callback

            super().__init__(inner_callback_args)

        def callback(
            self,
            terminal_layout: dlt.Layout,
            members: set[dlt.MemberAttr],
            dim_map: dict[dlt.DimensionAttr, IndexGetter],
            dims_left_to_loop: set[dlt.DimensionAttr],
            extent_resolver: ExtentResolver,
            ptr: SSAValue,
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue]]:
            assert len(dims_left_to_loop) == 0
            ops: list[Operation] = []
            true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.append(true_const)
            false_const = arith.Constant(IntegerAttr(0, IntegerType(1)))
            ops.append(false_const)
            one_op = arith.Constant(IntegerAttr(1, IndexType()))
            ops.append(one_op)

            getter_ops, index_range, index_range_found = self.semantics.get_getter_for(
                terminal_layout, dlt.IndexRangeType(), set(), {}, extent_resolver, ptr
            )
            ops.extend(getter_ops)
            extract_ops, start, end = _extract_indices_from_index_range(index_range, as_index=True)
            ops.extend(extract_ops)

            start_idx, end_idx, step = start, end, one_op.result

            block = Block()
            index = block.insert_arg(IndexType(), 0)
            block_inner_callback_args = [
                block.insert_arg(arg.type, len(block.args)) for arg in iter_args
            ]
            current_idx_buffer_ptrs = []
            for idx_buffer_ptr, idx_buffer_type in zip(
                self.idx_buffer_ptrs, self.s_coo_layout.index_buffer_types
            ):
                idx_buffer_ops, current_idx_buffer_ptr = (
                    NumericResult.from_mixed([], index)
                    * _get_accepted_type_size(idx_buffer_type).sum()
                ).add_to_llvm_pointer(idx_buffer_ptr)
                block.add_ops(idx_buffer_ops)
                current_idx_buffer_ptrs.append(current_idx_buffer_ptr)
            data_buffer_ops, current_data_buffer_ptr = (
                NumericResult.from_mixed([], index)
                * NumericResult.from_mixed([], self.data_element_size)
            ).add_to_llvm_pointer(self.data_buffer_ptr)
            block.add_ops(data_buffer_ops)

            looped_sparse_dims = {}
            checked_sparse_dims = {}
            checked = true_const.result
            for dim, current_idx_buffer_ptr, idx_buffer_type in zip(
                self.s_coo_layout.dimensions,
                current_idx_buffer_ptrs,
                self.s_coo_layout.index_buffer_types,
            ):
                sparse_index_load = llvm.LoadOp(current_idx_buffer_ptr, idx_buffer_type)
                block.add_op(sparse_index_load)
                convert_inner_idx_ops, inner_idx = index_convert(sparse_index_load.dereferenced_value, IndexType())
                block.add_ops(convert_inner_idx_ops)
                if idx_buffer_type.width.data > builtin.i64.width.data:
                    raise NotImplementedError(
                        f"{idx_buffer_type} not supported as Separated COO index buffer type"
                    )
                if dim in self.dims_to_loop:
                    looped_sparse_dims[dim] = inner_idx
                elif dim in self.dim_mapping:
                    checked_sparse_dims[dim] = inner_idx
                    val_wanted_ops, (val_wanted,) = self.dim_mapping[dim].get().output()
                    block.add_ops(val_wanted_ops)
                    cmp_op = arith.Cmpi(checked_sparse_dims[dim], val_wanted, "eq")
                    block.add_op(cmp_op)
                    checked_op = arith.AndI(cmp_op.result, checked)
                    block.add_op(checked_op)
                    checked = checked_op.result

            if_checked_true = []
            block_callback = DerivedLoopCallback(
                self.inner_callback,
                members,
                dim_map | looped_sparse_dims | checked_sparse_dims,
            )

            child_ops, new_inner_callback_arg = self.semantics.make_sparse_loop_for(
                self.s_coo_layout.child,
                self.end_layout,
                self.extent_resolver,
                current_data_buffer_ptr,
                block_callback,
                block_inner_callback_args,
                self.members,
                {
                    d: g
                    for d, g in self.dim_mapping.items()
                    if d not in checked_sparse_dims
                },
                self.dims_to_loop - set(looped_sparse_dims.keys()),
            )
            if_checked_true.extend(child_ops)
            if_checked_true.append(scf.Yield(*new_inner_callback_arg))
            if_checked_false = []
            if_checked_false.append(scf.Yield(*block_inner_callback_args))
            if_checked_op = scf.If(
                checked,
                [arg.type for arg in iter_args],
                if_checked_true,
                if_checked_false,
            )
            block.add_op(if_checked_op)

            block.add_op(scf.Yield(*if_checked_op.results))
            for_loop = scf.For(start_idx, end_idx, step, iter_args, block)
            ops.append(for_loop)
            output_callback_iter_args = list(for_loop.res)

            return ops, output_callback_iter_args

    def set_up_new_child(
        self,
        layout: dlt.SeparatedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        initialiser: Initialiser,
        idx_elem_ptrs: tuple[SSAValue, ...],
        data_elem_ptr: SSAValue,
    ) -> list[Operation]:
        assert isinstance(data_elem_ptr.type, llvm.LLVMPointerType)
        ops = []
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        ops.append(false_op)

        for idx_elem_ptr, dim, idx_buffer_type in zip(
            idx_elem_ptrs, layout.dimensions, layout.index_buffer_types
        ):
            idx_arg_ops, (idx_arg,) = dim_mapping[dim].get().output()
            ops.extend(idx_arg_ops)
            convert_idx_arg_ops, idx_val = index_convert(idx_arg, idx_buffer_type)
            ops.extend(convert_idx_arg_ops)
            if idx_buffer_type.width.data > builtin.i64.width.data:
                raise NotImplementedError(
                    f"{idx_buffer_type} not supported as COO index buffer type"
                )
            ops.append(llvm.StoreOp(idx_val, idx_elem_ptr))
        # this is a new data element and so it needs to be init-ed
        child_init_ops, _ = self.semantics.init_layout(
            layout.child,
            extent_resolver,
            data_elem_ptr,
            initialiser,
            NoCallback(),
            [],
            false_op.result,
            True,
        )
        ops.extend(child_init_ops)
        return ops

    def do_new_buffers(
        self,
        layout: dlt.SeparatedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        new_size: SSAValue,
        idx_buffer_ptrs: tuple[SSAValue, ...],
        data_buffer_ptr: SSAValue,
        old_size: SSAValue,
        index: SSAValue,
    ) -> tuple[
        list[Operation], tuple[SSAValue, ...], SSAValue, tuple[SSAValue, ...], SSAValue
    ]:
        assert isinstance(new_size.type, IndexType)
        assert all(
            isinstance(idx_buffer_ptr.type, llvm.LLVMPointerType)
            for idx_buffer_ptr in idx_buffer_ptrs
        )
        assert isinstance(data_buffer_ptr.type, llvm.LLVMPointerType)
        assert isinstance(index.type, IndexType)
        ops = []
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        ops.append(zero_op)

        # Malloc the new COO Buffers:
        malloc_ops, new_idx_buffer_ptrs, new_data_buffer_ptr = self.malloc_buffers(
            layout,
            new_size,
            self.semantics.get_size(layout.child, extent_resolver),
            old_buffers=(idx_buffer_ptrs, data_buffer_ptr),
        )
        ops.extend(malloc_ops)

        # Set up for lots of memory copies to move the sparse index tuples and child data into their new larger buffers
        # Starting with Index tuples
        idx_elem_ptrs = []
        for new_idx_buffer_ptr, idx_buffer_type in zip(
            new_idx_buffer_ptrs, layout.index_buffer_types
        ):
            index_tuple_size_ops, (index_size,) = (
                _get_accepted_type_size(idx_buffer_type).sum().output()
            )
            ops.extend(index_tuple_size_ops)

            idx_copy_ops, idx_elem_ptr = self.copy_to_new_buffers(
                new_idx_buffer_ptr,
                new_idx_buffer_ptr,
                index_size,
                index,
                new_size,
                old_size,
                zero_op.result,
                shuffle_instead=True,
            )
            ops.extend(idx_copy_ops)
            idx_elem_ptrs.append(idx_elem_ptr)

        data_elem_size_ops, (data_elem_size, data_elem_size_extra) = (
            self.semantics.get_size(layout.child, extent_resolver).output()
        )
        ops.extend(data_elem_size_ops)

        data_copy_ops, data_elem_ptr = self.copy_to_new_buffers(
            new_data_buffer_ptr,
            new_data_buffer_ptr,
            data_elem_size,
            index,
            new_size,
            old_size,
            data_elem_size_extra,
            shuffle_instead=True,
        )
        ops.extend(data_copy_ops)

        # ops.append(printf.PrintFormatOp("# free idx {}", idx_buffer_ptr))
        # not needed since using realloc: ops.append(llvm.CallOp("free", idx_buffer_ptr))
        # ops.append(printf.PrintFormatOp("# free data {}", data_buffer_ptr))
        # not needed since using realloc: ops.append(llvm.CallOp("free", data_buffer_ptr))

        return (
            ops,
            new_idx_buffer_ptrs,
            new_data_buffer_ptr,
            tuple(idx_elem_ptrs),
            data_elem_ptr,
        )

    def do_move_buffers(
        self,
        layout: dlt.SeparatedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        buffer_size: SSAValue,
        new_number_of_elems: SSAValue,
        old_number_of_elems: SSAValue,
        idx_buffer_ptrs: tuple[SSAValue, ...],
        data_buffer_ptr: SSAValue,
        index: SSAValue,
    ) -> tuple[list[Operation], tuple[SSAValue, ...], SSAValue]:
        assert isinstance(buffer_size.type, IndexType)
        assert isinstance(data_buffer_ptr.type, llvm.LLVMPointerType)
        assert isinstance(index.type, IndexType)
        ops = []
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        ops.append(zero_op)

        # Set up for lots of memory copies to move the sparse index tuples and child data into their new larger buffers
        # Starting with Index tuples
        idx_elem_ptrs = []
        for idx_buffer_ptr, idx_buffer_type in zip(
            idx_buffer_ptrs, layout.index_buffer_types
        ):
            index_size_ops, (index_size,) = (
                _get_accepted_type_size(idx_buffer_type).sum()
            ).output()
            ops.extend(index_size_ops)

            idx_copy_ops, idx_elem_ptr = self.copy_to_new_buffers(
                idx_buffer_ptr,
                idx_buffer_ptr,
                index_size,
                index,
                buffer_size,
                old_number_of_elems,
                zero_op.result,
                shuffle_instead=True,
            )
            ops.extend(idx_copy_ops)
            idx_elem_ptrs.append(idx_elem_ptr)

        data_elem_size_ops, (data_elem_size, data_elem_size_extra) = (
            self.semantics.get_size(layout.child, extent_resolver).output()
        )
        ops.extend(data_elem_size_ops)

        data_copy_ops, data_elem_ptr = self.copy_to_new_buffers(
            data_buffer_ptr,
            data_buffer_ptr,
            data_elem_size,
            index,
            buffer_size,
            old_number_of_elems,
            data_elem_size_extra,
            shuffle_instead=True,
        )
        ops.extend(data_copy_ops)

        return ops, tuple(idx_elem_ptrs), data_elem_ptr


class ValueMapInitialiser(Initialiser):
    def __init__(
        self,
        semantics: SemanticsMapper,
        extent_resolver: ExtentResolver,
        value_map: dict[dlt.TypeType, SSAValue],
    ):
        self.semantics = semantics
        self.extent_resolver = extent_resolver
        self.value_map = value_map
        super().__init__()

    def get_value(
        self,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, IndexGetter],
        base_type: dlt.AcceptedTypes,
    ) -> tuple[list[Operation], None | SSAValue, bool | SSAValue]:
        for dlt_type, dlt_ptr in self.value_map.items():
            if dlt_type.has_selectable(members, dim_map.keys(), base_type):
                dlt_ptr_type = dlt_ptr.type
                assert isinstance(dlt_ptr_type, dlt.PtrType)
                dlt_ptr_type = typing.cast(dlt.PtrType, dlt_ptr_type)

                ops = []
                data_type = self.semantics.get_data_type_from_dlt_ptr(dlt_ptr_type)
                cast_op = builtin.UnrealizedConversionCastOp.get([dlt_ptr], [data_type])
                ops.append(cast_op)
                data_struct = cast_op.outputs[0]
                extract_ops, data_ptr, ptr_dim_map, ptr_extent_map = (
                    self.semantics.extract_from_ptr_struct(dlt_ptr_type, data_struct)
                )
                ops.extend(extract_ops)
                inner_members = set(dlt_ptr_type.filled_members) | members
                inner_dim_map = ptr_dim_map | dim_map
                inner_extent_resolver = self.extent_resolver.with_new(ptr_extent_map)

                getter_ops, value, found = self.semantics.get_getter_for(
                    dlt_ptr_type.layout,
                    base_type,
                    inner_members,
                    inner_dim_map,
                    inner_extent_resolver,
                    data_ptr,
                )
                ops.extend(getter_ops)
                return ops, value, found
        return [], None, False

    def get_non_zero(
        self,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, None | IndexGetter],
        type_type: dlt.TypeType,
    ) -> tuple[list[Operation], bool | SSAValue]:
        full_type = type_type.add_members(members).add_dimensions(dim_map.keys())

        # Loop through all stored values - these are expected to be mutually exclusive - so we only check the first that can have the value we're looking for
        for dlt_type, dlt_ptr in self.value_map.items():
            possible_selections = dlt_type.has_selectable_type(full_type)
            if (dlt.SetAttr([]), dlt.SetAttr([])) not in possible_selections:
                continue
                # continue to check the next dlt_ptr in this ValueMapInitialiser

            # set up ops list with some useful consts
            ops = []
            true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.append(true_op)
            false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
            ops.append(false_op)

            # collect the dims that are known to select them - along with the known members
            known_dims = []
            known_vals = []
            unknown_dims = []
            for d, v in dim_map.items():
                if v is None:
                    unknown_dims.append(d)
                else:
                    known_dims.append(d)
                    v_ops, (v_val,) = v.get().output()
                    ops.extend(v_ops)
                    known_vals.append(v_val)
            select_op = dlt.SelectOp(dlt_ptr, members, known_dims, known_vals)
            ops.append(select_op)

            # generate 'search space' TypeType
            per_elem_type_type = type_type.add_dimensions(unknown_dims)

            assert (dlt.SetAttr([]), dlt.SetAttr([])) in typing.cast(
                dlt.PtrType, select_op.res.type
            ).contents_type.has_selectable_type(per_elem_type_type)
            # now search through each element of the tensor
            elem_loop_found_non_zero = false_op.result
            for elem in per_elem_type_type.elements:
                # only do the work if we haven't already found a non-zero:
                per_elem_if_not_found_non_zero_ops = []
                # select all the members in the elem
                inner_select_op = dlt.SelectOp(
                    select_op.res, elem.member_specifiers, [], []
                )
                per_elem_if_not_found_non_zero_ops.append(inner_select_op)

                # iterate over all the unknown dimensions:
                # first collect extents from the dlt_ptr that may not exist in self.extent_map
                extract_ops, data_ptr, selected_dim_map, filled_extent_map = (
                    self.semantics.extract_from_ptr_struct(
                        typing.cast(dlt.PtrType, inner_select_op.res.type),
                        inner_select_op.res,
                    )
                )
                per_elem_if_not_found_non_zero_ops.extend(extract_ops)
                inner_dims = list(elem.dimensions)
                # make iterate op
                make_inner_iter_ops, inner_iter_op = _make_iterate(
                    inner_dims,
                    self.extent_resolver.with_new(filled_extent_map),
                    [inner_select_op.res],
                    [elem_loop_found_non_zero],
                )
                per_elem_if_not_found_non_zero_ops.extend(make_inner_iter_ops)
                # collect iterate op variables
                inner_loop_block = inner_iter_op.body.block
                inner_loop_block.erase_op(
                    inner_loop_block.last_op
                )  # remove yield as we will add our own
                inner_loop_tensor_arg = inner_iter_op.get_block_arg_for_tensor_arg_idx(
                    0
                )
                inner_loop_if_found_non_zero = (
                    inner_iter_op.get_block_arg_for_iter_arg_idx(0)
                )

                # Now we check if the value is actually non-zero - if a non=zero hasn't been found yet.
                inner_if_not_found_ops = []
                base_type = elem.base_type
                get_op = dlt.GetOp(inner_loop_tensor_arg, base_type)
                inner_if_not_found_ops.append(get_op)
                # non-zero condition:
                zero_cmp_ops, is_non_zero = _compare_is_non_zero(get_op.res)
                inner_if_not_found_ops.extend(zero_cmp_ops)

                inner_if_not_found_ops.append(scf.Yield(is_non_zero))
                # if non-zero already found in previous inner iteration, just return true - else check
                inner_if_op = scf.If(
                    inner_loop_if_found_non_zero,
                    [IntegerType(1)],
                    [scf.Yield(true_op.result)],
                    inner_if_not_found_ops,
                )
                inner_loop_block.add_op(inner_if_op)
                # inner loop yields true iff non-zero-found
                inner_loop_block.add_op(dlt.IterateYieldOp(inner_if_op.output[0]))
                # forward the inner loop's iter arg result the per elem if
                per_elem_if_not_found_non_zero_ops.append(
                    scf.Yield(inner_iter_op.get_result_for_iter_arg_idx(0))
                )
                # if a previous elem already found a non-zero, just return true, else check
                elem_if_op = scf.If(
                    elem_loop_found_non_zero,
                    [IntegerType(1)],
                    [scf.Yield(true_op.result)],
                    per_elem_if_not_found_non_zero_ops,
                )
                ops.append(elem_if_op)
                # update found_non_zeros from the inner loop's resulting iter_args
                elem_loop_found_non_zero = elem_if_op.output[0]
            # all elems have been checked now - so elem_loop_found_non_zero is our answer
            non_zero_found = elem_loop_found_non_zero
            return ops, non_zero_found
        return [], False


class SingletonInitialiser(Initialiser):
    def __init__(
        self,
        value: SSAValue,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, IndexGetter | SSAValue],
    ):
        self.value = value
        assert isinstance(self.value.type, dlt.AcceptedTypes)
        self.members = members
        self.dim_map: dict[dlt.DimensionAttr, IndexGetter] = {
            dim: (ArgIndexGetter(idx) if isinstance(idx, SSAValue) else idx)
            for dim, idx in dim_map.items()
        }

    def get_value(
        self,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, IndexGetter],
        base_type: dlt.AcceptedTypes,
    ) -> tuple[list[Operation], None | SSAValue, bool | SSAValue]:
        if (
            self.members != members
            or set(self.dim_map.keys()) != set(dim_map.keys())
            or self.value.type != base_type
        ):
            return [], None, False
        ops = []
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)
        found = true_op.result
        for dim in dim_map:
            other_get_ops, (other_index,) = dim_map[dim].get().output()
            ops.extend(other_get_ops)
            self_get_ops, (self_index,) = self.dim_map[dim].get().output()
            ops.extend(self_get_ops)
            compare = arith.Cmpi(other_index, self_index, "eq")
            ops.append(compare)
            and_op = arith.AndI(found, compare.result)
            ops.append(and_op)
            found = and_op.result
        return ops, self.value, found

    def get_non_zero(
        self,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, None | IndexGetter],
        type_type: dlt.TypeType,
    ) -> tuple[list[Operation], bool | SSAValue]:
        full_type = type_type.add_members(members).add_dimensions(dim_map.keys())
        val_type = self.value.type
        assert isinstance(val_type, dlt.AcceptedTypes)
        val_type = typing.cast(dlt.AcceptedTypes, val_type)
        self_type_elem = dlt.ElementAttr(self.members, self.dim_map.keys(), val_type)

        if self_type_elem not in full_type.elements:
            return [], False
        if len([v for v in dim_map.values() if v is not None]) == 0:
            return [], True
        ops = []
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)
        found = true_op.result
        for dim in dim_map:
            if dim_map[dim] is not None:
                other_get_ops, (other_index,) = dim_map[dim].get().output()
                ops.extend(other_get_ops)
                self_get_ops, (self_index,) = self.dim_map[dim].get().output()
                ops.extend(self_get_ops)
                compare = arith.Cmpi(other_index, self_index, "eq")
                ops.append(compare)
                and_op = arith.AndI(found, compare.result)
                ops.append(and_op)
                found = and_op.result
        return ops, found


class GetFirstValueCallback(Callback):

    def __init__(self, semantics: SemanticsMapper, init_value: SSAValue):
        self.semantics = semantics
        self.target_type = init_value.type
        super().__init__([init_value], can_exits_early=True)

    def callback(
        self,
        terminal_layout: dlt.Layout,
        members: set[dlt.MemberAttr],
        dim_map: dict[dlt.DimensionAttr, SSAValue],
        extent_resolver: ExtentResolver,
        base_type: dlt.AcceptedTypes,
        ptr: SSAValue,
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        if base_type == self.target_type:
            getter_ops, value, value_found = self.semantics.get_getter_for(
                terminal_layout,
                base_type,
                set(),
                {},
                extent_resolver,
                ptr,
            )
            return getter_ops, [value], value_found
        else:
            return [], iter_args, False


def _get_accepted_type_size(t: dlt.AcceptedTypes) -> NumericResult2:
    if isinstance(t, dlt.DLTCompatibleElementBaseType):
        p, e = t.get_size()
        return NumericResult.from_mixed([], p, e)

    if isinstance(t, IntegerType):
        bit_width = t.width.data
    elif isinstance(t, AnyFloat):
        bit_width = t.get_bitwidth
    else:
        raise ValueError(f"Cannot get size of base element: {t}")
    num_bytes = -(bit_width // -8)
    return NumericResult.from_mixed([], num_bytes, 0)

def _get_ptr_size() -> NumericResult2:
    bit_width = i64.width.data
    num_bytes = -(bit_width // -8)
    return NumericResult2.from_const(num_bytes,0)
    # here it is assumed this is a 64bit pointer system. overcoming this appears too complex for now.


def _get_unpacked_zero_for_accepted_type(
    t: dlt.AcceptedTypes,
) -> tuple[list[Operation], SSAValue, SSAValue | None]:
    if isinstance(t, IntegerType):
        return [op := arith.Constant(IntegerAttr(0, t))], op.result, None
    elif isinstance(t, AnyFloat):
        return [op := arith.Constant(FloatAttr(0.0, t))], op.result, None
    elif isinstance(t, dlt.IndexRangeType):
        return [op := arith.Constant(IntegerAttr(0, i64))], op.result, op.result
    else:
        raise ValueError(f"Cannot get zero for base element: {t}")


def _get_packed_zero_for_accepted_type(
    t: dlt.AcceptedTypes,
) -> tuple[list[Operation], SSAValue]:
    if isinstance(t, IntegerType):
        return [op := arith.Constant(IntegerAttr(0, t))], op.result
    elif isinstance(t, AnyFloat):
        return [op := arith.Constant(FloatAttr(0.0, t))], op.result
    elif isinstance(t, dlt.IndexRangeType):
        op = arith.Constant(IntegerAttr(0, i64))
        pack_ops, zero = _pack_indices_in_index_range(op.result, op.result)
        return [op] + pack_ops, zero
    else:
        raise ValueError(f"Cannot get zero for base element: {t}")


def _compare_is_non_zero(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert isinstance(value.type, dlt.AcceptedTypes)
    ops, zero_val = _get_packed_zero_for_accepted_type(
        typing.cast(dlt.AcceptedTypes, value.type)
    )
    if isinstance(value.type, builtin.AnyFloat):
        cond_op = arith.Cmpf(zero_val, value, "une")
        return ops + [cond_op], cond_op.result
    elif isinstance(value.type, builtin.AnySignlessIntegerOrIndexType):
        cond_op = arith.Cmpi(zero_val, value, "ne")
        return ops + [cond_op], cond_op.result
    else:
        raise NotImplementedError()


def _make_iterate(
    dimensions: list[dlt.DimensionAttr],
    extent_resolver: ExtentResolver,
    ptrs: list[SSAValue],
    iter_args: list[SSAValue],
    loop_order: dlt.IterationOrder | None = None,
) -> tuple[list[Operation], dlt.IterateOp]:
    ops = []
    extents = [d.extent for d in dimensions]
    extent_args = []
    for e in extents:
        if e.get_stage() >= dlt.Stage.INIT:
            nr = extent_resolver.resolve(e)
            ext_ops, ext = nr.output()
            ops.extend(ext_ops)
            extent_args.extend(ext)
    dim_specifiers = []
    dlt_ptrs = []
    for dlt_ptr in ptrs:
        ptr_type = typing.cast(dlt.PtrType, dlt_ptr.type)
        assert isinstance(ptr_type, dlt.PtrType)
        assert ptr_type.contents_type.has_selectable([], dimensions)
        dim_specifiers.append([[d] for d in dimensions])
        dlt_ptrs.append(dlt_ptr)

    if loop_order is None:
        loop_order = dlt.NestedIterationOrderAttr.generate_for(
            list(range(len(extents)))
        )

    iter_op = dlt.IterateOp(
        extents,
        extent_args,
        dim_specifiers,
        dlt_ptrs,
        iter_args,
        loop_order,
        None,
        None,
    )
    ops.append(iter_op)
    return ops, iter_op


def _extract_indices_from_index_range(
    val: SSAValue,
    as_index: bool = False,
) -> tuple[list[Operation], SSAValue, SSAValue]:
    ops: list[Operation] = []
    cast = builtin.UnrealizedConversionCastOp.get(
        [val], [llvm.LLVMStructType.from_type_list([i64, i64])]
    )
    ops.append(cast)
    extract_start = llvm.ExtractValueOp(
        DenseArrayBase.from_list(i64, [0]), cast.outputs[0], i64
    )
    ops.append(extract_start)
    extract_end = llvm.ExtractValueOp(
        DenseArrayBase.from_list(i64, [1]), cast.outputs[0], i64
    )
    ops.append(extract_end)
    if not as_index:
        return ops, extract_start.res, extract_end.res

    convert_start_ops, start = index_convert(extract_start.res, IndexType())
    convert_end_ops, end = index_convert(extract_end.res, IndexType())
    return ops + convert_start_ops + convert_end_ops, start, end

def _pack_indices_in_index_range(
    start: SSAValue,
    end: SSAValue,
) -> tuple[list[Operation], SSAValue]:
    ops: list[Operation] = []
    undef_op = llvm.UndefOp(
        llvm.LLVMStructType.from_type_list([i64, i64])
    )
    ops.append(undef_op)
    insert_1_op = llvm.InsertValueOp(
        DenseArrayBase.from_list(i64, [0]), undef_op.res, start
    )
    ops.append(insert_1_op)
    insert_2_op = llvm.InsertValueOp(
        DenseArrayBase.from_list(i64, [1]), insert_1_op.res, end
    )
    ops.append(insert_2_op)
    cast = builtin.UnrealizedConversionCastOp.get(
        [insert_2_op.res], [dlt.IndexRangeType()]
    )
    ops.append(cast)
    return ops, cast.outputs[0]


def _make_bool_ssa(value: bool | SSAValue) -> tuple[list[Operation], SSAValue]:
    if isinstance(value, SSAValue):
        assert value.type == IntegerType(1)
        return [], value
    elif value is False:
        return [f := arith.Constant(IntegerAttr(0, IntegerType(1)))], f.result
    elif value is True:
        return [t := arith.Constant(IntegerAttr(1, IntegerType(1)))], t.result
    else:
        raise ValueError("Cannot convert to bool ssa")


# Semantic_Map = SemanticsMapper()

def load_all_semantics(semantics_map: SemanticsMapper) -> SemanticsMapper:
    semantics_map.add_direct(dlt.PrimitiveLayoutAttr, PrimitiveSemantics(semantics_map))
    semantics_map.add_direct(dlt.ConstantLayoutAttr, ConstantSemantics(semantics_map))
    semantics_map.add_direct(dlt.MemberLayoutAttr, MemberSemantics(semantics_map))
    semantics_map.add_direct(dlt.DenseLayoutAttr, DenseSemantics(semantics_map))
    semantics_map.add_direct(dlt.StructLayoutAttr, StructSemantics(semantics_map))
    semantics_map.add_direct(dlt.ArithDropLayoutAttr, ArithDropSemantics(semantics_map))
    semantics_map.add_direct(dlt.ArithReplaceLayoutAttr, ArithReplaceSemantics(semantics_map))

    semantics_map.add_direct(dlt.IndexingLayoutAttr, IndexingSemantics(semantics_map))
    semantics_map.add_indexed(dlt.UnpackedCOOLayoutAttr, UnpackCOOSemantics(semantics_map))
    semantics_map.add_indexed(
        dlt.SeparatedCOOLayoutAttr, SeparatedCOOSemantics(semantics_map)
    )
    return semantics_map
