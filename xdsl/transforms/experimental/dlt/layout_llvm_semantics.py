import abc
import typing
from collections.abc import Callable

from xdsl.dialects import arith, builtin, llvm, scf
from xdsl.dialects.builtin import (
    AnyFloat,
    DenseArrayBase,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    UnrealizedConversionCastOp,
    i64,
)
from xdsl.dialects.experimental import dlt
from xdsl.ir import Block, Operation, SSAValue

T = typing.TypeVar("T", bound=dlt.Layout)


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

    def __add__(self, other) -> typing.Self:
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

    def __sub__(self, other) -> typing.Self:
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

    def __mul__(self, other) -> typing.Self:
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
        ops.append(cast_op := UnrealizedConversionCastOp.get([val], [i64]))
        val = cast_op.outputs[0]
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


def _get_as_i64(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert isinstance(value.type, IndexType | IntegerType)
    if isinstance(value.type, IntegerType):
        t = typing.cast(IntegerType, value.type)
        assert (
            t.width.data <= i64.width.data
        ), f"Expected {i64.width.data} got {t.width.data}"
    return [op := UnrealizedConversionCastOp.get([value], [i64])], op.outputs[0]


def _get_as_index(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert isinstance(value.type, IndexType | IntegerType)
    if isinstance(value.type, IntegerType):
        t = typing.cast(IntegerType, value.type)
        assert (
            t.width.data <= i64.width.data
        ), f"Expected {i64.width.data} got {t.width.data}"
    return [op := UnrealizedConversionCastOp.get([value], [IndexType()])], op.outputs[0]


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


class StaticExtentGetter(ExtentGetter):
    def __init__(self, arg: dlt.StaticExtentAttr):
        assert arg.get_stage() <= dlt.Stage.STATIC
        if isinstance(arg, dlt.StaticExtentAttr):
            self.extent = arg.value.value.data
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

    def __init__(self, extent_map: dict[dlt.Extent, ExtentGetter]):
        self.map: dict[dlt.Extent, ExtentGetter] = extent_map

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
        selected_data: dict[dlt.TypeType, SSAValue],
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        raise NotImplementedError

    # returns ops, list of iter args to be passed in to the next invocation, and a value deciding if the iterating
    # should stop early. Exiting early is ignored by Init-iteration.


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
        selected_data: dict[dlt.TypeType, SSAValue],
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        print("DEBUG CALL BACK!")
        debug_const = arith.Constant(IntegerAttr(100, IntegerType(20)))
        debug_const.attributes["debug"] = builtin.StringAttr(
            "          NoCallBack here"
        )
        return [debug_const], [], False


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
        selected_data: dict[dlt.TypeType, SSAValue],
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
            selected_data,
            iter_args,
        )
        return ops, iter_args_out, exit_early


class SemanticsMapper:
    def __init__(self):
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
        assert val.owner in ops
        return ops, val

    def get_getter_for(
        self,
        starting_layout: T,
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

    def get_setter_for(
        self,
        starting_layout: T,
        set_val: SSAValue,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> list[Operation]:
        assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        assert starting_layout.contents_type.has_selectable(
            members, dim_mapping.keys(), set_val.type
        )
        ops = self.get_direct(starting_layout).get_setter_for(
            starting_layout, set_val, members, dim_mapping, extent_resolver, input_ptr
        )
        return ops

    def init_layout(
        self,
        layout: dlt.Layout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.TypeType, SSAValue],
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: bool | SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        assert isinstance(
            input_ptr.type, llvm.LLVMPointerType
        ), f"Input pointer expected to be LLVMPointerType but found {type(input_ptr.type)}"
        assert all(
            isinstance(val.type, dlt.PtrType) and val.type.contents_type == t
            for t, val in initial_values.items()
        )
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
        return ops, iter_args

    def linear_iterate(
        self,
        layout: dlt.Layout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_data: dict[dlt.TypeType, SSAValue],
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: bool | SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool = False,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        assert isinstance(
            input_ptr.type, llvm.LLVMPointerType
        ), f"Input pointer expected to be LLVMPointerType but found {type(input_ptr.type)}"
        assert all(
            isinstance(val.type, dlt.PtrType) and val.type.contents_type == t
            for t, val in selected_data.items()
        )
        ops: list[Operation] = []
        if isinstance(has_extra_space, bool):
            ops.append(
                op := arith.Constant(IntegerAttr(int(has_extra_space), IntegerType(1)))
            )
            has_extra_space = op.result
        init_ops, iter_args, exited_early = self.get_direct(layout).linear_iterate(
            layout,
            extent_resolver,
            input_ptr,
            selected_data,
            callback,
            callback_args,
            has_extra_space,
            is_last_element,
            reversed_direction,
        )
        ops.extend(init_ops)
        return ops, iter_args, exited_early

    def ensure_space(
        self,
        layout: dlt.Layout,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
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


class LayoutNodeSemantics(abc.ABC, typing.Generic[T]):

    def __init__(self, semantics: SemanticsMapper):
        self.semantics = semantics

    @abc.abstractmethod
    def get_size(self, layout: T, extent_resolver: ExtentResolver) -> NumericResult2:
        raise NotImplementedError

    @abc.abstractmethod
    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_values: dict[dlt.TypeType, SSAValue],
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        raise NotImplementedError

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
        initial_values: dict[dlt.TypeType, SSAValue],
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        # assert isinstance(input_ptr, llvm.LLVMPointerType)
        raise NotImplementedError

    @abc.abstractmethod
    def ensure_space(
        self,
        layout: T,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        raise NotImplementedError


class IndexedLayoutNodeSemantics(typing.Generic[T], LayoutNodeSemantics[T], abc.ABC):
    pass


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
        return [llvm.StoreOp(set_val, input_ptr)]

    def init_layout(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.TypeType, SSAValue],
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        dlt_type = layout.contents_type
        ops: list[Operation] = []
        if dlt_type in initial_values:
            ops.extend(
                [
                    init_val := dlt.GetOp(initial_values[dlt_type], layout.base_type),
                    llvm.StoreOp(init_val, input_ptr),
                ]
            )
        else:
            zero_ops, zero_val, zero_extra_val = _get_unpacked_zero_for_accepted_type(
                layout.base_type
            )
            ops.extend(
                zero_ops
                + [
                    llvm.StoreOp(zero_val, input_ptr),
                ]
            )
            if zero_extra_val is not None:
                if_extra_space = []
                ptr_ops, extra_ptr = (
                    _get_accepted_type_size(layout.base_type)
                    .keep(1)
                    .add_to_llvm_pointer(input_ptr)
                )
                if_extra_space.extend(
                    ptr_ops + [llvm.StoreOp(zero_extra_val, extra_ptr), scf.Yield()]
                )
                ops.append(scf.If(has_extra_space, [], if_extra_space))

        callback_ops, iter_args_out, exited = init_callback.callback(
            layout,
            set(),
            {},
            extent_resolver,
            layout.base_type,
            input_ptr,
            has_extra_space,
            is_last_element,
            initial_values,
            callback_args,
        )
        ops.extend(callback_ops)

        return ops, iter_args_out

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_values: dict[dlt.TypeType, SSAValue],
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
            selected_values,
            callback_args,
        )
        return callback_ops, iter_args_out, exited

    def ensure_space(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        # Not much to do here - the space already exist by definition.
        ops, val, found = self.get_getter_for(
            layout, base_type, members, dim_mapping, extent_resolver, input_ptr
        )
        # since a primitive always exists, there's no need to use the fill_value_getter
        return ops, val


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
        initial_values: dict[dlt.TypeType, SSAValue],
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        # There is nothing it init in a constant
        pass

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_values: dict[dlt.TypeType, SSAValue],
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

    def ensure_space(
        self,
        layout: dlt.ConstantLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
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
        initial_values: dict[dlt.TypeType, SSAValue],
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        ops, new_selected_values = self._select_values_selector(
            initial_values, [layout.member_specifier], {}
        )
        new_callback = DerivedCallback(init_callback, {layout.member_specifier}, {})
        child_ops, iter_args_out = self.semantics.init_layout(
            layout.child,
            extent_resolver,
            input_ptr,
            new_selected_values,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
        )
        ops.extend(child_ops)
        return ops, iter_args_out

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_values: dict[dlt.TypeType, SSAValue],
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        ops, new_selected_values = self._select_values_selector(
            selected_values, [layout.member_specifier], {}
        )
        new_callback = DerivedCallback(callback, {layout.member_specifier}, {})
        child_ops, iter_args_out, exited = self.semantics.linear_iterate(
            layout.child,
            extent_resolver,
            input_ptr,
            new_selected_values,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
            reversed_direction,
        )
        ops.extend(child_ops)
        return ops, iter_args_out, exited

    def ensure_space(
        self,
        layout: dlt.MemberLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
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
        fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
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
                    {},
                    get_index_range_callback,
                    [w_index_range],
                    False,
                    False,
                    reversed_direction=True,
                )
            )
            while_block.add_ops(w_lin_iter_ops)
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
        initial_values: dict[dlt.TypeType, SSAValue],
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
        false_const.attributes["debug"] = builtin.StringAttr(
            "                              Dense Init/iter Layout Start"
        )
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

        select_ops, new_init_values = self._select_values_selector(
            initial_values, [], {layout.dimension: index}
        )
        block.add_ops(select_ops)

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
        block.add_op(has_extra_if)

        if isinstance(is_last_element, SSAValue):
            is_last_element_if = scf.If(
                is_last_iter_op.result,
                [IntegerType(1)],
                [scf.Yield(is_last_element)],
                [scf.Yield(false_const)],
            )
            block.add_op(is_last_element_if)
            is_last_element = is_last_element_if.output[0]

        child_ops, iter_args_out = self.semantics.init_layout(
            layout.child,
            extent_resolver,
            ptr_arg,
            new_init_values,
            new_callback,
            new_callback_args,
            has_extra_if.output[0],
            is_last_element,
        )
        block.add_ops(child_ops)
        block.add_op(scf.Yield(*iter_args_out))

        loop = scf.For(lb, ub, step, callback_args, block)
        ops.append(loop)

        debug_const = arith.Constant(IntegerAttr(100, IntegerType(20)))
        debug_const.attributes["debug"] = builtin.StringAttr(
            "Dense Semantics Init/iter Layout end"
        )
        ops.append(debug_const)

        return ops, list(loop.res)

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_values: dict[dlt.TypeType, SSAValue],
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
        false_const.attributes["debug"] = builtin.StringAttr(
            "                              Dense Init/iter Layout Start"
        )
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
            ops.append(step := arith.Constant(IntegerAttr(-1, IndexType())))
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

        select_ops, new_init_values = self._select_values_selector(
            selected_values, [], {layout.dimension: index}
        )
        block.add_ops(select_ops)

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
        block.add_op(has_extra_if)

        if isinstance(is_last_element, SSAValue):
            is_last_element_if = scf.If(
                is_last_iter_op.result,
                [IntegerType(1)],
                [scf.Yield(is_last_element)],
                [scf.Yield(false_const)],
            )
            block.add_op(is_last_element_if)
            is_last_element = is_last_element_if.output[0]

        child_ops, iter_args_out, inner_exited = self.semantics.linear_iterate(
            layout.child,
            extent_resolver,
            ptr_arg,
            new_init_values,
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
        output_callback_iter_args = list(while_loop.results[2: 2 + len(callback_args)])
        did_exit_early = while_loop.results[1]

        debug_const = arith.Constant(IntegerAttr(100, IntegerType(20)))
        debug_const.attributes["debug"] = builtin.StringAttr(
            "Dense Semantics Init/iter Layout end"
        )
        ops.append(debug_const)

        return ops, output_callback_iter_args, did_exit_early


class StructSemantics(DirectLayoutNodeSemantics[dlt.StructLayoutAttr]):
    def get_size(
        self, layout: dlt.StructLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        result = NumericResult.from_mixed([], 0)
        for child in layout.children:
            c_size = self.semantics.get_size(child, extent_resolver).sum()
            result = result + c_size
        return result.extend(0)

    def pick_child(
        self,
        starting_layout: dlt.StructLayoutAttr,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout, int]:
        if (
            sum(
                1
                for c in starting_layout.children
                if c.contents_type.has_selectable(members, dim_mapping.keys()) > 0
            )
            != 1
        ):
            raise ValueError(
                f"Cannot select ambiguously, but there are multiple possible children that could be ment by selecting: "
                f"{members} and {dim_mapping.keys()} in {[c.contents_type for c in starting_layout.children]}"
            )
        child = None
        child_idx = None
        offset = NumericResult.from_mixed([], 0)
        for i, child_layout in enumerate(starting_layout.children):
            child_layout: dlt.Layout = child_layout
            if (
                child_layout.contents_type.has_selectable(members, dim_mapping.keys())
                > 0
            ):
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
            starting_layout, members, dim_mapping, extent_resolver, input_ptr
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
            starting_layout, members, dim_mapping, extent_resolver, input_ptr
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
            starting_layout, members, dim_mapping, extent_resolver, input_ptr
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
        fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        ops: list[Operation] = []
        ptr_ops, ptr, child, child_idx = self.pick_child(
            layout, members, dim_mapping, extent_resolver, input_ptr
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
                        {},
                        get_index_range_callback,
                        [val],
                        False,
                        False,
                        reversed_direction=True,
                    )
                )
                if_op = scf.If(
                    running_known,
                    [IntegerType(1), base_type],
                    [scf.Yield(true_op.result, running_val)],
                    lin_iter_ops + [scf.Yield(exited_early, callback_results[0])],
                )
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
        initial_values: dict[dlt.TypeType, SSAValue],
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
            child_offset = self.semantics.get_size(child, extent_resolver).sum()
            ptr_ops, current_input_ptr = child_offset.add_to_llvm_pointer(
                current_input_ptr
            )
            ops.extend(ptr_ops)
        return ops, current_callback_args

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_values: dict[dlt.TypeType, SSAValue],
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
                selected_values,
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
            ops.append(if_op)
            current_exit = if_op.output[0]
            current_ptr = if_op.output[1]
            current_callback_args = list(if_op.output[2: 2 + len(callback_args)])

        return ops, current_callback_args, current_exit


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
        starting_layout: T,
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
        starting_layout: T,
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
        fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
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
        initial_values: dict[dlt.TypeType, SSAValue],
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:
        zero_const: NumericResult1 = NumericResult.from_mixed([], 0)
        ops, (zero,) = zero_const.output()

        select_ops, new_init_values = self._select_values_selector(
            initial_values, [], {layout.dimension: zero}
        )
        ops.extend(select_ops)

        new_callback = DerivedCallback(init_callback, set(), {layout.dimension: zero})

        child_ops, iter_args_out = self.semantics.init_layout(
            layout.child,
            extent_resolver,
            input_ptr,
            new_init_values,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
        )
        ops.extend(child_ops)
        return ops, iter_args_out

    def linear_iterate(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_values: dict[dlt.TypeType, SSAValue],
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction: bool,
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        zero_const: NumericResult1 = NumericResult.from_mixed([], 0)
        ops, (zero,) = zero_const.output()

        select_ops, new_init_values = self._select_values_selector(
            selected_values, [], {layout.dimension: zero}
        )
        ops.extend(select_ops)

        new_callback = DerivedCallback(callback, set(), {layout.dimension: zero})

        child_ops, iter_args_out, exited = self.semantics.linear_iterate(
            layout.child,
            extent_resolver,
            input_ptr,
            new_init_values,
            new_callback,
            callback_args,
            has_extra_space,
            is_last_element,
            reversed_direction,
        )
        ops.extend(child_ops)
        return ops, iter_args_out, exited


class IndexingSemantics(DirectLayoutNodeSemantics[dlt.IndexingLayoutAttr]):

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
        # get index range from direct

        # pass index range to UnpackedCOO to get value

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

        extract_ops, start, end = _extract_indices_from_index_range(index_range)
        ops.extend(extract_ops)

        idx_buffer_ptr_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        ops.append(idx_buffer_ptr_op)

        while_op, index_is_found, index = UnpackCOOSemantics.loop_to_find_index(
            starting_layout.indexedChild,
            start,
            end,
            idx_buffer_ptr_op.dereferenced_value,
            dim_mapping,
        )
        ops.append(while_op)

        ptr_size = _get_accepted_type_size(IndexType()).sum()
        data_buffer_ptr_ops, data_buffer_ptr_ptr = ptr_size.add_to_llvm_pointer(
            input_ptr
        )
        ops.extend(data_buffer_ptr_ops)
        data_buffer_ptr_op = llvm.LoadOp(
            data_buffer_ptr_ptr, llvm.LLVMPointerType.opaque()
        )
        ops.append(data_buffer_ptr_op)
        data_ptr_ops, child_elem_ptr = (
            NumericResult.from_ssa(index)
            * self.semantics.get_size(
                starting_layout.indexedChild.child, extent_resolver
            ).keep(1)
        ).add_to_llvm_pointer(data_buffer_ptr_op.dereferenced_value)
        ops.extend(data_ptr_ops)

        # If index is found we can return it, else we return an appropriate zero element - non-zero sparse elements
        # are not supported (yet)
        if_found_true = []

        child_dim_mapping = {
            dim: val
            for dim, val in dim_mapping.items()
            if dim not in starting_layout.indexedChild.dimensions
            and dim not in direct_dim_mapping
        }
        child_members = members - direct_members
        child_ops, child_res, child_found = self.semantics.get_getter_for(
            starting_layout.indexedChild.child,
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

        # 3) if the direct child doesn't have a range, we must ask it to make one.
        if_index_range_found_true = [scf.Yield(index_range)]
        if_index_range_found_false = []

        def current_fill_value_getter() -> tuple[list[Operation], SSAValue]:
            f_ops, zero_idx_range = _get_packed_zero_for_accepted_type(
                dlt.IndexRangeType()
            )
            return f_ops, zero_idx_range

        ensure_space_ops, new_index_range = self.semantics.ensure_space(
            starting_layout.directChild,
            starting_layout.indexedChild.indexed_by(),
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
        ops.append(if_index_range_found_op)
        true_index_range = if_index_range_found_op.output[0]

        # 4) with the index-range, we must look for the matching element in the sparse index buffer
        extract_ops, start, end = _extract_indices_from_index_range(true_index_range)
        ops.extend(extract_ops)

        idx_buffer_ptr_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        ops.append(idx_buffer_ptr_op)
        idx_buffer_ptr = idx_buffer_ptr_op.dereferenced_value

        while_op, index_is_found, index = UnpackCOOSemantics.loop_to_find_index(
            starting_layout.indexedChild,
            start,
            end,
            idx_buffer_ptr,
            dim_mapping,
        )
        ops.append(while_op)
        # index_is_found iff we find an index tuple that matches the sparse dimension values from dim_mapping index
        # will be the index offset in the index-buffer (and data-buffer) of the match if there is one, else it will
        # be the index it was expected to be (i.e. the index where the loop stopped because either the range ran out,
        # or the index tuples got larger than the values in dim_mapping)

        # 5) Now get the data buffer ptr
        ptr_size = _get_accepted_type_size(IndexType()).sum()
        data_buffer_ptr_ops, data_buffer_ptr_ptr = ptr_size.add_to_llvm_pointer(
            input_ptr
        )
        ops.extend(data_buffer_ptr_ops)
        data_buffer_ptr_op = llvm.LoadOp(
            data_buffer_ptr_ptr, llvm.LLVMPointerType.opaque()
        )
        ops.append(data_buffer_ptr_op)
        data_buffer_ptr = data_buffer_ptr_op.dereferenced_value

        # 6) If index is found we can get the ptr for the child, else we need to reallocate and move lots of indices
        # around
        if_found_true = []

        data_ptr_ops, child_elem_ptr = (
            NumericResult.from_ssa(index)
            * self.semantics.get_size(
                starting_layout.indexedChild.child, extent_resolver
            ).keep(1)
        ).add_to_llvm_pointer(data_buffer_ptr)
        if_found_true.extend(data_ptr_ops)

        if_found_true.append(scf.Yield(child_elem_ptr))

        # If the value is not found we need to increment the indices of everything larger than this in the direct
        # data, then allocate new buffers with extra size for the new element, and mem copy the data across leaving
        # room for the new element. Then we init the new element and can continue setting from there.
        if_found_false = []

        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        zero_op = arith.Constant(IntegerAttr(0, IndexType()))
        if_found_false.extend([true_op, false_op, zero_op])

        increment_indices_callback = UnpackCOOSemantics.SparseIndexIncrementCallback(
            dict(direct_dim_mapping),
            set(direct_members),
            false_op.result,
            zero_op.result,
            zero_op.result,
        )

        increment_ops, increment_results, _increment_exited = (
            self.semantics.linear_iterate(
                starting_layout.directChild,
                extent_resolver,
                direct_data_ptr,
                {},
                increment_indices_callback,
                increment_indices_callback.initial_iter_args(),
                true_op.result,
                True,
            )
        )
        if_found_false.extend(increment_ops)
        # index_found = increment_results[0] - unused as this is for iter-callback communication - not a result
        last_index = increment_results[1]
        # last_index will be the number of non-zero elements now in the sparse buffer
        # found_index = increment_results[2]
        # found_index will be the index of the element we care about - the start of the range in the direct child
        # This is not actually required and should be removed

        # Malloc the new Unpacked COO index Buffer:
        malloc_ops, new_idx_buffer_ptr, new_data_buffer_ptr = (
            UnpackCOOSemantics.malloc_buffers(
                starting_layout.indexedChild,
                last_index,
                self.semantics.get_size(
                    starting_layout.indexedChild.child, extent_resolver
                ),
            )
        )
        if_found_false.extend(malloc_ops)

        # Set up for lots of memory copies to move the sparse index tuples and child data into their new larger buffers
        # Starting with Index tuples
        index_tuple_size_ops, (index_tuple_size,) = (
            _get_accepted_type_size(IndexType()).keep(1)
            * NumericResult.from_const(len(starting_layout.indexedChild.dimensions))
        ).output()
        if_found_false.extend(index_tuple_size_ops)

        idx_copy_ops, idx_elem_ptr = UnpackCOOSemantics.copy_to_new_buffers(
            idx_buffer_ptr,
            new_idx_buffer_ptr,
            index_tuple_size,
            index,
            last_index,
            zero_op.result,
        )
        if_found_false.extend(idx_copy_ops)

        data_elem_size_ops, (data_elem_size, data_elem_size_extra) = (
            self.semantics.get_size(
                starting_layout.indexedChild.child, extent_resolver
            ).output()
        )
        if_found_false.extend(data_elem_size_ops)

        data_copy_ops, data_elem_ptr = UnpackCOOSemantics.copy_to_new_buffers(
            data_buffer_ptr,
            new_data_buffer_ptr,
            data_elem_size,
            index,
            last_index,
            data_elem_size_extra,
        )
        if_found_false.extend(data_copy_ops)

        # New we fill in the new index tuple and data elem

        running_idx_buffer_ptr = idx_elem_ptr
        for dim in starting_layout.indexedChild.dimensions:
            idx_arg_ops, (idx_arg,) = dim_mapping[dim].get().output()
            if_found_false.extend(idx_arg_ops)
            if_found_false.append(llvm.StoreOp(idx_arg, running_idx_buffer_ptr))
            idx_size = _get_accepted_type_size(IndexType()).sum()
            new_idx_ptr_ops, running_idx_buffer_ptr = idx_size.add_to_llvm_pointer(
                running_idx_buffer_ptr
            )
            if_found_false.extend(new_idx_ptr_ops)

        # this is a new data element and so it needs to be init-ed
        child_init_ops, _ = self.semantics.init_layout(
            starting_layout.indexedChild.child,
            extent_resolver,
            data_elem_ptr,
            {},
            NoCallback(),
            [],
            false_op.result,
            True,
        )

        # The new buffers need to be put in the indexing struct and the old ones freed
        if_found_false.append(llvm.StoreOp(new_idx_buffer_ptr, input_ptr))
        if_found_false.append(llvm.StoreOp(new_data_buffer_ptr, data_buffer_ptr_ptr))
        if_found_false.append(llvm.CallOp("free", idx_buffer_ptr))
        if_found_false.append(llvm.CallOp("free", data_buffer_ptr))

        if_found_false.extend(child_init_ops)
        if_found_false.append(scf.Yield(data_elem_ptr))

        if_found_op = scf.If(
            index_is_found,
            [llvm.LLVMPointerType.opaque()],
            if_found_true,
            if_found_false,
        )
        ops.append(if_found_op)
        resulting_data_elem_ptr = if_found_op.results[0]

        # get the members and dim_mapping that will be needed by the recursive child get_setter_for
        child_dim_mapping = {
            dim: val
            for dim, val in dim_mapping.items()
            if dim not in starting_layout.indexedChild.dimensions
            and dim not in direct_dim_mapping
        }
        child_members = members - direct_members
        # and finally we can set the value in this child element
        child_setter_ops = self.semantics.get_setter_for(
            starting_layout.indexedChild.child,
            set_val,
            child_members,
            child_dim_mapping,
            extent_resolver,
            resulting_data_elem_ptr,
        )
        ops.extend(child_setter_ops)

        return ops

    def _get_fill_value_getter(
        self,
        child_layout: dlt.Layout,
        extent_resolver: ExtentResolver,
        parent_fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
        base_type: dlt.AcceptedTypes,
        data_buffer_ptr: SSAValue,
        data_elem_size: SSAValue,
        index: SSAValue,
    ) -> Callable[[], tuple[list[Operation], SSAValue]]:

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
                    child_layout,
                    extent_resolver,
                    f_elem_ptr,
                    {},
                    get_index_range_callback,
                    [w_index_range],
                    False,
                    False,
                    reversed_direction=True,
                )
            )
            while_block.add_ops(w_lin_iter_ops)
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

    def ensure_space(
        self,
        layout: dlt.IndexingLayoutAttr,
        base_type: dlt.AcceptedTypes,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        fill_value_getter: Callable[[], tuple[list[Operation], SSAValue]],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        ops = []
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
        ops.append(if_index_range_found_op)
        true_index_range = if_index_range_found_op.output[0]

        extract_ops, start, end = _extract_indices_from_index_range(true_index_range)
        ops.extend(extract_ops)

        idx_buffer_ptr_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        ops.append(idx_buffer_ptr_op)
        idx_buffer_ptr = idx_buffer_ptr_op.dereferenced_value

        while_op, index_is_found, index = UnpackCOOSemantics.loop_to_find_index(
            layout.indexedChild,
            start,
            end,
            idx_buffer_ptr,
            dim_mapping,
        )
        ops.append(while_op)

        # now get the data buffer ptr
        ptr_size = _get_accepted_type_size(IndexType()).sum()
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
            dim: val
            for dim, val in dim_mapping.items()
            if dim not in layout.indexedChild.dimensions
            and dim not in direct_dim_mapping
        }
        child_members = members - direct_members

        data_elem_size_ops, (data_elem_size, data_elem_size_extra) = (
            self.semantics.get_size(layout.indexedChild.child, extent_resolver).output()
        )
        ops.extend(data_elem_size_ops)

        child_fill_value_getter = self._get_fill_value_getter(
            layout.indexedChild.child,
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
            layout.indexedChild.child,
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

        increment_indices_callback = UnpackCOOSemantics.SparseIndexIncrementCallback(
            dict(direct_dim_mapping),
            set(direct_members),
            false_op.result,
            zero_op.result,
            zero_op.result,
        )

        increment_ops, increment_results, _increment_exited = (
            self.semantics.linear_iterate(
                layout.directChild,
                extent_resolver,
                direct_data_ptr,
                {},
                increment_indices_callback,
                increment_indices_callback.initial_iter_args(),
                true_op.result,
                True,
            )
        )
        if_found_false.extend(increment_ops)
        # index_found = increment_results[0]
        # index_found (bool) is not used here as it is an internal check for inside the callback.
        last_index = increment_results[1]
        # last_index will be the number of non-zero elements now in the sparse buffer
        # found_index = increment_results[2]
        # found_index will be the index of the element we care about

        # Malloc the new Unpacked COO index Buffer:
        malloc_ops, new_idx_buffer_ptr, new_data_buffer_ptr = (
            UnpackCOOSemantics.malloc_buffers(
                layout.indexedChild,
                last_index,
                self.semantics.get_size(layout.indexedChild.child, extent_resolver),
            )
        )
        if_found_false.extend(malloc_ops)

        # Set up for lots of memory copies to move the sparse index tuples and child data into their new larger buffers
        # Starting with Index tuples
        # Get their size as it's used repeatedly lots.
        index_tuple_size_ops, (index_tuple_size,) = (
            _get_accepted_type_size(IndexType()).keep(1)
            * NumericResult.from_const(len(layout.indexedChild.dimensions))
        ).output()
        if_found_false.extend(index_tuple_size_ops)

        idx_copy_ops, new_idx_elem_ptr = UnpackCOOSemantics.copy_to_new_buffers(
            idx_buffer_ptr,
            new_idx_buffer_ptr,
            index_tuple_size,
            index,
            last_index,
            zero_op.result,
        )
        if_found_false.extend(idx_copy_ops)

        data_copy_ops, new_data_elem_ptr = UnpackCOOSemantics.copy_to_new_buffers(
            data_buffer_ptr,
            new_data_buffer_ptr,
            data_elem_size,
            index,
            last_index,
            data_elem_size_extra,
        )
        if_found_false.extend(data_copy_ops)

        # New we fill in the new index tuple and data elem

        running_idx_buffer_ptr = new_idx_elem_ptr
        for dim in layout.indexedChild.dimensions:
            idx_arg_ops, (idx_arg,) = dim_mapping[dim].get().output()
            if_found_false.extend(idx_arg_ops)
            if_found_false.append(llvm.StoreOp(idx_arg, running_idx_buffer_ptr))
            idx_size = _get_accepted_type_size(IndexType()).sum()
            new_idx_ptr_ops, running_idx_buffer_ptr = idx_size.add_to_llvm_pointer(
                running_idx_buffer_ptr
            )
            if_found_false.extend(new_idx_ptr_ops)

        # This is a new data element and so it needs to be init-ed. we use a callback to write empty index range into
        # the But first we need to find out what index we should be using to fill in the ranges. To do this we loop
        # backwards over the data buffer children to find the first IndexRange, and failing that use the provided
        # fill_value_getter_function
        get_fill_value_ops, child_fill_value = child_fill_value_getter()
        if_found_false.extend(get_fill_value_ops)

        init_callback = UnpackCOOSemantics.WriteEmptyRangeCallback(child_fill_value)
        child_init_ops, _ = self.semantics.init_layout(
            layout.indexedChild.child,
            extent_resolver,
            new_data_elem_ptr,
            {},
            init_callback,
            [],
            false_op.result,
            True,
        )
        if_found_false.extend(child_init_ops)

        # The new buffers need to be put in the indexing struct and the old ones freed
        if_found_false.append(llvm.StoreOp(new_idx_buffer_ptr, input_ptr))
        if_found_false.append(llvm.StoreOp(new_data_buffer_ptr, data_buffer_ptr_ptr))
        if_found_false.append(llvm.CallOp("free", idx_buffer_ptr))
        if_found_false.append(llvm.CallOp("free", data_buffer_ptr))

        child_set_ops = self.semantics.get_setter_for(
            layout.indexedChild.child,
            child_fill_value,
            child_members,
            child_dim_mapping,
            extent_resolver,
            new_data_elem_ptr,
        )
        if_found_false.extend(child_set_ops)

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
        layout: dlt.IndexingLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.TypeType, SSAValue],
        init_callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
    ) -> tuple[list[Operation], list[SSAValue]]:

        ops: list[Operation] = []
        true_const = arith.Constant(IntegerAttr(1, IntegerType(1)))
        true_const.attributes["debug"] = builtin.StringAttr(
            "Indexing Semantics Init Layout start"
        )
        ops.append(true_const)
        ptr_space = self.semantics.get_size(layout.indexedChild, extent_resolver).sum()
        ptr_ops, direct_data_ptr = ptr_space.add_to_llvm_pointer(input_ptr)
        ops.extend(ptr_ops)

        running_total_initial_zero = arith.Constant(IntegerAttr(0, IndexType()))
        ops.append(running_total_initial_zero)
        set_idx_callback = UnpackCOOSemantics.SparseIndexWritingCallback(
            running_total_initial_zero.result,
            initial_values,
            layout.indexedChild.dimensions,
        )

        direct_init_ops, total_size_iter_args = self.semantics.init_layout(
            layout.directChild,
            extent_resolver,
            direct_data_ptr,
            initial_values,
            set_idx_callback,
            [running_total_initial_zero.result],
            true_const.result,
            False,
        )
        ops.extend(direct_init_ops)
        total_size = total_size_iter_args[0]

        malloc_ops, idx_buffer_ptr, data_buffer_ptr = UnpackCOOSemantics.malloc_buffers(
            layout.indexedChild,
            total_size,
            self.semantics.get_size(layout.indexedChild.child, extent_resolver),
        )
        ops.extend(malloc_ops)

        idx_buffer_ptr_ptr = input_ptr
        ops.append(llvm.StoreOp(idx_buffer_ptr, idx_buffer_ptr_ptr))

        ptr_size = _get_accepted_type_size(IndexType()).sum()
        ptr_ops, data_buffer_ptr_ptr = ptr_size.add_to_llvm_pointer(input_ptr)
        ops.extend(ptr_ops)
        ops.append(llvm.StoreOp(data_buffer_ptr, data_buffer_ptr_ptr))
        data_element_size_ops, (data_element_size,) = (
            self.semantics.get_size(layout.indexedChild.child, extent_resolver)
            .keep(1)
            .output()
        )
        ops.extend(data_element_size_ops)

        set_data_callback = UnpackCOOSemantics.SparseDataWritingCallback(
            self.semantics,
            layout.indexedChild,
            idx_buffer_ptr,
            data_buffer_ptr,
            running_total_initial_zero.result,
            total_size,
            initial_values,
            data_element_size,
            init_callback,
            callback_args,
        )

        set_data_ops, iter_args, _exited = self.semantics.linear_iterate(
            layout.directChild,
            extent_resolver,
            direct_data_ptr,
            {},
            set_data_callback,
            set_data_callback.initial_iter_args(),
            true_const.result,
            is_last_element,
        )
        ops.extend(set_data_ops)

        debug_const = arith.Constant(IntegerAttr(100, IntegerType(20)))
        debug_const.attributes["debug"] = builtin.StringAttr(
            "Indexing Semantics Init Layout end"
        )
        ops.append(debug_const)

        return ops, []

    def linear_iterate(
        self,
        layout: dlt.IndexingLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_values: dict[dlt.TypeType, SSAValue],
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

        idx_buffer_op = llvm.LoadOp(input_ptr, llvm.LLVMPointerType.opaque())
        idx_buffer_ptr = idx_buffer_op.dereferenced_value
        ops.append(idx_buffer_op)
        data_buffer_ops, data_buffer_ptr = (
            _get_accepted_type_size(IndexType()).sum().add_to_llvm_pointer(input_ptr)
        )
        ops.extend(data_buffer_ops)

        data_elem_size_ops, (data_elem_size,) = (
            self.semantics.get_size(layout.indexedChild.child, extent_resolver)
            .keep(1)
            .output()
        )
        ops.extend(data_elem_size_ops)

        linear_iter_callback: Callback = UnpackCOOSemantics.SparseLinearIterCallback(
            self.semantics,
            layout.indexedChild,
            idx_buffer_ptr,
            data_buffer_ptr,
            data_elem_size,
            callback,
            callback_args,
            is_last_element,
            reversed_direction,
        )
        return self.semantics.linear_iterate(
            layout.directChild,
            extent_resolver,
            direct_data_ptr,
            selected_values,
            linear_iter_callback,
            linear_iter_callback.initial_iter_args(),
            true_op.result,
            true_op.result,
            reversed_direction,
        )


class UnpackCOOSemantics(IndexedLayoutNodeSemantics[dlt.UnpackedCOOLayoutAttr]):

    def get_size(
        self, layout: dlt.UnpackedCOOLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        # [index_buffer, data_buffer]
        return _get_accepted_type_size(IndexType()) * NumericResult.from_mixed([], 2)

    def linear_iterate(
        self,
        layout: dlt.UnpackedCOOLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        selected_values: dict[dlt.TypeType, SSAValue],
        callback: Callback,
        callback_args: list[SSAValue],
        has_extra_space: SSAValue,
        is_last_element: bool | SSAValue,
        reversed_direction,
    ) -> tuple[list[Operation], list[SSAValue]]:
        raise NotImplementedError

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
            _get_accepted_type_size(IndexType()).sum()
            * NumericResult.from_const(num_dims)
            * NumericResult.from_ssa(before_index)
        ).output()
        if_in_range_true.extend(index_offset_ops)

        for i, dim in enumerate(upcoo_layout.dimensions):
            stored_idx_ptr_ops, stored_idx_ptr = (
                NumericResult.from_ssa(index_offset)
                + (
                    NumericResult.from_const(i)
                    * (_get_accepted_type_size(IndexType()).sum())
                )
            ).add_to_llvm_pointer(index_buffer_ptr)
            if_in_range_true.extend(stored_idx_ptr_ops)
            inner_load_stored_index = llvm.LoadOp(stored_idx_ptr, IndexType())
            if_in_range_true.append(inner_load_stored_index)
            get_index_ops, (get_index,) = dim_mapping[dim].get().output()
            if_in_range_true.extend(get_index_ops)

            inner_is_over_cmp = arith.Cmpi(
                inner_load_stored_index.dereferenced_value, get_index, "ugt"
            )
            if_in_range_true.append(inner_is_over_cmp)
            inner_is_found_cmp = arith.Cmpi(
                inner_load_stored_index.dereferenced_value, get_index, "eq"
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

    @staticmethod
    def malloc_buffers(
        upcoo_layout: dlt.UnpackedCOOLayoutAttr,
        num_non_zero: SSAValue,
        child_size: NumericResult2,
    ) -> tuple[list[Operation], SSAValue, SSAValue]:
        ops = []
        # Malloc the Unpacked COO index Buffer:
        alloc_idx_buffer_bytes_numeric = (
            _get_accepted_type_size(IndexType()).sum()
            * NumericResult.from_mixed([], len(upcoo_layout.dimensions))
            * NumericResult.from_mixed([], num_non_zero)
        )
        alloc_idx_buffer_bytes_ops, (alloc_idx_buffer_bytes,) = (
            alloc_idx_buffer_bytes_numeric.output()
        )
        ops.extend(alloc_idx_buffer_bytes_ops)
        malloc_idx_buffer = llvm.CallOp(
            "malloc", alloc_idx_buffer_bytes, return_type=llvm.LLVMPointerType.opaque()
        )
        ops.append(malloc_idx_buffer)

        # Malloc the Unpacked COO data Buffer:
        alloc_data_buffer_bytes_numeric = (
            child_size * NumericResult.from_mixed([], num_non_zero)
        ).sum()
        alloc_data_buffer_ops, (alloc_data_buffer_bytes,) = (
            alloc_data_buffer_bytes_numeric.output()
        )
        ops.extend(alloc_data_buffer_ops)
        malloc_data_buffer = llvm.CallOp(
            "malloc", alloc_data_buffer_bytes, return_type=llvm.LLVMPointerType.opaque()
        )
        ops.append(malloc_data_buffer)

        return ops, malloc_idx_buffer.returned, malloc_data_buffer.returned

    @staticmethod
    def copy_to_new_buffers(
        old_buffer_ptr: SSAValue,
        new_buffer_ptr: SSAValue,
        element_size: SSAValue,
        new_elem_index: SSAValue,
        last_index: SSAValue,
        extra_space: SSAValue,
    ) -> tuple[list[Operation], SSAValue]:
        ops = []
        # calculate how many bytes to copy before the new element
        pre_elem_idx_buffer_size_ops, (pre_elem_idx_buffer_size,) = (
            NumericResult.from_ssa(element_size)
            * NumericResult.from_ssa(new_elem_index)
        ).output()
        ops.extend(pre_elem_idx_buffer_size_ops)
        # Copy the data before the new element
        pre_elem_idx_copy_op = llvm.CallOp(
            "memcpy",
            new_buffer_ptr,
            old_buffer_ptr,
            pre_elem_idx_buffer_size,
            return_type=None,
        )
        ops.append(pre_elem_idx_copy_op)

        # calculate the ptr in the old buffer - the rest that wasn't copied before
        post_elem_idx_buffer_ptr_ops, post_elem_idx_buffer_ptr = NumericResult.from_ssa(
            pre_elem_idx_buffer_size
        ).add_to_llvm_pointer(old_buffer_ptr)
        ops.extend(post_elem_idx_buffer_ptr_ops)
        # calculate the ptr in the new buffer, adding an extra element_size to give room for the new element
        new_post_elem_idx_buffer_ptr_ops, new_post_elem_idx_buffer_ptr = (
            NumericResult.from_ssa(pre_elem_idx_buffer_size)
            + NumericResult.from_ssa(element_size)
        ).add_to_llvm_pointer(new_buffer_ptr)
        ops.extend(new_post_elem_idx_buffer_ptr_ops)
        # calculate how much data to move, this will be the new last_index (number of elements in the new buffer)
        # minus what was copied already (new_elem_index), minus 1 to account for 1 new element
        post_elem_idx_buffer_size_ops, (post_elem_idx_buffer_size,) = ((
            NumericResult.from_ssa(element_size)
            * (
                (
                    NumericResult.from_ssa(last_index)
                    - NumericResult.from_ssa(new_elem_index)
                )
                - NumericResult.from_const(1)
            )
        ) + NumericResult.from_ssa(extra_space)).output()
        ops.extend(post_elem_idx_buffer_size_ops)
        # Copy the elements after the new element
        pre_elem_idx_copy_op = llvm.CallOp(
            "memcpy",
            new_post_elem_idx_buffer_ptr,
            post_elem_idx_buffer_ptr,
            post_elem_idx_buffer_size,
            return_type=None,
        )
        ops.append(pre_elem_idx_copy_op)

        # calculate the ptr to the new_element to return
        new_elem_ptr_ops, new_elem_ptr = NumericResult.from_ssa(
            pre_elem_idx_buffer_size
        ).add_to_llvm_pointer(new_buffer_ptr)
        ops.extend(new_elem_ptr_ops)
        return ops, new_elem_ptr

    class SparseIndexWritingCallback(Callback):

        def __init__(
            self,
            # typetype: dlt.TypeType,
            init_total: SSAValue,
            sparse_initial_values: dict[dlt.TypeType, SSAValue],
            sparse_dims: list[dlt.DimensionAttr],
        ):
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
            selected_data: dict[dlt.TypeType, SSAValue],
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            # This callback keeps track of a running total of non-zeros, and sets the index value to that running total
            # as we go
            running_total = iter_args[0]
            ops: list[Operation] = []

            # first store the current running total as the index
            store_idx_op = llvm.StoreOp(running_total, ptr)
            store_idx_op.attributes["debug"] = builtin.StringAttr(
                "                 Debug Sparse index writting Callback"
            )
            ops.append(store_idx_op)

            # next we must loop over the sparse dimensions of the indexed node, to do this we set up the extents and
            # dimensions specifications for the iter_op
            # We set up the data to iterate over by first selecting based on the callback context (dim_map & members)
            dlt_ptrs = []
            for elem, dlt_ptr in self.sparse_initial_values.items():
                select_op = dlt.SelectOp(
                    dlt_ptr,
                    members,
                    builtin.ArrayAttr(dim_map.keys()),
                    list(dim_map.values()),
                )
                ops.append(select_op)
                dlt_ptrs.append(select_op.res)

            iter_ops, iter_op = _make_iterate(
                self.sparse_dims, extent_resolver, dlt_ptrs, [running_total]
            )
            ops.extend(iter_ops)

            block = iter_op.body.block
            block.erase_op(block.last_op)  # remove yield as we will add our own
            block_running_total = iter_op.get_block_arg_for_iter_arg_idx(0)

            # to detect if the element the indexed layout needs to store is non-zero, for each data source we must
            # iterate every value and set a boolean True if we find a non-zero
            inner_ptrs = (
                iter_op.get_block_args_for_tensor_args()
            )  # list(block.args[len(extents) : len(extents) + len(dlt_ptrs)])
            check_non_zero_ops, has_non_zero = UnpackCOOSemantics._check_for_non_zeros(
                inner_ptrs, extent_resolver
            )
            block.add_ops(check_non_zero_ops)

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
            extra_ptr_ops, extra_ptr = _get_accepted_type_size(
                IndexType()
            ).keep(1).add_to_llvm_pointer(ptr)
            if_has_extra_space_true.extend(extra_ptr_ops)
            store_idx_op = llvm.StoreOp(new_running_total, extra_ptr)
            store_idx_op.attributes["debug"] = builtin.StringAttr(
                "                 Debug Sparse index writing Callback for extra space"
            )
            if_has_extra_space_true.append(store_idx_op)
            if_has_extra_space_true.append(scf.Yield())
            ops.append(scf.If(has_extra_space, [], if_has_extra_space_true))

            return ops, [new_running_total], False

    @staticmethod
    def _check_for_non_zeros(
        ptrs: list[SSAValue], extent_resolver: ExtentResolver
    ) -> tuple[list[Operation], SSAValue]:
        ops: list[Operation] = []
        false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
        false_op.attributes["debug"] = builtin.StringAttr("_check_for_non_zeros Start")
        ops.append(false_op)
        true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
        ops.append(true_op)
        true = true_op.result

        has_non_zero = false_op.result

        for inner_ptr in ptrs:
            assert isinstance(inner_ptr.type, dlt.PtrType)
            inner_ptr_type = inner_ptr.type
            inner_elem = inner_ptr_type.contents_type.get_single_element()
            # We can assume each input is a single element ptr, but still need to remove extra member specifiers as
            # it doesn't matter what member a non-zero is from
            inner_members = inner_elem.member_specifiers
            inner_select = dlt.SelectOp(inner_ptr, inner_members, [], [])
            ops.append(inner_select)
            inner_dims = list(inner_elem.dimensions)
            # Now loop over all the inner dimensions and for each terminal/base_type detect non-zeros
            iter_ops, inner_iter_op = _make_iterate(
                inner_dims, extent_resolver, [inner_select.res], [has_non_zero]
            )
            ops.extend(iter_ops)
            inner_block = inner_iter_op.body.block
            inner_block.erase_op(
                inner_block.last_op
            )  # remove yield as we will add our own

            inner_has_non_zero = inner_iter_op.get_block_arg_for_iter_arg_idx(0)
            ptr = inner_iter_op.get_block_args_for_tensor_args()[0]
            assert isinstance(ptr.type, dlt.PtrType)
            elem = ptr.type.contents_type.get_single_element()
            assert isinstance(elem, dlt.ElementAttr)
            base_type = elem.base_type
            get_op = dlt.GetOp(ptr, base_type)
            inner_block.add_op(get_op)
            # non-zero condition:
            if isinstance(base_type, builtin.AnyFloat):
                zero = arith.Constant(builtin.FloatAttr(0.0, base_type))
                cond_op = arith.Cmpf(zero.result, get_op.res, "oeq")
                inner_block.add_op(zero)
                inner_block.add_op(cond_op)
            elif isinstance(base_type, builtin.AnySignlessIntegerOrIndexType):
                zero = arith.Constant(builtin.IntegerAttr(0, base_type))
                cond_op = arith.Cmpi(zero.result, get_op.res, "eq")
                inner_block.add_op(zero)
                inner_block.add_op(cond_op)
            else:
                raise NotImplementedError()

            inner_if_op = scf.If(
                cond_op.result,
                [IntegerType(1)],
                [scf.Yield(inner_has_non_zero)],
                [scf.Yield(true)],
            )
            inner_block.add_op(inner_if_op)
            inner_block.add_op(dlt.IterateYieldOp(inner_if_op.output[0]))
            has_non_zero = inner_iter_op.res[0]

            ops[-1].attributes["debug"] = builtin.StringAttr("_check_for_non_zeros End")

        return ops, has_non_zero

    class SparseDataWritingCallback(Callback):
        def __init__(
            self,
            semantics: SemanticsMapper,
            unpack_coo_layout: dlt.UnpackedCOOLayoutAttr,
            running_idx_buffer_ptr: SSAValue,
            running_data_buffer_ptr: SSAValue,
            running_total: SSAValue,
            total_non_zeros: SSAValue,
            sparse_initial_values: dict[dlt.TypeType, SSAValue],
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
            assert all(
                isinstance(v.type, dlt.PtrType) for v in sparse_initial_values.values()
            )
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
            selected_data: dict[dlt.TypeType, SSAValue],
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
                if not e.is_static():
                    nr = extent_resolver.resolve(e)
                    ext_ops, ext = nr.output()
                    ops.extend(ext_ops)
                    extent_args.extend(ext)

            dim_specifiers = []
            dlt_ptrs = []
            inner_initial_types = []
            for dlt_type, dlt_ptr in self.sparse_initial_values.items():
                dim_specifiers.append([[d] for d in self.unpack_coo_layout.dimensions])
                select_op = dlt.SelectOp(
                    dlt_ptr,
                    members,
                    builtin.ArrayAttr(dim_map.keys()),
                    list(dim_map.values()),
                )
                ops.append(select_op)
                dlt_ptrs.append(select_op.res)
                inner_initial_types.append(
                    dlt_type.select_members(members)
                    .select_dimensions(dim_map.keys())
                    .select_dimensions(self.unpack_coo_layout.dimensions)
                )

            # Create the loop(s) over sparse dimensions
            iter_op = dlt.IterateOp(
                extents,
                extent_args,
                dim_specifiers,
                dlt_ptrs,
                iter_args,
                builtin.StringAttr("nested"),
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

            sparse_iter_ptrs = [
                iter_op.get_block_arg_for_tensor_arg_idx(i)
                for i, ptr in enumerate(iter_op.tensors)
            ]

            check_non_zero_ops, has_non_zero = UnpackCOOSemantics._check_for_non_zeros(
                sparse_iter_ptrs, extent_resolver
            )
            sparse_iter_block.add_ops(check_non_zero_ops)

            # setup If statement - if has_non_zeros:
            true_ops: list[Operation] = []
            one = NumericResult.from_mixed([], 1)
            running_total_numeric = NumericResult.from_mixed([], block_running_total)
            new_total_ops, (new_running_total,) = (running_total_numeric + one).output()
            true_ops.extend(new_total_ops)

            current_block_running_idx_buffer_ptr = block_running_idx_buffer_ptr
            for idx_arg in iter_op.get_block_args_for_extent_args():
                true_ops.append(
                    llvm.StoreOp(idx_arg, current_block_running_idx_buffer_ptr)
                )
                idx_size = _get_accepted_type_size(IndexType()).sum()
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
                {
                    dlt_type: ptr
                    for dlt_type, ptr in zip(inner_initial_types, sparse_iter_ptrs)
                },
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
            selected_data: dict[dlt.TypeType, SSAValue],
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
            extract_ops, start, end = _extract_indices_from_index_range(index_range)
            ops.extend(extract_ops)
            ops.append(one_op := arith.Constant(IntegerAttr(1, IndexType())))

            idxs_step_ops, (idxs_step,) = (
                _get_accepted_type_size(IndexType()).sum()
                * NumericResult.from_mixed([], len(self.unpack_coo_layout.dimensions))
            ).output()
            ops.extend(idxs_step_ops)
            idx_step_ops, (idx_step,) = (
                _get_accepted_type_size(IndexType()).sum().output()
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
                sparse_index_load = llvm.LoadOp(current_idx_buffer_ptr, IndexType())
                block.add_op(sparse_index_load)
                sparse_dims[dim] = sparse_index_load.dereferenced_value
                increment_idx_ops, current_idx_buffer_ptr = NumericResult.from_mixed(
                    [], idx_step
                ).add_to_llvm_pointer(current_idx_buffer_ptr)
                block.add_ops(increment_idx_ops)

            select_ops, new_selected_values = (
                LayoutNodeSemantics._select_values_selector(
                    selected_data, [], sparse_dims
                )
            )
            block.add_ops(select_ops)

            block_callback = DerivedCallback(
                self.inner_callback, members, dim_map | sparse_dims
            )

            child_ops, new_inner_callback_arg, exited_early = (
                self.semantics.linear_iterate(
                    self.unpack_coo_layout.child,
                    extent_resolver,
                    current_data_buffer_ptr,
                    new_selected_values,
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
                while_loop.results[2: 2 + len(iter_args)]
            )
            did_exit_early = while_loop.results[1]

            return ops, output_inner_callback_iter_args, did_exit_early

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
            selected_data: dict[dlt.TypeType, SSAValue],
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            if not isinstance(base_type, dlt.IndexRangeType):
                return [], iter_args, False
            if members != self.direct_members:
                return [], iter_args, False
            ops = []
            zero_op = arith.Constant(IntegerAttr(0, IndexType()))
            one_op = arith.Constant(IntegerAttr(1, IndexType()))
            false_op = arith.Constant(IntegerAttr(0, IntegerType(1)))
            true_op = arith.Constant(IntegerAttr(1, IntegerType(1)))
            ops.extend([zero_op, one_op, false_op, true_op])

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
            load_op = llvm.LoadOp(ptr, IndexType())
            if_found_true.append(load_op)
            add_op = arith.Addi(load_op.dereferenced_value, one_op.result)
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
            load_op = llvm.LoadOp(ptr, IndexType())
            if_element_found_true.append(load_op)
            if_element_found_true.append(scf.Yield(load_op.dereferenced_value))
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
                _get_accepted_type_size(IndexType()).keep(1).add_to_llvm_pointer(ptr)
            )
            if_increment_extra_true.extend(ptr_ops)
            extra_load_op = llvm.LoadOp(extra_ptr, IndexType())
            if_increment_extra_true.append(extra_load_op)
            extra_add_op = arith.Addi(extra_load_op.dereferenced_value, one_op.result)
            if_increment_extra_true.append(extra_add_op)
            extra_store_op = llvm.StoreOp(extra_add_op.result, extra_ptr)
            if_increment_extra_true.append(extra_store_op)
            last_index_op = arith.Select(
                is_last_element, extra_add_op.result, zero_op.result
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

    class WriteEmptyRangeCallback(Callback):

        def __init__(
            self,
            fill_value: SSAValue,  # the index value at the point the index was found.
        ):
            self.fill_value = fill_value
            assert isinstance(self.fill_value.type, dlt.IndexRangeType)
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
            selected_data: dict[dlt.TypeType, SSAValue],
            iter_args: list[SSAValue],
        ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
            if isinstance(base_type, dlt.IndexRangeType):
                ops, start, end = _extract_indices_from_index_range(self.fill_value)
                store_idx_op = llvm.StoreOp(start, ptr)
                store_idx_op.attributes["debug"] = builtin.StringAttr(
                    "                 Debug Sparse index Empty Range Writing Callback"
                )
                ops.append(store_idx_op)

                if_extra_space = []
                store_extra_idx_op = llvm.StoreOp(start, ptr)
                store_extra_idx_op.attributes["debug"] = builtin.StringAttr(
                    "                 Debug Sparse extra index Empty Range WritingCallback"
                )
                if_extra_space.append(store_extra_idx_op)
                if_extra_space.append(scf.Yield())
                ops.append(scf.If(has_extra_space, [], if_extra_space))
                return ops, [], False
            return [], [], False


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
        selected_data: dict[dlt.TypeType, SSAValue],
        iter_args: list[SSAValue],
    ) -> tuple[list[Operation], list[SSAValue], bool | SSAValue]:
        if base_type == self.target_type:
            getter_ops, index_range, index_range_found = self.semantics.get_getter_for(
                terminal_layout,
                base_type,
                set(),
                {},
                extent_resolver,
                ptr,
            )
            return getter_ops, [index_range], index_range_found
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
    elif isinstance(t, IndexType):
        bit_width = i64.width.data
    else:
        raise ValueError(f"Cannot get size of base element: {t}")
    num_bytes = -(bit_width // -8)
    return NumericResult.from_mixed([], num_bytes, 0)


def _get_unpacked_zero_for_accepted_type(
    t: dlt.AcceptedTypes,
) -> tuple[list[Operation], SSAValue, SSAValue | None]:
    if isinstance(t, IntegerType):
        return [op := arith.Constant(IntegerAttr(0, t))], op.result, None
    elif isinstance(t, AnyFloat):
        return [op := arith.Constant(FloatAttr(0.0, t))], op.result, None
    elif isinstance(t, IndexType):
        return [op := arith.Constant(IntegerAttr(0, t))], op.result, None
    elif isinstance(t, dlt.IndexRangeType):
        return [op := arith.Constant(IntegerAttr(0, IndexType()))], op.result, op.result
    else:
        raise ValueError(f"Cannot get zero for base element: {t}")


def _get_packed_zero_for_accepted_type(
    t: dlt.AcceptedTypes,
) -> tuple[list[Operation], SSAValue]:
    if isinstance(t, IntegerType):
        return [op := arith.Constant(IntegerAttr(0, t))], op.result
    elif isinstance(t, AnyFloat):
        return [op := arith.Constant(FloatAttr(0.0, t))], op.result
    elif isinstance(t, IndexType):
        return [op := arith.Constant(IntegerAttr(0, t))], op.result
    elif isinstance(t, dlt.IndexRangeType):
        op = arith.Constant(IntegerAttr(0, IndexType()))
        pack_ops, zero = _pack_indices_in_index_range(op.result, op.result)
        return [op] + pack_ops, zero
    else:
        raise ValueError(f"Cannot get zero for base element: {t}")


def _make_iterate(
    dimensions: list[dlt.DimensionAttr],
    extent_resolver: ExtentResolver,
    ptrs: list[SSAValue],
    iter_args: list[SSAValue],
    loop_order: builtin.StringAttr = builtin.StringAttr("nested"),
) -> tuple[list[Operation], dlt.IterateOp]:
    ops = []
    extents = [d.extent for d in dimensions]
    extent_args = []
    for e in extents:
        if not e.is_static():
            nr = extent_resolver.resolve(e)
            ext_ops, ext = nr.output()
            ops.extend(ext_ops)
            extent_args.extend(ext)
    dim_specifiers = []
    dlt_ptrs = []
    for dlt_ptr in ptrs:
        assert isinstance(dlt_ptr.type, dlt.PtrType)
        assert dlt_ptr.type.contents_type.has_selectable([], dimensions)
        dim_specifiers.append([[d] for d in dimensions])
        dlt_ptrs.append(dlt_ptr)

    iter_op = dlt.IterateOp(
        extents,
        extent_args,
        dim_specifiers,
        dlt_ptrs,
        iter_args,
        loop_order,
        None,
    )
    ops.append(iter_op)
    return ops, iter_op


def _extract_indices_from_index_range(
    val: SSAValue,
) -> tuple[list[Operation], SSAValue, SSAValue]:
    ops: list[Operation] = []
    cast = builtin.UnrealizedConversionCastOp.get(
        [val], [llvm.LLVMStructType.from_type_list([IndexType(), IndexType()])]
    )
    ops.append(cast)
    extract_start = llvm.ExtractValueOp(
        DenseArrayBase.from_list(i64, [0]), cast.outputs[0], IndexType()
    )
    ops.append(extract_start)
    extract_end = llvm.ExtractValueOp(
        DenseArrayBase.from_list(i64, [1]), cast.outputs[0], IndexType()
    )
    ops.append(extract_end)
    return ops, extract_start.res, extract_end.res


def _pack_indices_in_index_range(
    start: SSAValue,
    end: SSAValue,
) -> tuple[list[Operation], SSAValue]:
    ops: list[Operation] = []
    undef_op = llvm.UndefOp(
        llvm.LLVMStructType.from_type_list([IndexType(), IndexType()])
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


Semantic_Map = SemanticsMapper()
Semantic_Map.add_direct(dlt.PrimitiveLayoutAttr, PrimitiveSemantics(Semantic_Map))
Semantic_Map.add_direct(dlt.ConstantLayoutAttr, ConstantSemantics(Semantic_Map))
Semantic_Map.add_direct(dlt.MemberLayoutAttr, MemberSemantics(Semantic_Map))
Semantic_Map.add_direct(dlt.DenseLayoutAttr, DenseSemantics(Semantic_Map))
Semantic_Map.add_direct(dlt.StructLayoutAttr, StructSemantics(Semantic_Map))
Semantic_Map.add_direct(dlt.ArithDropLayoutAttr, ArithDropSemantics(Semantic_Map))

Semantic_Map.add_direct(dlt.IndexingLayoutAttr, IndexingSemantics(Semantic_Map))
Semantic_Map.add_indexed(dlt.UnpackedCOOLayoutAttr, UnpackCOOSemantics(Semantic_Map))
