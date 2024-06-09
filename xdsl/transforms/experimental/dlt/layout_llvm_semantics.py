import abc
import typing

from xdsl.dialects import arith, llvm, scf
from xdsl.dialects.builtin import (
    AnyFloat,
    DenseArrayBase,
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

    @typing.overload
    @staticmethod
    def from_mixed(
        operations: typing.Iterable[Operation], value1: typing.Union[int, SSAValue]
    ) -> typing.Annotated["NumericResult", 1]: ...

    @typing.overload
    @staticmethod
    def from_mixed(
        operations: typing.Iterable[Operation],
        value1: typing.Union[int, SSAValue],
        value2: typing.Union[int, SSAValue],
    ) -> typing.Annotated["NumericResult", 2]: ...

    @typing.overload
    @staticmethod
    def from_mixed(
        operations: typing.Iterable[Operation],
        value1: typing.Union[int, SSAValue],
        value2: typing.Union[int, SSAValue],
        value3: typing.Union[int, SSAValue],
    ) -> typing.Annotated["NumericResult", 3]: ...

    @typing.overload
    @staticmethod
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
        *values: tuple[typing.Union[int, SSAValue], ...],
    ) -> "NumericResult":
        static_values = tuple([v if isinstance(v, int) else 0 for v in values])
        ssa_values = tuple([v if isinstance(v, SSAValue) else None for v in values])
        result = typing.cast(
            typing.Annotated[NumericResult, len(values)],
            NumericResult(operations, static_values, ssa_values),
        )
        return result

    def __add__(self, other) -> typing.Self:
        if not isinstance(other, NumericResult):
            return NotImplemented
        if len(self.static_values) < len(other.static_values):
            raise ValueError()
        if self.used | other.used:
            raise ValueError()

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

    def __mul__(self, other) -> typing.Self:
        if not isinstance(other, NumericResult):
            return NotImplemented
        if len(self.static_values) < len(other.static_values):
            raise ValueError()
        if self.used | other.used:
            raise ValueError()

        all_ops = self.operations + other.operations
        new_static = []
        new_ssa = []
        for i, (a_s, a_d) in enumerate(zip(self.static_values, self.ssa_values)):
            if i >= len(other.ssa_values):
                new_static.append(a_s)
                new_ssa.append(a_d)
            elif other.ssa_values[i] == None and other.static_values[i] == 1:
                new_static.append(a_s)
                new_ssa.append(a_d)
            elif other.ssa_values[i] == None and other.static_values[i] == 0:
                new_static.append(0)
                new_ssa.append(None)
            elif a_d == None and a_s == 1:
                new_static.append(other.static_values[i])
                new_ssa.append(other.ssa_values[i])
            elif a_d == None and a_s == 0:
                new_static.append(0)
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
        assert (
            value.type.width.data <= i64.width.data
        ), f"Expected {i64.width.data} got {value.type.width.data}"
    return [op := UnrealizedConversionCastOp.get([value], [i64])], op.outputs[0]


def _get_as_index(value: SSAValue) -> tuple[list[Operation], SSAValue]:
    assert isinstance(value.type, IndexType | IntegerType)
    if isinstance(value.type, IntegerType):
        assert (
            value.type.width.data <= i64.width.data
        ), f"Expected {i64.width.data} got {value.type.width.data}"
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
        input: SSAValue,
        ptr_type: dlt.PtrType,
        dim: dlt.DimensionAttr,
    ):
        self.input = input
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
        input: SSAValue,
        ptr_type: dlt.PtrType,
        extent: dlt.Extent,
    ):
        self.input = input
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

    def __init__(self, map: dict[dlt.Extent, ExtentGetter]):
        self.map: dict[dlt.Extent, ExtentGetter] = map

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


class SemanticsMapper:
    def __init__(self):
        self.map: dict[typing.Type[dlt.Layout], LayoutNodeSemantics] = {}

    def add(self, typ: typing.Type[dlt.Layout], node_semantics: "LayoutNodeSemantics"):
        self.map[typ] = node_semantics

    def get(self, layout: dlt.Layout) -> "LayoutNodeSemantics":
        for t, s in reversed(self.map.items()):
            if isinstance(layout, t):
                return s
        raise KeyError(f"Cannot find semantics for layout: {layout}")

    def get_size(
        self, layout: dlt.Layout, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        result = self.get(layout).get_size(layout, extent_resolver)
        assert result.size == 2
        return result

    def get_select_for(
        self,
        starting_layout: dlt.Layout,
        ending_layout: dlt.Layout | None,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout]:
        assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        assert ending_layout is None or starting_layout.has_sub_layout(ending_layout)
        assert starting_layout.contents_type.has_selectable(members, dim_mapping.keys())
        if starting_layout == ending_layout:
            return [], input_ptr, ending_layout
        ops, val, layout_out = self.get(starting_layout).get_select_for(
            starting_layout,
            ending_layout,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
        )
        assert isinstance(val.type, llvm.LLVMPointerType)
        assert ending_layout is None or ending_layout == layout_out
        return ops, val, layout_out

    def init_layout(
        self,
        layout: dlt.Layout,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.ElementAttr, SSAValue],
    ) -> list[Operation]:
        assert isinstance(
            input_ptr.type, llvm.LLVMPointerType
        ), f"Input pointer expected to be LLVMPointerType but found {type(input_ptr.type)}"
        return self.get(layout).init_layout(
            layout, extent_resolver, input_ptr, initial_values
        )


class LayoutNodeSemantics(abc.ABC, typing.Generic[T]):

    def __init__(self, semantics: SemanticsMapper):
        self.semantics = semantics

    @abc.abstractmethod
    def get_size(self, layout: T, extent_resolver: ExtentResolver) -> NumericResult2:
        raise NotImplementedError

    @abc.abstractmethod
    def get_select_for(
        self,
        starting_layout: T,
        ending_layout: dlt.Layout | None,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout]:
        # assert isinstance(input_ptr.type, llvm.LLVMPointerType)
        # assert ending_layout is None or starting_layout.has_sub_layout(ending_layout)
        # assert starting_layout.contents_type.has_selectable(members, dim_mapping.keys())
        raise NotImplementedError

    @abc.abstractmethod
    def init_layout(
        self,
        layout: T,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.ElementAttr, SSAValue],
    ) -> list[Operation]:
        # assert isinstance(input_ptr, llvm.LLVMPointerType)
        raise NotImplementedError


class PrimitiveSemantics(LayoutNodeSemantics[dlt.PrimitiveLayoutAttr]):

    def get_size(
        self, layout: dlt.PrimitiveLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        if isinstance(layout.base_type, dlt.DLTCompatibleElementBaseType):
            p, e = layout.base_type.get_size()
            return NumericResult.from_mixed([], p, e)

        if isinstance(layout.base_type, IntegerType):
            bit_width = layout.base_type.width.data
        elif isinstance(layout.base_type, AnyFloat):
            bit_width = layout.base_type.get_bitwidth
        elif isinstance(layout.base_type, IndexType):
            bit_width = i64.width.data
        else:
            raise ValueError(f"Cannot get size of base element: {layout.base_type}")
        bytes = -(bit_width // -8)
        return NumericResult.from_mixed([], bytes, 0)

    def get_select_for(
        self,
        starting_layout: dlt.PrimitiveLayoutAttr,
        ending_layout: dlt.Layout | None,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout]:
        assert ending_layout is None or starting_layout == ending_layout
        return [], input_ptr, starting_layout

    def init_layout(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.ElementAttr, SSAValue],
    ) -> list[Operation]:
        elem = layout.contents_type.get_single_element()
        if elem not in initial_values:
            return []
        return [
            init_val := dlt.GetOp(initial_values[elem], layout.base_type),
            llvm.StoreOp(init_val, input_ptr),
        ]


class ConstantSemantics(LayoutNodeSemantics[dlt.PrimitiveLayoutAttr]):

    def get_size(
        self, layout: dlt.PrimitiveLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        return NumericResult.from_mixed([], 0, 0)

    def get_select_for(
        self,
        starting_layout: dlt.PrimitiveLayoutAttr,
        ending_layout: dlt.Layout | None,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout]:
        assert ending_layout is None or starting_layout == ending_layout
        return [null := llvm.NullOp()], null.nullptr, starting_layout

    def init_layout(
        self,
        layout: dlt.PrimitiveLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.ElementAttr, SSAValue],
    ) -> list[Operation]:
        return []


class MemberSemantics(LayoutNodeSemantics[dlt.MemberLayoutAttr]):

    def get_size(
        self, layout: dlt.MemberLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        return self.semantics.get(layout.child).get_size(layout.child, extent_resolver)

    def get_select_for(
        self,
        starting_layout: dlt.MemberLayoutAttr,
        ending_layout: dlt.Layout | None,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout]:
        assert starting_layout.member_specifier in members
        return self.semantics.get_select_for(
            starting_layout.child,
            ending_layout,
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
        initial_values: dict[dlt.ElementAttr, SSAValue],
    ) -> list[Operation]:
        ops = []
        new_init_values = {}
        for elem, dlt_ptr in initial_values.items():
            ops.append(
                select_op := dlt.SelectOp(dlt_ptr, [layout.member_specifier], [], [])
            )
            new_elem = typing.cast(
                dlt.PtrType, select_op.res.type
            ).contents_type.get_single_element()
            assert new_elem == elem.select_member(layout.member_specifier)
            new_init_values[new_elem] = select_op.res

        return ops + self.semantics.init_layout(
            layout.child, extent_resolver, input_ptr, new_init_values
        )


class DenseSemantics(LayoutNodeSemantics[dlt.DenseLayoutAttr]):

    def get_size(
        self, layout: dlt.DenseLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        child_size: NumericResult2 = self.semantics.get_size(
            layout.child, extent_resolver
        )
        extent_size: NumericResult1 = extent_resolver.resolve(layout.dimension.extent)

        return child_size * extent_size

    def get_select_for(
        self,
        starting_layout: dlt.DenseLayoutAttr,
        ending_layout: dlt.Layout | None,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout]:

        child_layout = starting_layout.child

        child_size: NumericResult1 = self.semantics.get_size(
            child_layout, extent_resolver
        ).keep(1)
        dim_val: NumericResult1 = dim_mapping.pop(starting_layout.dimension).get()
        offset: NumericResult1 = child_size * dim_val

        ptr_ops, ptr = offset.add_to_llvm_pointer(input_ptr)
        child_ops, child_res, out_laout = self.semantics.get_select_for(
            child_layout, ending_layout, members, dim_mapping, extent_resolver, ptr
        )
        return ptr_ops + child_ops, child_res, out_laout

    def init_layout(
        self,
        layout: dlt.DenseLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.ElementAttr, SSAValue],
    ) -> list[Operation]:
        child_size: NumericResult1 = self.semantics.get_size(
            layout.child, extent_resolver
        ).keep(1)
        ops, child_size = child_size.split()

        block = Block()  # loop body
        index = block.insert_arg(
            IndexType(), 0
        )  # index - to run through the dense dimension
        offset = child_size * NumericResult.from_mixed([], index)
        ptr_add_ops, ptr_arg = offset.add_to_llvm_pointer(input_ptr)
        block.add_ops(ptr_add_ops)

        new_init_values = {}
        for elem, dlt_ptr in initial_values.items():
            block.add_op(
                select_op := dlt.SelectOp(dlt_ptr, [], [layout.dimension], [index])
            )
            new_elem = typing.cast(
                dlt.PtrType, select_op.res.type
            ).contents_type.get_single_element()
            assert new_elem == elem.select_dimension(layout.dimension)
            new_init_values[new_elem] = select_op.res

        block.add_ops(
            self.semantics.init_layout(
                layout.child, extent_resolver, ptr_arg, new_init_values
            )
        )
        block.add_op(scf.Yield())

        ops.append(lb := arith.Constant(IntegerAttr(0, IndexType())))
        ops.append(step := arith.Constant(IntegerAttr(1, IndexType())))
        ub_ops, (ub,) = extent_resolver.resolve(layout.dimension.extent).output()
        ops.extend(ub_ops)

        loop = scf.For(lb, ub, step, [], block)
        ops.append(loop)
        return ops


class StructSemantics(LayoutNodeSemantics[dlt.StructLayoutAttr]):
    def get_size(
        self, layout: dlt.StructLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        result = NumericResult.from_mixed([], 0)
        for child in layout.children:
            c_size = self.semantics.get_size(child, extent_resolver).sum()
            result = result + c_size
        return result.extend(0)

    def get_select_for(
        self,
        starting_layout: dlt.StructLayoutAttr,
        ending_layout: dlt.Layout | None,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout]:
        if (
            len(
                [
                    b
                    for b in [
                        c.contents_type.has_selectable(members, dim_mapping.keys()) > 0
                        for c in starting_layout.children
                    ]
                    if b
                ]
            )
            != 1
        ):
            raise ValueError(
                f"Cannot select ambiguously, but there are multple possible children that could be ment by selecting: {members} and {dim_mapping.keys()} in {[c.contents_type for c in starting_layout.children]}"
            )

        child = None
        offset = NumericResult.from_mixed([], 0)
        for i, child_layout in enumerate(starting_layout.children):
            child_layout: dlt.Layout = child_layout
            if (
                child_layout.contents_type.has_selectable(members, dim_mapping.keys())
                > 0
            ):
                child = child_layout
                break
            else:
                c_offset = self.semantics.get_size(child_layout, extent_resolver).sum()
                offset = offset + c_offset

        ptr_ops, ptr = offset.add_to_llvm_pointer(input_ptr)
        child_ops, child_res, out_layout = self.semantics.get_select_for(
            child, ending_layout, members, dim_mapping, extent_resolver, ptr
        )
        return ptr_ops + child_ops, child_res, out_layout

    def init_layout(
        self,
        layout: dlt.StructLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.ElementAttr, SSAValue],
    ) -> list[Operation]:
        ops = []
        current_input_ptr = input_ptr
        for child in layout.children:
            child = typing.cast(dlt.Layout, child)
            child_initial_values = {
                elem: val
                for elem, val in initial_values.items()
                if elem in child.contents_type
            }
            ops.extend(
                self.semantics.init_layout(
                    child, extent_resolver, current_input_ptr, child_initial_values
                )
            )
            child_offset = self.semantics.get_size(child, extent_resolver).sum()
            ptr_ops, current_input_ptr = child_offset.add_to_llvm_pointer(
                current_input_ptr
            )
            ops.extend(ptr_ops)
        return ops


class ArithDropSemantics(LayoutNodeSemantics[dlt.ArithDropLayoutAttr]):

    def get_size(
        self, layout: dlt.ArithDropLayoutAttr, extent_resolver: ExtentResolver
    ) -> NumericResult2:
        return self.semantics.get_size(layout.child, extent_resolver)

    def get_select_for(
        self,
        starting_layout: dlt.ArithDropLayoutAttr,
        ending_layout: dlt.Layout | None,
        members: set[dlt.MemberAttr],
        dim_mapping: dict[dlt.DimensionAttr, IndexGetter],
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
    ) -> tuple[list[Operation], SSAValue, dlt.Layout]:

        child_layout = starting_layout.child
        dim_mapping.pop(starting_layout.dimension)
        child_ops, child_res, out_layout = self.semantics.get_select_for(
            child_layout,
            ending_layout,
            members,
            dim_mapping,
            extent_resolver,
            input_ptr,
        )
        return child_ops, child_res, out_layout

    def init_layout(
        self,
        layout: dlt.ArithDropLayoutAttr,
        extent_resolver: ExtentResolver,
        input_ptr: SSAValue,
        initial_values: dict[dlt.ElementAttr, SSAValue],
    ) -> list[Operation]:
        zero_const: NumericResult1 = NumericResult.from_mixed([], 0)
        ops, (zero,) = zero_const.output()

        new_init_values = {}
        for elem, dlt_ptr in initial_values.items():
            ops.append(
                select_op := dlt.SelectOp(dlt_ptr, [], [layout.dimension], [zero])
            )
            new_elem = typing.cast(
                dlt.PtrType, select_op.res.type
            ).contents_type.get_single_element()
            assert new_elem == elem.select_dimension(layout.dimension)
            new_init_values[new_elem] = select_op.res

        ops.extend(
            self.semantics.init_layout(
                layout.child, extent_resolver, input_ptr, new_init_values
            )
        )
        return ops


Semantic_Map = SemanticsMapper()
Semantic_Map.add(dlt.PrimitiveLayoutAttr, PrimitiveSemantics(Semantic_Map))
Semantic_Map.add(dlt.ConstantLayoutAttr, ConstantSemantics(Semantic_Map))
Semantic_Map.add(dlt.MemberLayoutAttr, MemberSemantics(Semantic_Map))
Semantic_Map.add(dlt.DenseLayoutAttr, DenseSemantics(Semantic_Map))
Semantic_Map.add(dlt.StructLayoutAttr, StructSemantics(Semantic_Map))
Semantic_Map.add(dlt.ArithDropLayoutAttr, ArithDropSemantics(Semantic_Map))
