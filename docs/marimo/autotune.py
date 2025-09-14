import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import xdsl.utils.marimo as xmo
    return (xmo,)


@app.cell(hide_code=True)
def _(mo):
    min_val = 1
    max_val = 10
    m = mo.ui.slider(min_val, max_val, value=2, label="M")
    n = mo.ui.slider(min_val, max_val, value=2, label="N")
    k = mo.ui.slider(min_val, max_val, value=2, label="K")
    return k, m, n


@app.cell(hide_code=True)
def _(a_shape, b_shape, c_shape):
    from math import prod

    a_len = prod(a_shape)
    b_len = prod(b_shape)
    c_len = prod(c_shape)
    return a_len, b_len, c_len


@app.cell(hide_code=True)
def _(k, m, mo, n):
    mo.md(
        f"""
    We can parametrize the shapes of the matrices operated on:

    {m}{m.value}

    {n}{n.value}

    {k}{k.value}
    """
    )
    return


@app.cell(hide_code=True)
def _(k, m, mo, n):
    a_shape = (m.value, k.value)
    b_shape = (k.value, n.value)
    c_shape = (m.value, n.value)

    mo.md(
        f"""

        ```
        A: {'x'.join(str(dim) for dim in a_shape)}xf64 B: {'x'.join(str(dim) for dim in b_shape)}xf64 C: {'x'.join(str(dim) for dim in c_shape)}xf64
        ```
        """
    )
    return a_shape, b_shape, c_shape


@app.cell
def _():
    from xdsl.dialects.builtin import MemRefType, ModuleOp
    from xdsl.dialects import arith, func, linalg
    from xdsl.utils.op_selector import OpSelector
    return MemRefType, ModuleOp, OpSelector, arith, func, linalg


@app.cell(hide_code=True)
def _(MemRefType, ModuleOp, arith, func, linalg):
    def build_matmul(_a_type: MemRefType, _b_type: MemRefType, _c_type: MemRefType) -> ModuleOp:
        from xdsl.builder import ImplicitBuilder
        from xdsl.dialects.builtin import (
            AffineMap,
            AffineMapAttr,
            ModuleOp,
            f64,
        )
        from xdsl.ir import Block, Region

        kernel_op = func.FuncOp("matmul", ((_a_type, _b_type, _c_type), ()))
        with ImplicitBuilder(kernel_op.body) as (a, b, c):
            # Add name hints to make it easier to track how values are lowered
            a.name_hint = "A"
            b.name_hint = "B"
            c.name_hint = "C"
            body = Region(Block(arg_types=(f64, f64, f64)))
            linalg.GenericOp(
                inputs=(a, b),
                outputs=(c,),
                body=body,
                indexing_maps=(
                    AffineMapAttr(AffineMap.from_callable(lambda m, n, k: (m, k))),
                    AffineMapAttr(AffineMap.from_callable(lambda m, n, k: (k, n))),
                    AffineMapAttr(AffineMap.from_callable(lambda m, n, k: (m, n))),
                ),
                iterator_types=(
                    linalg.IteratorTypeAttr.parallel(),
                    linalg.IteratorTypeAttr.parallel(),
                    linalg.IteratorTypeAttr.reduction(),
                ),
            )
            with ImplicitBuilder(body) as (a_val, b_val, acc_old_val):
                prod_val = arith.MulfOp(a_val, b_val).result
                acc_new_val = arith.AddfOp(acc_old_val, prod_val).result
                linalg.YieldOp(acc_new_val)
                # Add more name hints to make it easier to track how values are lowered
                a_val.name_hint = "a"
                b_val.name_hint = "b"
                acc_old_val.name_hint = "acc_old"
                prod_val.name_hint = "prod"
                acc_new_val.name_hint = "acc_new"
            func.ReturnOp()

        linalg_module = ModuleOp((kernel_op,))

        return linalg_module
    return (build_matmul,)


@app.cell(hide_code=True)
def _(MemRefType, a_shape, b_shape, build_matmul, c_shape, mo, xmo):
    from xdsl.dialects.builtin import f64

    a_type = MemRefType(f64, a_shape)
    b_type = MemRefType(f64, b_shape)
    c_type = MemRefType(f64, c_shape)

    linalg_module = build_matmul(a_type, b_type, c_type)

    mo.md(f"""

    Here is matrix multiplication defined in the `linalg` dialect, with the iteration space decoupled from the computation:

    {xmo.module_html(linalg_module)}
    """)
    return f64, linalg_module


@app.cell(hide_code=True)
def _():
    from xdsl.context import Context
    from xdsl.dialects import get_all_dialects

    linalg_ctx = Context()

    for dialect_name, dialect_factory in get_all_dialects().items():
        linalg_ctx.register_dialect(dialect_name, dialect_factory)
    return Context, linalg_ctx


@app.cell(hide_code=True)
def _(linalg_ctx, linalg_module, xmo):
    from xdsl.passes import PassPipeline
    from xdsl.transforms import arith_add_fastmath, convert_linalg_to_memref_stream

    memref_stream_passes = PassPipeline(
        [
            convert_linalg_to_memref_stream.ConvertLinalgToMemRefStreamPass(),
            arith_add_fastmath.AddArithFastMathFlagsPass(),
        ]
    )

    memref_stream_ctx, memref_stream_module, memref_stream_html = xmo.pipeline_html(
        linalg_ctx,
        linalg_module,
        tuple(("", p) for p in memref_stream_passes.passes),
    )

    memref_stream_html
    return (
        PassPipeline,
        memref_stream_ctx,
        memref_stream_module,
        memref_stream_passes,
    )


@app.cell
def _(PassPipeline, memref_stream_ctx, memref_stream_module, mo, xmo):
    from xdsl.transforms import convert_memref_stream_to_loops
    from xdsl.transforms.test_lower_linalg_to_snitch import (
        LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
        LOWER_SNITCH_STREAM_TO_ASM_PASSES,
    )

    riscv_passes = PassPipeline(
        [
            convert_memref_stream_to_loops.ConvertMemRefStreamToLoopsPass(),
            *LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
            *LOWER_SNITCH_STREAM_TO_ASM_PASSES,
        ]
    )

    riscv_ctx, riscv_module, riscv_html = xmo.pipeline_html(
        memref_stream_ctx,
        memref_stream_module,
        tuple(("", p) for p in riscv_passes.passes),
    )

    mo.md(f"""
    We can use existing passes to lower the memref_stream representation without any unroll-and-jam:

    {riscv_html}
    """)
    return riscv_ctx, riscv_module, riscv_passes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Cycle Estimation""")
    return


@app.cell
def _():
    from dataclasses import dataclass, field
    return dataclass, field


@app.cell
def _():
    from collections.abc import Iterable

    class CycleResourceQueue:

        cycles_by_resource: dict[object, int]

        def __init__(self):
            self.cycles_by_resource = {}

        def enqueue(self, cycles_by_resource: dict[object, int]):
            assert not self.cycles_by_resource.keys() & cycles_by_resource
            self.cycles_by_resource |= cycles_by_resource

        def advance(self, cycles: int = 1):
            new_cycles_by_resource = {}

            for r, c in self.cycles_by_resource.items():
                if c > cycles:
                    new_cycles_by_resource[r] = c - cycles

            self.cycles_by_resource = new_cycles_by_resource

        def wait(self, resources: Iterable[object]) -> int:
            max_cycles = max((
                cycles
                for resource in resources
                if (cycles := self.cycles_by_resource.get(resource)) is not None),
                default=0
            )
            self.advance(max_cycles)
            return max_cycles
    return (CycleResourceQueue,)


@app.cell
def _(CycleResourceQueue):
    _queue = CycleResourceQueue()
    _queue.enqueue({"f0": 3, "f1": 2})
    print(_queue.wait(("f1",)))
    print(_queue.wait(("f2",)))
    _queue.cycles_by_resource
    return


@app.cell
def _(CycleResourceQueue, dataclass, field):
    from xdsl.dialects import riscv, riscv_cf, riscv_func
    from xdsl.interpreter import Interpreter, PythonValues
    from xdsl.ir import Operation

    SKIPPED_OPS = {
        riscv.GetRegisterOp,
        riscv_cf.BranchOp,
        riscv.LabelOp,
    }
    """Ops that do not contribute to the cycle count calculation"""

    CYCLES_PER_OP = {
        riscv.MulOp: 1,
        riscv.MVOp: 1,
        riscv.LiOp: 1,
        riscv.AddOp: 1,
        riscv.FLdOp: 1,
        riscv.FMulDOp: 3,
        riscv.FAddDOp: 3,
        riscv.FMAddDOp: 3,
        riscv.FSdOp: 1,
        riscv.AddiOp: 1,
        riscv_cf.BltOp: 1,
        riscv.ReturnOp: 1,
        riscv_func.ReturnOp: 1,
    }

    @dataclass
    class SnitchCycleEstimator(Interpreter.Listener):
        cycles: int = field(default=0)
        queue: CycleResourceQueue = field(default_factory=CycleResourceQueue)

        def will_interpret_op(self, op: Operation, args: PythonValues):
            if type(op) in SKIPPED_OPS:
                return
            self.cycles += self.queue.wait(op.operand_types)

        def did_interpret_op(self, op: Operation, results: PythonValues):
            if type(op) in SKIPPED_OPS:
                return
            self.cycles += 1
            self.queue.advance()
            cycles = CYCLES_PER_OP[type(op)]
            self.queue.enqueue({
                t: cycles
                for t in op.result_types
            })
    return Interpreter, SnitchCycleEstimator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Cost Model""")
    return


@app.cell
def _():
    import abc
    return (abc,)


@app.cell
def _(ModuleOp, abc):
    class CostModel(abc.ABC):
        @abc.abstractmethod
        def estimate_cost(self, module: ModuleOp) -> float | None: ...
    return (CostModel,)


@app.cell
def _(Context, CostModel, ModuleOp):
    from math import inf
    from xdsl.passes import ModulePass

    class LensCostModel(CostModel):
        inner: CostModel
        pass_pipeline: tuple[ModulePass, ...]

        def __init__(self, inner: CostModel, pass_pipeline: tuple[ModulePass, ...]):
            self.inner = inner
            self.pass_pipeline = pass_pipeline

        def estimate_cost(self, module: ModuleOp, ctx: Context) -> int | None:
            module_copy = module.clone()
            ctx_copy = ctx.clone()

            try:
                for p in self.pass_pipeline:
                    p.apply(ctx_copy, module_copy)
            except:
                return inf

            return self.inner.estimate_cost(module_copy, ctx_copy)
    return LensCostModel, ModulePass


@app.cell
def _(Context, CostModel, Interpreter, ModuleOp, SnitchCycleEstimator):
    from xdsl.interpreters import register_implementations
    from xdsl.traits import CallableOpInterface
    from xdsl.ir import Attribute

    class SnitchCycleCostModel(CostModel):
        func_name: str
        params: tuple[Attribute, ...]

        def __init__(self, func_name: str, params: tuple[Attribute, ...]):
            self.func_name = func_name
            self.params = params

        def estimate_cost(self, module: ModuleOp, ctx: Context) -> int | None:
            snitch_cycle_estimator = SnitchCycleEstimator()
            interpreter = Interpreter(module, listeners=(snitch_cycle_estimator,))

            register_implementations(interpreter, ctx)

            op = interpreter.get_op_for_symbol(self.func_name)
            trait = op.get_trait(CallableOpInterface)
            assert trait is not None

            args = tuple(
                interpreter.value_for_attribute(attr, attr_type)
                for attr, attr_type in zip(self.params, trait.get_argument_types(op))
            )

            interpreter.call_op(op, args)

            return snitch_cycle_estimator.cycles
    return Attribute, SnitchCycleCostModel


@app.cell
def _():
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    return (DenseIntOrFPElementsAttr,)


@app.cell
def _(
    DenseIntOrFPElementsAttr,
    TensorType,
    a_len,
    a_shape,
    b_len,
    b_shape,
    c_len,
    c_shape,
    f64,
):
    a_attr = DenseIntOrFPElementsAttr.from_list(
        TensorType(f64, a_shape), [i + 1 for i in range(a_len)]
    )
    b_attr = DenseIntOrFPElementsAttr.from_list(
        TensorType(f64, b_shape), [(i + 1) / 100 for i in range(b_len)]
    )
    c_attr = DenseIntOrFPElementsAttr.from_list(TensorType(f64, c_shape), [0.0] * c_len)

    a_attr, b_attr, c_attr
    return a_attr, b_attr, c_attr


@app.cell
def _(
    SnitchCycleCostModel,
    a_attr,
    b_attr,
    c_attr,
    mo,
    riscv_ctx,
    riscv_module,
):
    snitch_cost_model = SnitchCycleCostModel("matmul", (a_attr, b_attr, c_attr))

    cycles = snitch_cost_model.estimate_cost(riscv_module, riscv_ctx)

    mo.md(f"""
    The cost of running our assembly-level code without unroll-and-jam is {cycles}.
    """)
    return


@app.cell
def _(
    LensCostModel,
    SnitchCycleCostModel,
    a_attr,
    b_attr,
    c_attr,
    memref_stream_ctx,
    memref_stream_module,
    mo,
    riscv_passes,
):
    memref_stream_cost_model = LensCostModel(
        SnitchCycleCostModel("matmul", (a_attr, b_attr, c_attr)), riscv_passes.passes
    )

    memref_stream_cost = memref_stream_cost_model.estimate_cost(
        memref_stream_module, memref_stream_ctx
    )

    mo.md(f"""
    The cost of running our memref_stream-level code without unroll-and-jam is also {memref_stream_cost}.
    """)
    return (memref_stream_cost_model,)


@app.cell
def _(
    memref_stream_cost_model,
    memref_stream_ctx,
    memref_stream_module,
    mo,
    xmo,
):
    from xdsl.transforms import memref_stream_interleave

    interleaved_ctx = memref_stream_ctx.clone()
    interleaved_module = memref_stream_module.clone()

    memref_stream_interleave.MemRefStreamInterleavePass().apply(
        interleaved_ctx, interleaved_module
    )

    interleaved_cost = memref_stream_cost_model.estimate_cost(interleaved_module, interleaved_ctx)

    mo.md(f"""
    We can use the existing interleave pass to choose the unroll-and-jam factor, with cost {interleaved_cost}:

    {xmo.module_html(interleaved_module)}
    """)
    return interleaved_cost, memref_stream_interleave


@app.cell
def _(memref_stream_module, mo):
    from xdsl.dialects import memref_stream
    from xdsl.transforms.memref_stream_interleave import PipelineGenericPattern

    msg_op = next(child for child in memref_stream_module.walk() if isinstance(child, memref_stream.GenericOp))

    msg_factors = PipelineGenericPattern.indices_and_factors(msg_op)

    mo.md(f"""
    We can also find the possible range of indices and factors to unroll-and-jam manually:

    {msg_factors}
    """)
    return (msg_factors,)


@app.cell
def _(mo, msg_factors):
    from xdsl.transforms.memref_stream_interleave import MemRefStreamInterleavePass

    uaj_passes = tuple(
        MemRefStreamInterleavePass(4, 2, index, factor)
        for index, factor in msg_factors
    )

    _passes_str = "\n".join(str(p.pipeline_pass_spec()) for p in uaj_passes)

    mo.md(f"""
    We construct passes from these indices:

    ```
    {_passes_str}
    ```
    """)
    return MemRefStreamInterleavePass, uaj_passes


@app.cell(hide_code=True)
def _(Context, ModuleOp, ModulePass):
    def apply(p: ModulePass, ctx: Context, op: ModuleOp) -> ModuleOp:
        op = op.clone()
        ctx = ctx.clone()
        p.apply(ctx, op)
        return ctx, op
    return (apply,)


@app.cell
def _(
    apply,
    memref_stream_cost_model,
    memref_stream_ctx,
    memref_stream_module,
    mo,
    uaj_passes,
):
    scores = tuple(
        memref_stream_cost_model.estimate_cost(apply(p, memref_stream_ctx, memref_stream_module)[1], memref_stream_ctx)
        for p in uaj_passes
    )

    mo.md(f"""
    We can evaluate the impact of each of the passes with our cost model:

    {scores}
    """)
    return


@app.cell
def _():
    from random import Random
    return (Random,)


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(Float32Type, Float64Type, IntAttr, IntegerType, np, ptr):
    from xdsl.dialects.builtin import PackableType

    def to_dtype(
        xtype: PackableType[int] | PackableType[float],
    ) -> type[np.int32] | type[np.int64] | type[np.float32] | type[np.float64]:
        match xtype:
            case IntegerType(width=IntAttr(data=32)):
                return np.int32
            case IntegerType(width=IntAttr(data=64)):
                return np.int64
            case Float32Type():
                return np.float32
            case Float64Type():
                return np.float64
            case _:
                raise NotImplementedError()


    def from_dtype(
        dtype: np.dtype[np.float32 | np.float64 | np.int32 | np.int64],
    ) -> PackableType[float] | PackableType[int]:
        if dtype == np.float32:
            return ptr.float32
        elif dtype == np.float64:
            return ptr.float64
        elif dtype == np.float32:
            return ptr.int32
        elif dtype == np.float64:
            return ptr.int64
        else:
            raise NotImplementedError()
    return


@app.cell
def _(Attribute, DenseIntOrFPElementsAttr, Random, f64):
    from xdsl.dialects.builtin import FloatAttr, ShapedType, ContainerType, TensorType


    def random_attr_of_type(t: Attribute, rng: Random) -> Attribute | None:
        # if isinstance(type, ShapedType):
        if t == f64:
            return FloatAttr(rng.random(), f64)
        elif isinstance(t, ShapedType) and isinstance(t, ContainerType):
            return DenseIntOrFPElementsAttr.from_list(
                t, tuple(rng.random() for _ in range(t.element_count()))
            )

    _rng = Random("autotune")
    random_attr_of_type(f64, _rng), random_attr_of_type(TensorType(f64, (2, 3)), _rng)
    return TensorType, random_attr_of_type


@app.cell
def _(
    Context,
    LensCostModel,
    MemRefStreamInterleavePass,
    ModuleOp,
    ModulePass,
    OpSelector,
    Random,
    SnitchCycleCostModel,
    func,
    random_attr_of_type,
    riscv_passes,
):
    class AutomaticUnrollAndJamPass(ModulePass):

        name = "automatic-unroll-and-jam"

        def apply(self, ctx: Context, op: ModuleOp) -> None:
            passes = MemRefStreamInterleavePass.schedule_space(ctx, op)

            if not passes:
                return

            assert len(set(p.op_index for p in passes)) == 1
            op_index = passes[0].op_index
            msg_op = OpSelector(op_index, "memref_stream.generic").get_op(op)

            func_op = msg_op.parent_op()

            assert isinstance(func_op, func.FuncOp), func_op.name

            func_op_name = func_op.sym_name.data

            arg_types = func_op.function_type.inputs

            rng = Random("autotune")
            attrs = tuple(random_attr_of_type(t, rng) for t in arg_types)

            cost_model = LensCostModel(SnitchCycleCostModel(func_op_name, attrs), riscv_passes.passes)

            scores = tuple(
                enumerate(
                    cost_model.estimate_cost(apply(p, ctx, op)[1], ctx)
                    for p in passes
                )
            )

            best_pass_index = min(scores, key=lambda x: x[1])

            passes[best_pass_index[0]].apply(ctx, op)
    return (AutomaticUnrollAndJamPass,)


@app.cell
def _(k, m, mo, n):
    mo.md(
        f"""
    Here are the sliders again:

    {m}{m.value}

    {n}{n.value}

    {k}{k.value}
    """
    )
    return


@app.cell
def _(
    AutomaticUnrollAndJamPass,
    apply,
    interleaved_cost,
    memref_stream_cost_model,
    memref_stream_ctx,
    memref_stream_module,
    mo,
    xmo,
):
    automated_ctx, automated_module = apply(AutomaticUnrollAndJamPass(), memref_stream_ctx, memref_stream_module)

    automated_cost = memref_stream_cost_model.estimate_cost(automated_module, automated_ctx)

    mo.md(f"""
    Here's the updated module with our automated approach, with cost {automated_cost} vs the heuristic's {interleaved_cost}:

    {xmo.module_html(automated_module)}
    """)
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(
    AutomaticUnrollAndJamPass,
    LensCostModel,
    MemRefType,
    Random,
    SnitchCycleCostModel,
    apply,
    build_matmul,
    f64,
    linalg_ctx,
    memref_stream_interleave,
    memref_stream_passes,
    random_attr_of_type,
    riscv_passes,
):
    keys = []
    automated_costs = []
    heuristic_costs = []

    for _n in range(4, 10):
        _a_shape = (2, 2)
        _b_shape = (2, _n)
        _c_shape = (2, _n)

        _a_type = MemRefType(f64, _a_shape)
        _b_type = MemRefType(f64, _b_shape)
        _c_type = MemRefType(f64, _c_shape)

        _linalg_ctx = linalg_ctx.clone()
        _linalg_module = build_matmul(_a_type, _b_type, _c_type)

        _memref_stream_ctx, _memref_stream_module = apply(memref_stream_passes, _linalg_ctx, _linalg_module)
        # memref_stream_passes.apply(linalg_ctx, _linalg_module)

        # automated
        _automatex_ctx, _automated_module = apply(AutomaticUnrollAndJamPass(), _memref_stream_ctx, _memref_stream_module)

        _rng = Random("autotune")
        _attrs = tuple(random_attr_of_type(t, _rng) for t in (_a_type, _b_type, _c_type))
        _memref_stream_cost_model = LensCostModel(
            SnitchCycleCostModel("matmul", _attrs), riscv_passes.passes
        )
        _automated_cost = _memref_stream_cost_model.estimate_cost(_automated_module, _automatex_ctx)

        # heuristic

        _interleaved_ctx = _memref_stream_ctx.clone()
        _interleaved_module = _memref_stream_module.clone()

        memref_stream_interleave.MemRefStreamInterleavePass().apply(
            _interleaved_ctx, _interleaved_module
        )

        _interleaved_cost = _memref_stream_cost_model.estimate_cost(_interleaved_module, _interleaved_ctx)

        keys.append(f"{_n}")
        automated_costs.append(_automated_cost)
        heuristic_costs.append(_interleaved_cost)
    return automated_costs, heuristic_costs, keys


@app.cell
def _(automated_costs, heuristic_costs, keys, pd):
    df = pd.DataFrame(index=keys)
    df["Automated"] = automated_costs
    df["Heuristic"] = heuristic_costs
    df
    return (df,)


@app.cell
def _():
    from matplotlib import pyplot as plt
    return (plt,)


@app.cell
def _(df, plt):
    # plt.xticks(rotation=45, ha='right')

    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()

    df.plot(
        title="Matrix Multiplication 2x2 * 2xN",
        ylabel='Cycles (Estimate)',
        xlabel='N',
        ylim=(0, None),
    )
    return


if __name__ == "__main__":
    app.run()
