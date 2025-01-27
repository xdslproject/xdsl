import marimo

__generated_with = "0.10.17"
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
    return k, m, max_val, min_val, n


@app.cell(hide_code=True)
def _(a_shape, b_shape, c_shape):
    from math import prod

    a_len = prod(a_shape)
    b_len = prod(b_shape)
    c_len = prod(c_shape)
    return a_len, b_len, c_len, prod


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
    from xdsl.dialects.builtin import MemRefType
    from xdsl.dialects import arith, func, linalg
    return MemRefType, arith, func, linalg


@app.cell(hide_code=True)
def _(MemRefType, arith, func, linalg):
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
    return a_type, b_type, c_type, f64, linalg_module


@app.cell(hide_code=True)
def _():
    from xdsl.context import MLContext
    from xdsl.dialects import get_all_dialects

    linalg_ctx = MLContext()

    for dialect_name, dialect_factory in get_all_dialects().items():
        linalg_ctx.register_dialect(dialect_name, dialect_factory)
    return (
        MLContext,
        dialect_factory,
        dialect_name,
        get_all_dialects,
        linalg_ctx,
    )


@app.cell(hide_code=True)
def _(linalg_ctx, linalg_module, xmo):
    from xdsl.passes import PipelinePass
    from xdsl.transforms import arith_add_fastmath, convert_linalg_to_memref_stream

    memref_stream_passes = PipelinePass(
        [
            convert_linalg_to_memref_stream.ConvertLinalgToMemrefStreamPass(),
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
        PipelinePass,
        arith_add_fastmath,
        convert_linalg_to_memref_stream,
        memref_stream_ctx,
        memref_stream_html,
        memref_stream_module,
        memref_stream_passes,
    )


@app.cell
def _(PipelinePass, memref_stream_ctx, memref_stream_module, mo, xmo):
    from xdsl.transforms import convert_memref_stream_to_loops
    from xdsl.transforms.test_lower_linalg_to_snitch import (
        LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
        LOWER_SNITCH_STREAM_TO_ASM_PASSES,
    )

    riscv_passes = PipelinePass(
        [
            convert_memref_stream_to_loops.ConvertMemrefStreamToLoopsPass(),
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
    return (
        LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
        LOWER_SNITCH_STREAM_TO_ASM_PASSES,
        convert_memref_stream_to_loops,
        riscv_ctx,
        riscv_html,
        riscv_module,
        riscv_passes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Cycle Estimation""")
    return


@app.cell
def _():
    from dataclasses import dataclass, field
    return dataclass, field


@app.cell
def _(Operation, PythonValues, dataclass, field):
    from xdsl.dialects import riscv, riscv_cf, riscv_func
    from xdsl.interpreter import Interpreter

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
        riscv.FMulDOp: 1,
        riscv.FAddDOp: 1,
        riscv.FMAddDOp: 1,
        riscv.FSdOp: 1,
        riscv.AddiOp: 1,
        riscv_cf.BltOp: 1,
        riscv.ReturnOp: 1,
        riscv_func.ReturnOp: 1,
    }

    @dataclass
    class SnitchCycleEstimator(Interpreter.Listener):
        cycles: int = field(default=0)

        def will_interpret_op(self, op: Operation, args: PythonValues):
            if type(op) in SKIPPED_OPS:
                return
            self.cycles += CYCLES_PER_OP[type(op)]
    return (
        CYCLES_PER_OP,
        Interpreter,
        SKIPPED_OPS,
        SnitchCycleEstimator,
        riscv,
        riscv_cf,
        riscv_func,
    )


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
        def estimate_cost(self, module: ModuleOp) -> int | None: ...
    return (CostModel,)


@app.cell
def _(CostModel, MLContext, ModuleOp):
    from xdsl.passes import ModulePass

    class LensCostModel(CostModel):
        inner: CostModel
        pass_pipeline: tuple[ModulePass, ...]

        def __init__(self, inner: CostModel, pass_pipeline: tuple[ModulePass, ...]):
            self.inner = inner
            self.pass_pipeline = pass_pipeline

        def estimate_cost(self, module: ModuleOp, ctx: MLContext) -> int | None:
            module_copy = module.clone()
            ctx_copy = ctx.clone()

            for p in self.pass_pipeline:
                p.apply(ctx_copy, module_copy)

            return self.inner.estimate_cost(module_copy, ctx_copy)
    return LensCostModel, ModulePass


@app.cell
def _(
    Attribute,
    CostModel,
    Interpreter,
    MLContext,
    ModuleOp,
    SnitchCycleEstimator,
):
    from xdsl.interpreters import register_implementations
    from xdsl.traits import CallableOpInterface

    class SnitchCycleCostModel(CostModel):
        func_name: str
        params: tuple[Attribute, ...]

        def __init__(self, func_name: str, params: tuple[Attribute, ...]):
            self.func_name = func_name
            self.params = params

        def estimate_cost(self, module: ModuleOp, ctx: MLContext) -> int | None:
            snitch_cycle_estimator = SnitchCycleEstimator()
            interpreter = Interpreter(module, listeners=(snitch_cycle_estimator,))

            register_implementations(
                interpreter, ctx, include_wgpu=False, include_onnx=False
            )

            op = interpreter.get_op_for_symbol(self.func_name)
            trait = op.get_trait(CallableOpInterface)
            assert trait is not None

            args = tuple(
                interpreter.value_for_attribute(attr, attr_type)
                for attr, attr_type in zip(self.params, trait.get_argument_types(op))
            )

            interpreter.call_op(op, args)

            return snitch_cycle_estimator.cycles
    return (
        CallableOpInterface,
        SnitchCycleCostModel,
        register_implementations,
    )


@app.cell
def _(a_len, a_shape, b_len, b_shape, c_len, c_shape, f64):
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    a_attr = DenseIntOrFPElementsAttr.tensor_from_list(
        [i + 1 for i in range(a_len)], f64, a_shape
    )
    b_attr = DenseIntOrFPElementsAttr.tensor_from_list(
        [(i + 1) / 100 for i in range(b_len)], f64, b_shape
    )
    c_attr = DenseIntOrFPElementsAttr.tensor_from_list([0.0] * c_len, f64, c_shape)

    a_attr, b_attr, c_attr
    return DenseIntOrFPElementsAttr, a_attr, b_attr, c_attr


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
    return cycles, snitch_cost_model


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
    return memref_stream_cost, memref_stream_cost_model


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

    memref_stream_interleave.MemrefStreamInterleavePass().apply(
        interleaved_ctx, interleaved_module
    )

    interleaved_cost = memref_stream_cost_model.estimate_cost(interleaved_module, interleaved_ctx)

    mo.md(f"""
    We can use the existing interleave pass to choose the unroll-and-jam factor, with cost {interleaved_cost}:

    {xmo.module_html(interleaved_module)}
    """)
    return (
        interleaved_cost,
        interleaved_ctx,
        interleaved_module,
        memref_stream_interleave,
    )


@app.cell
def _(memref_stream_module, mo):
    from xdsl.dialects import memref_stream
    from xdsl.transforms.memref_stream_unroll_and_jam import unroll_and_jam_bound_indices_and_factors

    msg_op = next(child for child in memref_stream_module.walk() if isinstance(child, memref_stream.GenericOp))

    msg_factors = unroll_and_jam_bound_indices_and_factors(msg_op)

    mo.md(f"""
    We can also find the possible range of indices and factors to unroll-and-jam manually:

    {msg_factors}
    """)
    return (
        memref_stream,
        msg_factors,
        msg_op,
        unroll_and_jam_bound_indices_and_factors,
    )


@app.cell
def _(mo, msg_factors):
    from xdsl.transforms.memref_stream_unroll_and_jam import MemrefStreamUnrollAndJamPass

    uaj_passes = tuple(
        MemrefStreamUnrollAndJamPass(0, index, factor)
        for index, factor in msg_factors
    )

    _passes_str = "\n".join(str(p.pipeline_pass_spec()) for p in uaj_passes)

    mo.md(f"""
    We construct passes from these indices:

    ```
    {_passes_str}
    ```
    """)
    return MemrefStreamUnrollAndJamPass, uaj_passes


@app.cell(hide_code=True)
def _(MLContext, ModuleOp, ModulePass):
    def apply(p: ModulePass, ctx: MLContext, op: ModuleOp) -> ModuleOp:
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
    return (scores,)


@app.cell
def _():
    from random import Random
    return (Random,)


@app.cell
def _():
    from xdsl.interpreters.onnx import to_dtype
    return (to_dtype,)


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
    return (
        ContainerType,
        FloatAttr,
        ShapedType,
        TensorType,
        random_attr_of_type,
    )


@app.cell
def _(
    LensCostModel,
    MLContext,
    MemrefStreamUnrollAndJamPass,
    ModuleOp,
    ModulePass,
    Random,
    SnitchCycleCostModel,
    func,
    memref_stream,
    msg_factors,
    random_attr_of_type,
    riscv_passes,
    uaj_passes,
):
    class AutomaticUnrollAndJamPass(ModulePass):

        name = "automatic-unroll-and-jam"

        def apply(self, ctx: MLContext, op: ModuleOp) -> None:
            msg_ops = tuple(child for child in op.walk() if isinstance(child, memref_stream.GenericOp))

            if not msg_ops:
                return

            assert len(msg_ops) == 1

            msg_op = msg_ops[0]

            passes = tuple(
                MemrefStreamUnrollAndJamPass(0, index, factor)
                for index, factor in msg_factors
            )

            if not passes:
                return

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
                    for p in uaj_passes
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
    return automated_cost, automated_ctx, automated_module


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

        memref_stream_interleave.MemrefStreamInterleavePass().apply(
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
