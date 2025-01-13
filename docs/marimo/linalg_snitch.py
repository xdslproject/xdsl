import marimo

__generated_with = "0.10.9"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import xdsl.utils.marimo as xmo
    return (xmo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Compiling `linalg` to Snitch

        This notebook walks through compiling micro-kernels defined in `linalg` to RISC-V and RISC-V with extensions for [Snitch](https://pulp-platform.github.io/snitch/), a neural network accelerator.

        _Toggle app view with `⌘` + `.` or `ctrl` + `.`_
        """
    )
    return


@app.cell
def _():
    # Import all the necessary functionality from xDSL for this notebook
    # If you see an error about xdsl not being defined run this cell manually

    from xdsl.backend.riscv.lowering import (
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_memref_to_riscv,
        convert_scf_to_riscv_scf,
    )
    from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import (
        ConvertRiscvScfToRiscvCfPass,
    )
    from xdsl.backend.riscv.lowering.convert_snitch_stream_to_snitch import (
        ConvertSnitchStreamToSnitch,
    )
    from xdsl.builder import ImplicitBuilder
    from xdsl.context import MLContext
    from xdsl.dialects import arith, func, linalg
    from xdsl.dialects.builtin import AffineMap, ArrayAttr, AffineMapAttr, MemRefType, ModuleOp, f64
    from xdsl.dialects.riscv import riscv_code
    from xdsl.interpreters.utils.ptr import TypedPtr
    from xdsl.ir import Attribute, Block, Region, SSAValue
    from xdsl.passes import PipelinePass
    from xdsl.tools.command_line_tool import get_all_dialects
    from xdsl.traits import CallableOpInterface
    from xdsl.transforms import (
        arith_add_fastmath,
        convert_linalg_to_loops,
        convert_linalg_to_memref_stream,
        convert_memref_stream_to_loops,
        convert_memref_stream_to_snitch_stream,
        convert_riscv_scf_for_to_frep,
        dead_code_elimination,
        loop_hoist_memref,
        lower_affine,
        memref_streamify,
        reconcile_unrealized_casts,
    )
    from xdsl.transforms.canonicalize import CanonicalizePass
    from xdsl.transforms.lower_snitch import LowerSnitchPass
    from xdsl.transforms.mlir_opt import MLIROptPass
    from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation
    from xdsl.transforms.riscv_scf_loop_range_folding import (
        RiscvScfLoopRangeFoldingPass,
    )
    from xdsl.transforms.snitch_register_allocation import SnitchRegisterAllocation
    return (
        AffineMap,
        AffineMapAttr,
        ArrayAttr,
        Attribute,
        Block,
        CallableOpInterface,
        CanonicalizePass,
        ConvertRiscvScfToRiscvCfPass,
        ConvertSnitchStreamToSnitch,
        ImplicitBuilder,
        LowerSnitchPass,
        MLContext,
        MLIROptPass,
        MemRefType,
        ModuleOp,
        PipelinePass,
        RISCVRegisterAllocation,
        Region,
        RiscvScfLoopRangeFoldingPass,
        SSAValue,
        SnitchRegisterAllocation,
        TypedPtr,
        arith,
        arith_add_fastmath,
        convert_arith_to_riscv,
        convert_func_to_riscv_func,
        convert_linalg_to_loops,
        convert_linalg_to_memref_stream,
        convert_memref_stream_to_loops,
        convert_memref_stream_to_snitch_stream,
        convert_memref_to_riscv,
        convert_riscv_scf_for_to_frep,
        convert_scf_to_riscv_scf,
        dead_code_elimination,
        f64,
        func,
        get_all_dialects,
        linalg,
        loop_hoist_memref,
        lower_affine,
        memref_streamify,
        reconcile_unrealized_casts,
        riscv_code,
    )


@app.cell
def _(
    AffineMap,
    AffineMapAttr,
    Block,
    ImplicitBuilder,
    MemRefType,
    ModuleOp,
    Region,
    a_shape,
    arith,
    b_shape,
    c_shape,
    f64,
    func,
    linalg,
    mo,
    xmo,
):
    a_type = MemRefType(f64, a_shape)
    b_type = MemRefType(f64, b_shape)
    c_type = MemRefType(f64, c_shape)

    kernel_op = func.FuncOp("matmul", ((a_type, b_type, c_type), ()))
    with ImplicitBuilder(kernel_op.body) as (a, b, c):
        # Add name hints to make it easier to track how values are lowered
        a.name_hint = "A"
        b.name_hint = "B"
        c.name_hint = "C"
        body = Region(Block(arg_types = (f64, f64, f64)))
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
            )
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

    mo.md(f"""

    Here is matrix multiplication defined in the `linalg` dialect, with the iteration space decoupled from the computation:

    {xmo.module_html(linalg_module)}
    """)
    return (
        a,
        a_type,
        a_val,
        acc_new_val,
        acc_old_val,
        b,
        b_type,
        b_val,
        body,
        c,
        c_type,
        kernel_op,
        linalg_module,
        prod_val,
    )


@app.cell
def _(mo):
    min_val = 1
    max_val = 10
    m = mo.ui.slider(min_val, max_val, value=2, label="M")
    n = mo.ui.slider(min_val, max_val, value=2, label="N")
    k = mo.ui.slider(min_val, max_val, value=2, label="K")
    return k, m, max_val, min_val, n


@app.cell
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


@app.cell
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
def _(mo):
    mo.md("""### Compiling to RISC-V""")
    return


@app.cell
def _(MLContext, get_all_dialects):
    linalg_ctx = MLContext()

    for dialect_name, dialect_factory in get_all_dialects().items():
        linalg_ctx.register_dialect(dialect_name, dialect_factory)
    return dialect_factory, dialect_name, linalg_ctx


@app.cell
def _(mo):
    mo.md("""We can take this representation, and lower to RISC-V-specific dialects:""")
    return


@app.cell
def _(
    PipelinePass,
    convert_arith_to_riscv,
    convert_func_to_riscv_func,
    convert_linalg_to_loops,
    convert_memref_to_riscv,
    convert_scf_to_riscv_scf,
    linalg_ctx,
    linalg_module,
    reconcile_unrealized_casts,
    xmo,
):
    lower_to_riscv = PipelinePass(
        [
            convert_linalg_to_loops.ConvertLinalgToLoopsPass(),
            convert_func_to_riscv_func.ConvertFuncToRiscvFuncPass(),
            convert_memref_to_riscv.ConvertMemrefToRiscvPass(),
            convert_arith_to_riscv.ConvertArithToRiscvPass(),
            convert_scf_to_riscv_scf.ConvertScfToRiscvPass(),
            reconcile_unrealized_casts.ReconcileUnrealizedCastsPass(),
        ]
    )

    riscv_ctx, riscv_module, riscv_html = xmo.pipeline_html(
        linalg_ctx, linalg_module, tuple(("", p) for p in lower_to_riscv.passes)
    )

    riscv_html
    return lower_to_riscv, riscv_ctx, riscv_html, riscv_module


@app.cell
def _(mo):
    mo.md(
        """
        #### Register allocation

        xDSL provides a register allocator for our RISC-V representation, that works on functions with structured control flow:
        """
    )
    return


@app.cell
def _(
    CanonicalizePass,
    PipelinePass,
    RISCVRegisterAllocation,
    riscv_ctx,
    riscv_module,
    xmo,
):
    allocate_registers = PipelinePass(
        [
            RISCVRegisterAllocation(),
            CanonicalizePass(),
        ]
    )

    regalloc_ctx, regalloc_module, regalloc_html = xmo.pipeline_html(
        riscv_ctx, riscv_module, tuple(("", p) for p in allocate_registers.passes),
    )

    regalloc_html
    return allocate_registers, regalloc_ctx, regalloc_html, regalloc_module


@app.cell
def _(
    CanonicalizePass,
    ConvertRiscvScfToRiscvCfPass,
    PipelinePass,
    regalloc_ctx,
    regalloc_module,
    xmo,
):
    lower_to_asm = PipelinePass(
        [
            ConvertRiscvScfToRiscvCfPass(),
            CanonicalizePass(),
        ]
    )

    riscv_asm_ctx, riscv_asm_module, assembly_html = xmo.pipeline_html(
        regalloc_ctx, regalloc_module, (("", lower_to_asm),),
    )

    assembly_html
    return assembly_html, lower_to_asm, riscv_asm_ctx, riscv_asm_module


@app.cell
def _(mo):
    mo.md("""This representation of the program in xDSL corresponds ~1:1 to RISC-V assembly, and we can use a helper function to print that out.""")
    return


@app.cell
def _(mo, riscv_asm_module, riscv_code, xmo):
    riscv_asm = riscv_code(riscv_asm_module)

    mo.md(f"""\
    **RISC-V Assembly:**

    {xmo.asm_html(riscv_asm)}
    """
    )
    return (riscv_asm,)


@app.cell
def _(mo):
    mo.md(
        """
        ### Compiling to Snitch

        xDSL is also capable of targeting Snitch, and making use of its streaming registers and fixed-repetition loop. We use a different lowering flow from the linalg.generic representation to represent a high-level, structured, but Snitch-specific representation of the code:
        """
    )
    return


@app.cell
def _(
    PipelinePass,
    arith_add_fastmath,
    convert_linalg_to_memref_stream,
    convert_riscv_scf_for_to_frep,
    linalg_ctx,
    linalg_module,
    xmo,
):
    from xdsl.transforms.test_lower_linalg_to_snitch import LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES, OPTIMISE_MEMREF_STREAM_PASSES

    convert_linalg_to_snitch = PipelinePass(
        [
            convert_linalg_to_memref_stream.ConvertLinalgToMemrefStreamPass(),
            arith_add_fastmath.AddArithFastMathFlagsPass(),
            *OPTIMISE_MEMREF_STREAM_PASSES,
            *LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
            convert_riscv_scf_for_to_frep.ConvertRiscvScfForToFrepPass(),
        ]
    )

    snitch_stream_ctx, snitch_stream_module, snitch_stream_html = xmo.pipeline_html(
        linalg_ctx, linalg_module, tuple(("", p) for p in convert_linalg_to_snitch.passes),
    )

    snitch_stream_html
    return (
        LOWER_MEMREF_STREAM_TO_SNITCH_STREAM_PASSES,
        OPTIMISE_MEMREF_STREAM_PASSES,
        convert_linalg_to_snitch,
        snitch_stream_ctx,
        snitch_stream_html,
        snitch_stream_module,
    )


@app.cell
def _(mo):
    mo.md("""We can then lower this to assembly that includes assembly instructions from the Snitch-extended ISA:""")
    return


@app.cell
def _(snitch_stream_ctx, snitch_stream_module, xmo):
    from xdsl.transforms.test_lower_linalg_to_snitch import LOWER_SNITCH_STREAM_TO_ASM_PASSES

    snitch_asm_ctx, snitch_asm_module, snitch_asm_html = xmo.pipeline_html(
        snitch_stream_ctx, snitch_stream_module, tuple(("", p) for p in LOWER_SNITCH_STREAM_TO_ASM_PASSES)
    )

    snitch_asm_html
    return (
        LOWER_SNITCH_STREAM_TO_ASM_PASSES,
        snitch_asm_ctx,
        snitch_asm_html,
        snitch_asm_module,
    )


@app.cell
def _(k, m, mo, n):
    mo.md(
        f"""
        We can see how changing our input sizes affects the assembly produced:

        {m}{m.value}

        {n}{n.value}

        {k}{k.value}
        """
    )
    return


@app.cell
def _(mo, riscv_code, snitch_asm_module, xmo):
    snitch_asm = riscv_code(snitch_asm_module)

    mo.md(f"""\
    **Snitch Assembly:**

    {xmo.asm_html(snitch_asm)}
    """
    )
    return (snitch_asm,)


@app.cell
def _(mo):
    mo.md(
        """
        ### Interpreting the assembly using xDSL

        One of the useful features of xDSL is its interpreter. Here we've implemented all the necessary functions to interpret the code at a low level, to check that our compilation is correct. Here's the slider modifying the shape variable defined above, we can slide it to see the result of the code compiled with different input shapes, and interpreted at the RISC-V level.
        """
    )
    return


@app.cell
def _():
    from xdsl.interpreter import Interpreter, OpCounter
    from xdsl.interpreters import register_implementations
    from xdsl.interpreters.shaped_array import ShapedArray
    return Interpreter, OpCounter, ShapedArray, register_implementations


@app.cell
def _():
    from dataclasses import dataclass, field
    return dataclass, field


@app.cell
def _(Interpreter, Operation, PythonValues, dataclass, field):
    from xdsl.dialects import riscv, riscv_cf, riscv_func

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
        SKIPPED_OPS,
        SnitchCycleEstimator,
        riscv,
        riscv_cf,
        riscv_func,
    )


@app.cell
def _():
    import abc
    return (abc,)


@app.cell
def _(ModuleOp, abc):
    class CostModel(abc.ABC):

        @abc.abstractmethod
        def estimate_cost(self, module: ModuleOp) -> int | None:
            ...
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
    CallableOpInterface,
    CostModel,
    Interpreter,
    MLContext,
    ModuleOp,
    SnitchCycleEstimator,
    register_implementations,
):
    class SnitchCycleCostModel(CostModel):

        func_name: str
        params: tuple[Attribute, ...]

        def __init__(self, func_name: str, params: tuple[Attribute, ...]):
            self.func_name = func_name
            self.params = params

        def estimate_cost(self, module: ModuleOp, ctx: MLContext) -> int | None:
            snitch_cycle_estimator = SnitchCycleEstimator()
            interpreter = Interpreter(module, listeners=(snitch_cycle_estimator,))

            register_implementations(interpreter, ctx, include_wgpu=False, include_onnx=False)

            op = interpreter.get_op_for_symbol(self.func_name)
            trait = op.get_trait(CallableOpInterface)
            assert trait is not None

            args = tuple(
                interpreter.value_for_attribute(attr, attr_type)
                for attr, attr_type in zip(
                    self.params, trait.get_argument_types(op)
                )
            )

            interpreter.call_op(op, args)

            return snitch_cycle_estimator.cycles
    return (SnitchCycleCostModel,)


@app.cell
def _():
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    return (DenseIntOrFPElementsAttr,)


@app.cell
def _(a_shape, b_shape, c_shape):
    from math import prod

    a_len = prod(a_shape)
    b_len = prod(b_shape)
    c_len = prod(c_shape)
    return a_len, b_len, c_len, prod


@app.cell
def _(
    DenseIntOrFPElementsAttr,
    a_len,
    a_shape,
    b_len,
    b_shape,
    c_len,
    c_shape,
    f64,
):
    a_attr = DenseIntOrFPElementsAttr.tensor_from_list([i + 1 for i in range(a_len)], f64, a_shape)
    b_attr = DenseIntOrFPElementsAttr.tensor_from_list([(i + 1) / 100 for i in range(b_len)], f64, b_shape)
    c_attr = DenseIntOrFPElementsAttr.tensor_from_list([0.0] * c_len, f64, c_shape)
    a_attr, b_attr, c_attr
    return a_attr, b_attr, c_attr


@app.cell
def _(
    SnitchCycleCostModel,
    a_attr,
    b_attr,
    c_attr,
    riscv_asm_ctx,
    riscv_asm_module,
):
    cost_model = SnitchCycleCostModel("matmul", (a_attr, b_attr, c_attr))

    cycles = cost_model.estimate_cost(riscv_asm_module, riscv_asm_ctx)

    cycles
    return cost_model, cycles


@app.cell
def _(
    Interpreter,
    OpCounter,
    ShapedArray,
    SnitchCycleEstimator,
    TypedPtr,
    a_attr,
    a_shape,
    b_attr,
    b_shape,
    c_attr,
    c_shape,
    f64,
    mo,
    register_implementations,
    riscv,
    riscv_asm_module,
    riscv_ctx,
):
    riscv_op_counter = OpCounter()
    riscv_cycle_estimator = SnitchCycleEstimator()
    riscv_interpreter = Interpreter(riscv_asm_module, listeners=(riscv_cycle_estimator, riscv_op_counter))

    register_implementations(riscv_interpreter, riscv_ctx, include_wgpu=False, include_onnx=False)

    riscv_a = riscv_interpreter.value_for_attribute(a_attr, riscv.Registers.A0)
    riscv_b = riscv_interpreter.value_for_attribute(b_attr, riscv.Registers.A1)
    riscv_c = riscv_interpreter.value_for_attribute(c_attr, riscv.Registers.A2)

    riscv_a_shaped = ShapedArray(TypedPtr(riscv_a, xtype=f64), a_shape)
    riscv_b_shaped = ShapedArray(TypedPtr(riscv_b, xtype=f64), b_shape)
    riscv_c_shaped = ShapedArray(TypedPtr(riscv_c, xtype=f64), c_shape)

    riscv_interpreter.call_op("matmul", (riscv_a, riscv_b, riscv_c))

    mo.md(f"""
    **RISC-V Results:**

    A: {riscv_a_shaped}

    B: {riscv_b_shaped}

    C: {riscv_c_shaped}

    Cycles: {riscv_cycle_estimator.cycles}
    """)
    return (
        riscv_a,
        riscv_a_shaped,
        riscv_b,
        riscv_b_shaped,
        riscv_c,
        riscv_c_shaped,
        riscv_cycle_estimator,
        riscv_interpreter,
        riscv_op_counter,
    )


@app.cell
def _(
    Interpreter,
    OpCounter,
    ShapedArray,
    TypedPtr,
    a_attr,
    a_shape,
    b_attr,
    b_shape,
    c_attr,
    c_len,
    c_shape,
    f64,
    mo,
    register_implementations,
    riscv,
    riscv_interpreter,
    snitch_stream_ctx,
    snitch_stream_module,
):
    snitch_op_counter = OpCounter()
    snitch_interpreter = Interpreter(
        snitch_stream_module, listeners=(snitch_op_counter,)
    )

    snitch_c_shaped = ShapedArray(TypedPtr.new_float64([0.0] * c_len), c_shape)

    snitch_a = riscv_interpreter.value_for_attribute(a_attr, riscv.Registers.A0)
    snitch_b = riscv_interpreter.value_for_attribute(b_attr, riscv.Registers.A1)
    snitch_c = riscv_interpreter.value_for_attribute(c_attr, riscv.Registers.A2)

    snitch_a_shaped = ShapedArray(TypedPtr(snitch_a, xtype=f64), a_shape)
    snitch_b_shaped = ShapedArray(TypedPtr(snitch_b, xtype=f64), b_shape)
    snitch_c_shaped = ShapedArray(TypedPtr(snitch_c, xtype=f64), c_shape)

    register_implementations(snitch_interpreter, snitch_stream_ctx, include_wgpu=False, include_onnx=False)

    snitch_interpreter.call_op("matmul", (snitch_a, snitch_b, snitch_c))

    mo.md(f"""

    **Snitch Results:**

    A: {snitch_a_shaped}

    B: {snitch_b_shaped}

    C: {snitch_c_shaped}
    """)
    return (
        snitch_a,
        snitch_a_shaped,
        snitch_b,
        snitch_b_shaped,
        snitch_c,
        snitch_c_shaped,
        snitch_interpreter,
        snitch_op_counter,
    )


@app.cell
def _(k, m, mo, n, riscv_op_counter, snitch_op_counter):
    rv_dict = dict(riscv_op_counter.ops)
    sn_dict = dict(snitch_op_counter.ops)

    all_keys = sorted(set(rv_dict) | set(sn_dict))
    max_len_key = max(len(k) for k in all_keys)
    max_len_value = max(len(str(val)) for val in (*rv_dict.values(), *sn_dict.values()))

    def format_row(key: str, *values: str):
        paddings = tuple(" " * (max_len_value - len(val)) for val in values)
        vals = "".join(f"\t{padding}{val}" for padding, val in zip(paddings, values))
        return f"{key}{' ' * (max_len_key - len(key))}{vals}\n"

    rows = " " * max_len_key + "\trv\tsn\tdiff\n"

    ZERO_VAL = "."

    for key in all_keys:
        rv_val = rv_dict.get(key, 0)
        sn_val = sn_dict.get(key, 0)
        diff_val = sn_val - rv_val

        rv_str = str(rv_val) if rv_val else ZERO_VAL
        sn_str = str(sn_val) if sn_val else ZERO_VAL
        diff_str = (
            (f"+{diff_val}" if diff_val > 0 else f"{diff_val}") if diff_val else "="
        )

        rows += key
        rows += " " * (max_len_key - len(key))
        rows += "\t"
        rows += " " * (max_len_value - len(rv_str))
        rows += rv_str
        rows += "\t"
        rows += " " * (max_len_value - len(sn_str))
        rows += sn_str
        rows += "\t"
        rows += " " * (max_len_value - len(diff_str))
        rows += diff_str
        rows += "\n"

    rv_sum = sum(rv_dict.values())
    sn_sum = sum(sn_dict.values())
    total_diff = sn_sum - rv_sum

    rows += format_row("total", str(rv_sum), str(sn_sum), str(total_diff))

    mo.md(
        f"""
    The interpreter kept track of the number of times an operation was executed, which we can use as a proxy for performance.

    For example, we can see that one version has many more instructions executed overall ({rv_sum} vs {sn_sum}), and that one version uses explicit load and store instructions, while the other uses the streaming equivalents:

    {m}{m.value}

    {n}{n.value}

    {k}{k.value}

    ```
    {rows}
    ```
    """
    )
    return (
        ZERO_VAL,
        all_keys,
        diff_str,
        diff_val,
        format_row,
        key,
        max_len_key,
        max_len_value,
        rows,
        rv_dict,
        rv_str,
        rv_sum,
        rv_val,
        sn_dict,
        sn_str,
        sn_sum,
        sn_val,
        total_diff,
    )


if __name__ == "__main__":
    app.run()
