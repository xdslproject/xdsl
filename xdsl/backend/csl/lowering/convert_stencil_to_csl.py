from attr import dataclass

from xdsl.dialects import func
from xdsl.dialects.arith import Cmpi, Constant, MinUI, Muli, OrI
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    Signedness,
    StringAttr,
    TensorType,
)
from xdsl.dialects.csl import (
    ComptimeStructType,
    ConcatStructOp,
    ConstStructOp,
    CslModuleOp,
    GetColorOp,
    ImportModuleConstOp,
    LayoutOp,
    MemberCallOp,
    ModuleKind,
    ModuleKindAttr,
    ParamOp,
    SetRectangleOp,
    SetTileCodeOp,
    SignednessCastOp,
)
from xdsl.dialects.scf import For, Yield
from xdsl.dialects.stencil import ApplyOp
from xdsl.ir import Attribute, Block, MLContext, Operation, Region
from xdsl.passes import ModulePass
from xdsl.utils.hints import isa


class TranslationContext:
    grid_dim: tuple[int, ...] = (0, 0, 0)
    pattern: int
    program_module: CslModuleOp
    layout_module: CslModuleOp
    program_sym_name: StringAttr

    program_params: dict[str, ParamOp] = {}
    layout_params: dict[str, ParamOp] = {}
    program_imports: dict[str, ImportModuleConstOp] = {}
    layout_imports: dict[str, ImportModuleConstOp] = {}
    program_vars: dict[str, Operation] = {}
    layout_vars: dict[str, Operation] = {}

    def add_param_to_program(self, param: ParamOp):
        self.program_module.body.block.add_op(param)
        self.program_params[param.param_name.data] = param

    def add_param_to_layout(self, param: ParamOp):
        self.layout_module.body.block.add_op(param)
        self.layout_params[param.param_name.data] = param

    def add_import_module_to_program(self, lib: ImportModuleConstOp):
        self.program_module.body.block.add_op(lib)
        self.program_imports[lib.module.data] = lib

    def add_import_module_to_layout(self, lib: ImportModuleConstOp):
        self.layout_module.body.block.add_op(lib)
        self.layout_imports[lib.module.data] = lib

    def add_params_to_program(self, *params: ParamOp):
        for param in params:
            self.add_param_to_program(param)

    def add_params_to_layout(self, *params: ParamOp):
        for param in params:
            self.add_param_to_layout(param)

    def add_import_modules_to_program(self, *lib: ImportModuleConstOp):
        for l in lib:
            self.add_import_module_to_program(l)

    def add_import_modules_to_layout(self, *lib: ImportModuleConstOp):
        for l in lib:
            self.add_import_module_to_layout(l)

    def add_var_to_program(self, name: str, op: Operation):
        self.program_module.body.block.add_op(op)
        self.program_vars[name] = op

    def add_var_to_layout(self, name: str, op: Operation):
        self.layout_module.body.block.add_op(op)
        self.layout_vars[name] = op


def setup_program_module(ctx: TranslationContext) -> None:
    ctx.add_params_to_program(
        ParamOp.get("stencil_comms_params", ComptimeStructType()),
        ParamOp.get("memcpy_params", ComptimeStructType()),
        ParamOp.get("z_dim", IntegerType(16, Signedness.SIGNED)),
        ParamOp.get("is_border_region_pe", IntegerType(1)),
    )

    ctx.add_import_modules_to_program(
        ImportModuleConstOp.get("<memcpy/memcpy>", ctx.program_params["memcpy_params"]),
        ImportModuleConstOp.get("<time>"),
        util := ImportModuleConstOp.get("util.csl"),
    )

    ctx.add_var_to_program(
        "computeChunks",
        num_chunks := MemberCallOp.get(
            "computeChunks",
            IntegerType(16, Signedness.UNSIGNED),
            util,
            ctx.program_params["z_dim"],
        ),
    )
    ctx.add_var_to_program(
        "computeChunkSize",
        chunk_size := MemberCallOp.get(
            "computeChunkSize",
            IntegerType(16, Signedness.UNSIGNED),
            util,
            ctx.program_params["z_dim"],
            num_chunks,
        ),
    )
    ctx.add_var_to_program("padded_z_dim", Muli(num_chunks, chunk_size))
    ctx.add_var_to_program(
        "pattern",
        pattern_op := Constant(
            IntegerAttr(ctx.pattern, IntegerType(16, Signedness.UNSIGNED))
        ),
    )
    ctx.add_var_to_program(
        "stencil_comms_params_ext",
        stencil_comms_params_ext := ConstStructOp.get(
            ("pattern", pattern_op), ("chunkSize", chunk_size)
        ),
    )
    ctx.add_var_to_program(
        "stencil_comms_params_combined",
        stencil_comms_params_combined := ConcatStructOp.get(
            stencil_comms_params_ext, ctx.program_params["stencil_comms_params"]
        ),
    )

    ctx.add_import_module_to_program(
        ImportModuleConstOp.get("stencil_comms.csl", stencil_comms_params_combined),
    )


def setup_layout_module(ctx: TranslationContext) -> None:
    ctx.add_var_to_layout(
        "LAUNCH_ID",
        launch_id := Constant(IntegerAttr(0, IntegerType(16, Signedness.SIGNED))),
    )

    ctx.add_var_to_layout("LAUNCH", launch := GetColorOp.get(launch_id))
    ctx.add_var_to_layout(
        "width",
        width := Constant(
            IntegerAttr(ctx.grid_dim[0], IntegerType(16, Signedness.UNSIGNED))
        ),
    )
    ctx.add_var_to_layout(
        "height",
        height := Constant(
            IntegerAttr(ctx.grid_dim[1], IntegerType(16, Signedness.UNSIGNED))
        ),
    )
    ctx.add_var_to_layout(
        "z_dim",
        Constant(IntegerAttr(ctx.grid_dim[2], IntegerType(16, Signedness.UNSIGNED))),
    )
    ctx.add_var_to_layout(
        "pattern",
        pattern_op := Constant(
            IntegerAttr(ctx.pattern, IntegerType(16, Signedness.UNSIGNED))
        ),
    )
    # ("iteration_task_id", iterationTaskId: local_task_id = @get_local_task_id(3); #todo)

    ctx.add_var_to_layout(
        "memcpy_call_params",
        memcpy_call_params := ConstStructOp.get(
            ("width", width), ("height", height), ("LAUNCH", launch)
        ),
    )
    ctx.add_var_to_layout(
        "routes_params",
        routes_params := ConstStructOp.get(
            ("pattern", pattern_op), ("peWidth", width), ("peHeight", height)
        ),
    )

    ctx.add_import_modules_to_layout(
        memcpy := ImportModuleConstOp.get("<memcpy/get_params>", memcpy_call_params),
        routes := ImportModuleConstOp.get("routes.csl", routes_params),
    )

    ctx.add_var_to_layout("layout", layout_op := LayoutOp(Region(Block())))
    layout = layout_op.body.block
    layout.add_op(SetRectangleOp(operands=[width, height]))
    layout.add_op(zero := Constant(IntegerAttr(0, IntegerType(16))))
    layout.add_op(one := Constant(IntegerAttr(1, IntegerType(16))))
    layout.add_op(one_u := SignednessCastOp.get_u(one))
    layout.add_op(width_sl := SignednessCastOp.get(width))
    layout.add_op(height_sl := SignednessCastOp.get(height))
    outer_loop_body = Block(arg_types=[IntegerType(16)])
    inner_loop_body = Block(arg_types=[IntegerType(16)])
    layout.add_op(
        For(lb=zero, ub=width_sl, step=one, iter_args=[], body=outer_loop_body)
    )
    outer_loop_body.add_op(x_id := SignednessCastOp.get_u(outer_loop_body.args[0]))
    outer_loop_body.add_op(
        For(lb=zero, ub=height_sl, step=one, iter_args=[], body=inner_loop_body)
    )

    inner_loop_body.add_ops(
        [
            y_id := SignednessCastOp.get_u(inner_loop_body.args[0]),
            pattern_minus_one := MinUI(pattern_op, one_u),
            width_minus_xid := MinUI(width, x_id),
            height_minus_yid := MinUI(height, y_id),
            first := Cmpi(x_id, pattern_minus_one, "ult"),
            second := Cmpi(y_id, pattern_minus_one, "ult"),
            third := Cmpi(width_minus_xid, pattern_op, "ult"),
            fourth := Cmpi(height_minus_yid, pattern_op, "ult"),
            or_one := OrI(first, second),
            or_two := OrI(or_one, third),
            is_border_pe := OrI(or_two, fourth),
            params_task := ConstStructOp.get(("isBorderRegionPE", is_border_pe)),
            memcpy_params := MemberCallOp.get(
                "get_params", ComptimeStructType(), memcpy, x_id
            ),
            route_params := MemberCallOp.get(
                "computeAllRoutes",
                ComptimeStructType(),
                routes,
                x_id,
                y_id,
                width,
                height,
                pattern_op,
            ),
            set_tile_params := ConstStructOp.get(
                ("memcpyParams", memcpy_params), ("stencilCommsParams", route_params)
            ),
            set_tile_params_ext := ConcatStructOp.get(params_task, set_tile_params),
            SetTileCodeOp.get(
                ctx.program_module.sym_name,
                x_id,
                y_id,
                set_tile_params_ext,
            ),
            Yield(),
        ]
    )
    outer_loop_body.add_op(Yield())


@dataclass(frozen=True)
class ConvertStencilToCsl(ModulePass):
    name = "convert-stencil-to-csl"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        t_ctx = TranslationContext()
        t_ctx.program_module = CslModuleOp(
            regions=[Region(Block())],
            properties={"kind": ModuleKindAttr(ModuleKind.PROGRAM)},
            attributes={"sym_name": StringAttr("pe.csl")},
        )
        t_ctx.layout_module = CslModuleOp(
            regions=[Region(Block())],
            properties={"kind": ModuleKindAttr(ModuleKind.LAYOUT)},
            attributes={"sym_name": StringAttr("layout.csl")},
        )
        assert len(op.body.block.ops) == 1
        assert isinstance(stencil_func := op.body.block.first_op, func.FuncOp)
        t_ctx.program_sym_name = stencil_func.sym_name
        t_ctx.pattern = 0
        for o in stencil_func.body.block.ops:
            if isinstance(o, ApplyOp):
                for access in o.get_accesses():
                    for l, r in access.halos():
                        t_ctx.pattern = max(t_ctx.pattern, abs(l), abs(r))

        # determine the grid dimensions by looking at the function args
        for arg in stencil_func.args:
            if isa(arg.type, MemRefType[Attribute]) and isa(
                arg.type.element_type, TensorType[Attribute]
            ):
                shape = arg.type.get_shape()
                t_ctx.grid_dim = tuple(
                    max(a, b)
                    for a, b in zip(
                        t_ctx.grid_dim,
                        (shape[0], shape[1], arg.type.element_type.get_shape()[0]),
                    )
                )

        op.body.block.add_op(t_ctx.program_module)
        op.body.block.add_op(t_ctx.layout_module)

        setup_program_module(t_ctx)
        setup_layout_module(t_ctx)
