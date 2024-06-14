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
from xdsl.utils.str_enum import StrEnum


class Named(StrEnum):
    """
    Enum type to store names used across the program, for instance
    names of parameters passed from the layout to the program module.
    """

    pattern = "pattern"
    stencil_comms_params = "stencil_comms_params"
    memcpy_params = "memcpy_params"
    z_dim = "z_dim"
    is_border_region_pe = "is_border_region_pe"
    layout_block = "layout_block"


@dataclass(frozen=True)
class TranslationContext:
    """
    Helper class for translating to CSL.
    Keeps track of stencil parameters such as stencil size, grid dimension.
    Provides several methods to register any top-level params, imports, and consts/vars
    both for program and layout modules. Params and imports are intrinsically names, consts/vars
    should be given a name (which is only used for this translation)
    """

    grid_dim: tuple[int, int, int]
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


def generate_program_module(ctx: TranslationContext) -> None:
    """
    Generates the basic structure of a program module needed for stencil computations.
    This function should *not* translate the actual stencil computation, but set up everything
    needed for these, including imports, params, and top-level constants.
    Params are generated to match `@set_tile_code` invocation of the
    layout module. Also imports memcpy, utils, and stencil_comms modules. Vars/consts
    are set up as required to perform these imports.
    """

    # program params to be supplied by layout module
    ctx.add_params_to_program(
        s_params := ParamOp(Named.stencil_comms_params, ComptimeStructType()),
        m_params := ParamOp(Named.memcpy_params, ComptimeStructType()),
        ParamOp(Named.is_border_region_pe, IntegerType(1)),
    )

    # size of the z dimension
    ctx.add_var_to_program(
        Named.z_dim,
        z_dim := Constant(
            IntegerAttr(ctx.grid_dim[2], IntegerType(16, Signedness.UNSIGNED))
        ),
    )

    # module imports
    ctx.add_import_modules_to_program(
        ImportModuleConstOp("<memcpy/memcpy>", m_params),
        ImportModuleConstOp("<time>"),
        util := ImportModuleConstOp("util.csl"),
    )

    # generates: num_chunks = util.computeChunks(z_dim)
    ctx.add_var_to_program(
        "num_chunks",
        num_chunks := MemberCallOp(
            "computeChunks",
            IntegerType(16, Signedness.UNSIGNED),
            util,
            z_dim,
        ),
    )

    # generates: chunk_size = util.computeChunks(z_dim, num_chunks)
    ctx.add_var_to_program(
        "chunk_size",
        chunk_size := MemberCallOp(
            "computeChunkSize",
            IntegerType(16, Signedness.UNSIGNED),
            util,
            z_dim,
            num_chunks,
        ),
    )

    # when sending / receiving buffers with stencil_comms, use padded_z_dim rather than z_dim as buffer size
    ctx.add_var_to_program("padded_z_dim", Muli(num_chunks, chunk_size))

    # stencil pattern, i.e. neighbours in any direction + 1
    ctx.add_var_to_program(
        Named.pattern,
        pattern_op := Constant(
            IntegerAttr(ctx.pattern, IntegerType(16, Signedness.UNSIGNED))
        ),
    )

    # setting up structs to import stencil_comms.csl
    ctx.add_var_to_program(
        "stencil_comms_params_ext",
        stencil_comms_params_ext := ConstStructOp(
            (Named.pattern, pattern_op), ("chunkSize", chunk_size)
        ),
    )
    ctx.add_var_to_program(
        "stencil_comms_params_combined",
        stencil_comms_params_combined := ConcatStructOp(
            stencil_comms_params_ext, s_params
        ),
    )

    # importing stencil_comms.csl
    ctx.add_import_module_to_program(
        ImportModuleConstOp("stencil_comms.csl", stencil_comms_params_combined),
    )


def generate_layout_module(ctx: TranslationContext) -> None:
    """
    Generates a layout module for the stencil parameters given in the translation context.
    For each PE, the layout module determines program parameters (incl.memcpy and route params)
    before invoking `@set_tile_code` to assign a csl program to each PE.
    This is done in a two-dimensional loop over the compute grid size. Otherwise, the structure
    of the layout module is static and varies only in terms of
    exported symbol names (not implemented yet).
    """
    ctx.add_var_to_layout(
        "LAUNCH_ID",
        launch_id := Constant(IntegerAttr(0, IntegerType(16, Signedness.SIGNED))),
    )

    ctx.add_var_to_layout("LAUNCH", launch := GetColorOp(launch_id))

    # width is the number of PEs in the x dimension
    ctx.add_var_to_layout(
        "width",
        width := Constant(
            IntegerAttr(ctx.grid_dim[0], IntegerType(16, Signedness.UNSIGNED))
        ),
    )

    # height is the number of PEs in the x dimension
    ctx.add_var_to_layout(
        "height",
        height := Constant(
            IntegerAttr(ctx.grid_dim[1], IntegerType(16, Signedness.UNSIGNED))
        ),
    )

    # stencil pattern, i.e. neighbours in any direction + 1
    ctx.add_var_to_layout(
        Named.pattern,
        pattern_op := Constant(
            IntegerAttr(ctx.pattern, IntegerType(16, Signedness.UNSIGNED))
        ),
    )
    # ("iteration_task_id", iterationTaskId: local_task_id = @get_local_task_id(3); #todo)

    # setting up struct to import memcpy module
    ctx.add_var_to_layout(
        "memcpy_call_params",
        memcpy_call_params := ConstStructOp(
            ("width", width), ("height", height), ("LAUNCH", launch)
        ),
    )

    # setting up struct to import routes module
    ctx.add_var_to_layout(
        "routes_params",
        routes_params := ConstStructOp(
            (Named.pattern, pattern_op), ("peWidth", width), ("peHeight", height)
        ),
    )

    # import memcpy and routes module
    ctx.add_import_modules_to_layout(
        memcpy := ImportModuleConstOp("<memcpy/get_params>", memcpy_call_params),
        routes := ImportModuleConstOp("routes.csl", routes_params),
    )

    # add layout block and set grid dimensions
    ctx.add_var_to_layout("layout_block", layout_op := LayoutOp(Region(Block())))
    layout = layout_op.body.block
    layout.add_op(SetRectangleOp(operands=[width, height]))

    # setting up constants and signedness casts
    layout.add_op(zero := Constant(IntegerAttr(0, IntegerType(16))))
    layout.add_op(one := Constant(IntegerAttr(1, IntegerType(16))))
    layout.add_op(one_u := SignednessCastOp(one))
    layout.add_op(width_sl := SignednessCastOp(width))
    layout.add_op(height_sl := SignednessCastOp(height))

    # setting up two-dimensional loop nest over physical x-y PEs
    outer_loop_body = Block(arg_types=[IntegerType(16)])
    inner_loop_body = Block(arg_types=[IntegerType(16)])

    # adding outer loop to layout block
    layout.add_op(
        For(lb=zero, ub=width_sl, step=one, iter_args=[], body=outer_loop_body)
    )

    # signedness cast for outer loop variable
    # outer loop is x_id
    outer_loop_body.add_op(x_id := SignednessCastOp(outer_loop_body.args[0]))

    # adding inner loop inside outer loop
    outer_loop_body.add_op(
        For(lb=zero, ub=height_sl, step=one, iter_args=[], body=inner_loop_body)
    )

    # preparing calls to route module, memcpy module, and `@set_tile_code`
    inner_loop_body.add_ops(
        [
            # inner loop is y_id
            y_id := SignednessCastOp(inner_loop_body.args[0]),
            # compute boolean expression is_border_region_pe
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
            # generates: memcpy.get_params(xId)
            memcpy_params := MemberCallOp(
                "get_params", ComptimeStructType(), memcpy, x_id
            ),
            # generates: routes.computeAllRoutes(x_id, y_id, width, height, pattern)
            route_params := MemberCallOp(
                "computeAllRoutes",
                ComptimeStructType(),
                routes,
                x_id,
                y_id,
                width,
                height,
                pattern_op,
            ),
            # setting up param structs for `@set_tile_code`
            set_tile_params := ConstStructOp(
                (Named.memcpy_params, memcpy_params),
                (Named.stencil_comms_params, route_params),
            ),
            params_task := ConstStructOp((Named.is_border_region_pe, is_border_pe)),
            set_tile_params_ext := ConcatStructOp(params_task, set_tile_params),
            # generates: @set_tile_code(xId, yId, "pe.csl", .. param structs ..)
            SetTileCodeOp(
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
        # initialise both CSL modules
        program_module = CslModuleOp(
            regions=[Region(Block())],
            properties={"kind": ModuleKindAttr(ModuleKind.PROGRAM)},
            attributes={"sym_name": StringAttr("pe.csl")},
        )
        layout_module = CslModuleOp(
            regions=[Region(Block())],
            properties={"kind": ModuleKindAttr(ModuleKind.LAYOUT)},
            attributes={"sym_name": StringAttr("layout.csl")},
        )

        assert len(op.body.block.ops) == 1
        assert isinstance(stencil_func := op.body.block.first_op, func.FuncOp)

        # iterate over stencil.apply ops to find the maximum stencil width ('arm length')
        neighbours = 0
        for o in stencil_func.body.block.ops:
            if isinstance(o, ApplyOp):
                for access in o.get_accesses():
                    for l, r in access.halos():
                        neighbours = max(neighbours, abs(l), abs(r))

        # determine the grid dimensions by looking at the function args
        grid_dim: tuple[int, int, int] = (0, 0, 0)
        for arg in stencil_func.args:
            if isa(arg.type, MemRefType[Attribute]) and isa(
                arg.type.element_type, TensorType[Attribute]
            ):
                shape = arg.type.get_shape()
                new_maxs = tuple(
                    max(a, b)
                    for a, b in zip(
                        grid_dim,
                        (shape[0], shape[1], arg.type.element_type.get_shape()[0]),
                    )
                )
                assert len(new_maxs) == 3
                grid_dim = new_maxs

        # the CSL `pattern` variable corresponds to the neighbours in each direction plus 1
        t_ctx = TranslationContext(
            grid_dim=grid_dim,
            pattern=neighbours + 1,
            program_sym_name=stencil_func.sym_name,
            program_module=program_module,
            layout_module=layout_module,
        )

        # add modules to the builtin module
        op.body.block.add_op(t_ctx.program_module)
        op.body.block.add_op(t_ctx.layout_module)

        # set up basic module structure
        generate_program_module(t_ctx)
        generate_layout_module(t_ctx)

        # todo translate stencil computation

        # cleanup stencil func after translation is done
        op.body.block.erase_op(stencil_func)
