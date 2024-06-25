from dataclasses import dataclass, field

from xdsl.dialects import builtin, csl, func
from xdsl.ir import Block, MLContext, Operation, Region
from xdsl.parser import Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


def _get_single_region(op: Operation) -> Region:
    if len(op.regions) != 1:
        raise RuntimeError(f"Cannot convert {op.name} with {len(op.regions)} regions")
    return op.regions[0]


def _detach_single_block(op: Operation) -> Block:
    region = _get_single_region(op)
    return region.detach_block(region.block)


@dataclass
class FuncToCsl(RewritePattern):
    """
    Convert all func.func to csl.func and all func.return to csl.return
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        block = _detach_single_block(op)
        match block.last_op:
            case func.Return(arguments=[]) as ret:
                rewriter.replace_op(ret, csl.ReturnOp())
            case func.Return(arguments=[arg]) as ret:
                rewriter.replace_op(ret, csl.ReturnOp(arg))
            case func.Return():
                raise RuntimeError(
                    f"Cannot convert {func.Return.name} with multiple results"
                )
            case other:
                raise RuntimeError(
                    f"Expected last op of {func.FuncOp.name} to be {func.Return.name}, got {other}"
                )
        new_func = csl.FuncOp(
            op.sym_name.data,
            op.function_type,
            Region(block),
            arg_attrs=op.arg_attrs,
            res_attrs=op.res_attrs,
        )
        rewriter.replace_matched_op(new_func)


@dataclass
class WrapInProgramModule(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter, /):
        csl_module = csl.CslModuleOp(
            properties={"kind": csl.ModuleKindAttr(csl.ModuleKind.PROGRAM)},
            attributes={"sym_name": builtin.StringAttr("program")},
            regions=[Region(Block())],
        )

        rewriter.inline_block(
            _get_single_region(op).block,
            InsertPoint.at_start(csl_module.body.block),
        )
        rewriter.inline_region_at_end(
            Region(Block([csl_module])),
            _get_single_region(op),
        )


@dataclass
class AddLayout(RewritePattern):
    pe_program: str
    pe_x_count: int
    pe_y_count: int
    launch_color: int
    ctx: MLContext

    def _build_layout_module(self):
        return Parser(
            self.ctx,
            f"""
"csl.module"() <{{kind = #csl<module_kind layout>}}> ({{

  %LAUNCH = "csl.get_color"() <{{id = {self.launch_color} : i5}}> : () -> !csl.color

  %memcpy_init_params = "csl.const_struct"(%LAUNCH) <{{
      items = {{ width = {self.pe_x_count} : i32, height = {self.pe_y_count} : i32}},
      ssa_fields = ["LAUNCH"]
    }}> : (!csl.color) -> !csl.comptime_struct

  %memcpy = "csl.import_module"(%memcpy_init_params) <{{module = "<memcpy/get_params>"}}> : (!csl.comptime_struct) -> !csl.imported_module

  csl.layout {{

    %x_dim_idx = arith.constant {self.pe_x_count} : index
    %y_dim_idx = arith.constant {self.pe_x_count} : index
    %x_dim = arith.index_cast %x_dim_idx : index to i32
    %y_dim = arith.index_cast %y_dim_idx : index to i32

    "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %x_coord_idx = %c0 to %x_dim_idx step %c1 {{
        scf.for %y_coord_idx = %c0 to %y_dim_idx step %c1 {{
            %x_coord = arith.index_cast %x_coord_idx : index to i32
            %y_coord = arith.index_cast %y_coord_idx : index to i32
            %memcpy_params = "csl.member_call"(%memcpy, %x_coord) <{{field = "get_params"}}> : (!csl.imported_module, i32) -> !csl.comptime_struct
            %tile_code_params = "csl.const_struct"(%memcpy_params) <{{ssa_fields = ["memcpy_params"]}}> : (!csl.comptime_struct) -> !csl.comptime_struct
            "csl.set_tile_code"(%x_coord, %y_coord, %tile_code_params) <{{file = "{self.pe_program}"}}> : (i32, i32, !csl.comptime_struct) -> ()
        }}
    }}

  }}
}}) {{sym_name = "layout"}} :  () -> ()
""",
        ).parse_op()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl.CslModuleOp, rewriter: PatternRewriter, /):
        rewriter.insert_op(self._build_layout_module(), InsertPoint.after(op))


@dataclass(frozen=True)
class BuiltinToCsl(ModulePass):
    name = "builtin-to-csl"

    pe_program: str = field(default="pe_program.csl")
    pe_x_count: int = field(default=1)
    pe_y_count: int = field(default=1)
    launch_color: int = field(default=0)

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    WrapInProgramModule(),
                    FuncToCsl(),
                ]
            ),
            apply_recursively=False,
        )
        walker.rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddLayout(
                        pe_program=self.pe_program,
                        pe_x_count=self.pe_x_count,
                        pe_y_count=self.pe_y_count,
                        launch_color=self.launch_color,
                        ctx=ctx,
                    ),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
