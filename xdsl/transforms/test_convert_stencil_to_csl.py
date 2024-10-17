from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass, PipelinePass
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)
from xdsl.transforms.convert_stencil_to_csl_stencil import (
    ConvertStencilToCslStencilPass,
)
from xdsl.transforms.csl_stencil_bufferize import CslStencilBufferize
from xdsl.transforms.csl_stencil_handle_async_flow import (
    CslStencilHandleAsyncControlFlow,
)
from xdsl.transforms.csl_stencil_materialize_stores import CslStencilMaterializeStores
from xdsl.transforms.csl_stencil_to_csl_wrapper import CslStencilToCslWrapperPass
from xdsl.transforms.csl_wrapper_hoist_buffers import CslWrapperHoistBuffers
from xdsl.transforms.experimental.dmp.stencil_global_to_local import (
    DistributeStencilPass,
)
from xdsl.transforms.experimental.stencil_tensorize_z_dimension import (
    StencilTensorizeZDimension,
)
from xdsl.transforms.lift_arith_to_linalg import LiftArithToLinalg
from xdsl.transforms.linalg_to_csl import LinalgToCsl
from xdsl.transforms.lower_csl_stencil import LowerCslStencil
from xdsl.transforms.lower_csl_wrapper import LowerCslWrapperPass
from xdsl.transforms.memref_to_dsd import MemrefToDsdPass
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.transforms.shape_inference import ShapeInferencePass
from xdsl.transforms.stencil_bufferize import StencilBufferize


@dataclass(frozen=True)
class TestConvertStencilToCslPass(ModulePass):
    """
    This is a pass combining all passes currently required to lower code from
    the `stencil` dialect to the `csl` dialect.
    """

    name = "test-convert-stencil-to-csl"

    slices: tuple[int, int]

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PipelinePass(
            (
                DistributeStencilPass(strategy="2d-grid", slices=self.slices),
                ShapeInferencePass(),
                StencilTensorizeZDimension(),
                CommonSubexpressionElimination(),
                StencilBufferize(),
                ConvertStencilToCslStencilPass(),
                LiftArithToLinalg(),
                CslStencilBufferize(),
                CslStencilToCslWrapperPass(),
                CommonSubexpressionElimination(),
                MLIROptPass(
                    arguments=(
                        "--allow-unregistered-dialect",
                        "--one-shot-bufferize=allow-unknown-ops",
                        "--cse",
                        "--canonicalize",
                    )
                ),
                CslStencilMaterializeStores(),
                LinalgToCsl(),
                CslStencilHandleAsyncControlFlow(),
                LowerCslStencil(),
                CslWrapperHoistBuffers(),
                MemrefToDsdPass(),
                CommonSubexpressionElimination(),
                LowerCslWrapperPass(),
            )
        ).apply(ctx, op)
