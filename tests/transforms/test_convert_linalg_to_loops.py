import pytest

from xdsl.context import Context
from xdsl.dialects import get_all_dialects
from xdsl.parser import Parser
from xdsl.transforms.convert_linalg_to_loops import ConvertLinalgToLoopsPass
from xdsl.utils.exceptions import PassFailedException


def test_convert_linalg_to_loops_rejects_tensor():
    ctx = Context()
    for name in ("affine", "arith", "builtin", "linalg", "memref", "test"):
        ctx.register_dialect(name, get_all_dialects()[name])

    module = Parser(
        ctx,
        """
builtin.module {
  %input = "test.op"() : () -> tensor<?x?xf32>
  %output = "test.op"() : () -> memref<?x?xf32>
  linalg.generic {
      indexing_maps = [
          affine_map<(i, j) -> (i, j)>,
          affine_map<(i, j) -> (i, j)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<?x?xf32>) outs(%output : memref<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
}
""",
    ).parse_module()

    with pytest.raises(
        PassFailedException,
        match="convert-linalg-to-loops requires buffer semantics",
    ):
        ConvertLinalgToLoopsPass().apply(ctx, module)
