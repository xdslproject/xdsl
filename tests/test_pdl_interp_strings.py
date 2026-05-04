from xdsl.parser import Parser
from xdsl.context import Context
from xdsl.dialects import pdl, pdl_interp, builtin
ctx = Context()
pdl.PDL(ctx)
pdl_interp.PDLInterp(ctx)
builtin.Builtin(ctx)
prog = """
builtin.module {
  pdl_interp.func @foo(%arg0: !pdl.operation) {
    %0 = pdl_interp.create_operation "my_op"(%arg0 : !pdl.operation) {"foo" = %arg0} -> <inferred>
    pdl_interp.finalize
  }
}
"""
ctx.clone().parse_module(prog)
