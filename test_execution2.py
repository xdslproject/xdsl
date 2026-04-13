import sys
from xdsl.tools.xdsl_opt import main
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
old_run_replace = PDLInterpFunctions.run_replace
def new_run_replace(self, interpreter, op, args):
    print("RUN REPLACE CALLED ON:", op)
    return old_run_replace(self, interpreter, op, args)
PDLInterpFunctions.run_replace = new_run_replace
sys.argv = ['xdsl-opt', 'tests/filecheck/transforms/ematch-saturate/ematch_saturate_func_call_reg.mlir', '-p', 'ematch-saturate{max_iterations=3}']
try:
    main()
except Exception as e:
    import traceback
    traceback.print_exc()
