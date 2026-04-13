import sys
from xdsl.tools.xdsl_opt import main
sys.argv = ['xdsl-opt', 'tests/filecheck/transforms/ematch-saturate/ematch_saturate_func_call_reg.mlir', '-p', 'ematch-saturate{max_iterations=3}']
try:
    main()
except Exception as e:
    import traceback
    traceback.print_exc()
