from xdsl.dialects import pdl
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext, Operation


def pdl_analysis_pass(context: MLContext, prog: ModuleOp):

    def analyze_if_pdl_pattern(op: Operation):
        if not isinstance(op, pdl.PatternOp):
            return
        print("Found pattern:", op.name)

    prog.walk(analyze_if_pdl_pattern)
    print("Analysis not implemented yet.")
