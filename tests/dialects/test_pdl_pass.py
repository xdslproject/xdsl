from xdsl.ir import MLContext
from xdsl.parser import Parser

import xdsl.dialects.pdl as pdl
from xdsl.dialects.builtin import Builtin
from xdsl.transforms.pdl_analysis import pdl_analysis_pass


def test_pdl_pass():
    test = """
"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.attribute"() : () -> !pdl.attribute
    %1 = "pdl.type"() : () -> !pdl.type
    %2 = "pdl.operation"(%0, %1) {attributeValueNames = ["attr"], operand_segment_sizes = array<i32: 0, 1, 1>} : (!pdl.attribute, !pdl.type) -> !pdl.operation
    %3 = "pdl.result"(%2) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %4 = "pdl.operand"() : () -> !pdl.value
    %5 = "pdl.operation"(%3, %4) {attributeValueNames = [], operand_segment_sizes = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
    "pdl.rewrite"(%5) ({
    }) {name = "rewriter", operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "operations"} : () -> ()
}) : () -> ()
"""

    ctx = MLContext()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(pdl.PDL)

    parser = Parser(ctx=ctx, input=test)
    _module = parser.parse_module()

    pdl_analysis_pass(ctx, _module)


if __name__ == '__main__':
    test_pdl_pass()
