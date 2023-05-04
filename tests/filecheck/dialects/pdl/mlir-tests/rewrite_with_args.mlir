// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operand"() : () -> !pdl.value
    %1 = "pdl.operation"(%0) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 0>} : (!pdl.value) -> !pdl.operation
    "pdl.rewrite"(%1, %0) ({
    }) {name = "rewriter", operand_segment_sizes = array<i32: 1, 1>} : (!pdl.operation, !pdl.value) -> ()
  }) {benefit = 1 : i16, sym_name = "rewrite_with_args"} : () -> ()
}) : () -> ()



// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "pdl.pattern"() ({
// CHECK-NEXT:      %0 = "pdl.operand"() : () -> !pdl.value
// CHECK-NEXT:      %1 = "pdl.operation"(%0) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 1, 0, 0>} : (!pdl.value) -> !pdl.operation
// CHECK-NEXT:      "pdl.rewrite"(%1, %0) ({
// CHECK-NEXT:      }) {"name" = "rewriter", "operand_segment_sizes" = array<i32: 1, 1>} : (!pdl.operation, !pdl.value) -> ()
// CHECK-NEXT:    }) {"benefit" = 1 : i16, "sym_name" = "rewrite_with_args"} : () -> ()
// CHECK-NEXT:  }) : () -> ()
