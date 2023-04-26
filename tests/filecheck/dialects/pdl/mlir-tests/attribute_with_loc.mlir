// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.attribute"() : () -> !pdl.attribute
    %1 = "pdl.operation"(%0) {attributeValueNames = ["attribute"], operand_segment_sizes = array<i32: 0, 1, 0>} : (!pdl.attribute) -> !pdl.operation
    "pdl.rewrite"(%1) ({
    }) {name = "rewriter", operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "attribute_with_loc"} : () -> ()
}) : () -> ()


// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "pdl.pattern"() ({
// CHECK-NEXT:      %0 = "pdl.attribute"() : () -> !pdl.attribute
// CHECK-NEXT:      %1 = "pdl.operation"(%0) {"attributeValueNames" = ["attribute"], "operand_segment_sizes" = array<i32: 0, 1, 0>} : (!pdl.attribute) -> !pdl.operation
// CHECK-NEXT:      "pdl.rewrite"(%1) ({
// CHECK-NEXT:      }) {"name" = "rewriter", "operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK-NEXT:    }) {"benefit" = 1 : i16, "sym_name" = "attribute_with_loc"} : () -> ()
// CHECK-NEXT:  }) : () -> ()
