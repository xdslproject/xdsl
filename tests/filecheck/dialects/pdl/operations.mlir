// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.attribute"() : () -> !pdl.attribute
    %1 = "pdl.type"() : () -> !pdl.type
    %2 = "pdl.operation"(%0, %1) {"attributeValueNames" = ["attr"], "operand_segment_sizes" = array<i32: 0, 1, 1>} : (!pdl.attribute, !pdl.type) -> !pdl.operation
    %3 = "pdl.result"(%2) {"index" = 0 : i32} : (!pdl.operation) -> !pdl.value
    %4 = "pdl.operand"() : () -> !pdl.value
    %5 = "pdl.operation"(%3, %4) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
    "pdl.rewrite"(%5) ({}) {"name" = "rewriter", "operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {"benefit" = 1 : i16, "sym_name" = "operations"} : () -> ()
}) : () -> ()

// CHECK:       "builtin.module"() ({
// CHECK-NEXT:     "pdl.pattern"() ({
// CHECK-NEXT:       %0 = "pdl.attribute"() : () -> !pdl.attribute
// CHECK-NEXT:       %1 = "pdl.type"() : () -> !pdl.type
// CHECK-NEXT:       %2 = "pdl.operation"(%0, %1) {"attributeValueNames" = ["attr"], "operand_segment_sizes" = array<i32: 0, 1, 1>} : (!pdl.attribute, !pdl.type) -> !pdl.operation
// CHECK-NEXT:       %3 = "pdl.result"(%2) {"index" = 0 : i32} : (!pdl.operation) -> !pdl.value
// CHECK-NEXT:       %4 = "pdl.operand"() : () -> !pdl.value
// CHECK-NEXT:       %5 = "pdl.operation"(%3, %4) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
// CHECK-NEXT:       "pdl.rewrite"(%5) ({}) {"name" = "rewriter", "operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK-NEXT:     }) {"benefit" = 1 : i16, "sym_name" = "operations"} : () -> ()
// CHECK-NEXT:   }) : () -> ()

