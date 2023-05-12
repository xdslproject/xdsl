// RUN: xdsl-opt %s --print-op-generic | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operand"() : () -> !pdl.value
    %1 = "pdl.operand"() : () -> !pdl.value
    %2 = "pdl.type"() : () -> !pdl.type
    %3 = "pdl.operation"(%0, %2) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
    %4 = "pdl.result"(%3) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %5 = "pdl.operation"(%4) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 0>} : (!pdl.value) -> !pdl.operation
    %6 = "pdl.operation"(%1, %2) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
    %7 = "pdl.result"(%6) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %8 = "pdl.operation"(%4, %7) {attributeValueNames = [], operand_segment_sizes = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
    "pdl.rewrite"(%5, %8) ({
    }) {name = "rewriter", operand_segment_sizes = array<i32: 0, 2>} : (!pdl.operation, !pdl.operation) -> ()
  }) {benefit = 2 : i16, sym_name = "rewrite_multi_root_optimal"} : () -> ()
}) : () -> ()



// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "pdl.pattern"() ({
// CHECK-NEXT:      %0 = "pdl.operand"() : () -> !pdl.value
// CHECK-NEXT:      %1 = "pdl.operand"() : () -> !pdl.value
// CHECK-NEXT:      %2 = "pdl.type"() : () -> !pdl.type
// CHECK-NEXT:      %3 = "pdl.operation"(%0, %2) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
// CHECK-NEXT:      %4 = "pdl.result"(%3) {"index" = 0 : i32} : (!pdl.operation) -> !pdl.value
// CHECK-NEXT:      %5 = "pdl.operation"(%4) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 1, 0, 0>} : (!pdl.value) -> !pdl.operation
// CHECK-NEXT:      %6 = "pdl.operation"(%1, %2) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
// CHECK-NEXT:      %7 = "pdl.result"(%6) {"index" = 0 : i32} : (!pdl.operation) -> !pdl.value
// CHECK-NEXT:      %8 = "pdl.operation"(%4, %7) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
// CHECK-NEXT:      "pdl.rewrite"(%5, %8) ({
// CHECK-NEXT:      }) {"name" = "rewriter", "operand_segment_sizes" = array<i32: 0, 2>} : (!pdl.operation, !pdl.operation) -> ()
// CHECK-NEXT:    }) {"benefit" = 2 : i16, "sym_name" = "rewrite_multi_root_optimal"} : () -> ()
// CHECK-NEXT:  }) : () -> ()
