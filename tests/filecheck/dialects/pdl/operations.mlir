// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

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

// CHECK-LABEL:   "pdl.pattern"() ({
// CHECK:           %[[VAL_0:.*]] = "pdl.attribute"() : () -> !pdl.attribute
// CHECK:           %[[VAL_1:.*]] = "pdl.type"() : () -> !pdl.type
// CHECK:           %[[VAL_2:.*]] = "pdl.operation"(%[[VAL_0]], %[[VAL_1]]) {attributeValueNames = ["attr"], operand_segment_sizes = array<i32: 0, 1, 1>} : (!pdl.attribute, !pdl.type) -> !pdl.operation
// CHECK:           %[[VAL_3:.*]] = "pdl.result"(%[[VAL_2]]) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
// CHECK:           %[[VAL_4:.*]] = "pdl.operand"() : () -> !pdl.value
// CHECK:           %[[VAL_5:.*]] = "pdl.operation"(%[[VAL_3]], %[[VAL_4]]) {attributeValueNames = [], operand_segment_sizes = array<i32: 2, 0, 0>} : (!pdl.value, !pdl.value) -> !pdl.operation
// CHECK:           "pdl.rewrite"(%[[VAL_5]]) ({
// CHECK:           }) {name = "rewriter", operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK:         }) {benefit = 1 : i16, sym_name = "operations"} : () -> ()
