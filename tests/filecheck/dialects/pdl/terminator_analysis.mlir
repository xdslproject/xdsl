// RUN: xdsl-opt -t mlir %s -p pdl-analysis | filecheck %s

// CHECK: "builtin.module"()
"builtin.module"() ({

  // replacing a terminator with a terminator
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    %2 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK-NOT: "terminator_replaced_with_non_terminator"
    "pdl.rewrite"(%2) ({
      %3 = "pdl.type"() : () -> !pdl.type
      %4 = "pdl.operation"() {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
      "pdl.replace"(%2, %4) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "valid_terminator_replacement"} : () -> ()

  // -----
  // replacing a terminator with an unknown op

  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    %2 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK: "terminator_replaced_with_non_terminator"
    "pdl.rewrite"(%2) ({
      %3 = "pdl.type"() : () -> !pdl.type
      %4 = "pdl.operation"() {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
      "pdl.replace"(%2, %4) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "invalid_terminator_replacement"} : () -> ()
}) : () -> ()