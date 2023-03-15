// RUN: xdsl-opt -t mlir %s -p pdl-analysis | filecheck %s

// CHECK: "builtin.module"()
"builtin.module"() ({

  // replacing a terminator with a terminator
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    // CHECK-NEXT: "pdl.operation"()
    %0 = "pdl.operation"() {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK-NOT: "terminator_replaced_with_non_terminator"
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%0) ({
      %1 = "pdl.operation"() {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
      "pdl.replace"(%0, %1) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "replacing_a_terminator_with_a_terminator"} : () -> ()

  // -----
  // replacing an unknown op with a terminator
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    // CHECK-NEXT: "pdl.operation"()
    %0 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK-NOT: "terminator_replaced_with_non_terminator"
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%0) ({
      %1 = "pdl.operation"() {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
      "pdl.replace"(%0, %1) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "replacing_an_unknown_op_with_a_terminator"} : () -> ()


  // -----
  // replacing a terminator op with an unknown op
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operation"() {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK-NEXT: "terminator_replaced_with_non_terminator"
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%0) ({
      %1 = "pdl.operation"() {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
      "pdl.replace"(%0, %1) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "replacing_a_terminator_op_with_an_unknown_op"} : () -> ()


  // -----
  // replacing an unknown op with an unknown op
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK-NEXT: "terminator_replaced_with_non_terminator"
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%0) ({
      %1 = "pdl.operation"() {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
      "pdl.replace"(%0, %1) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "replacing_an_unknown_op_with_an_unknown_op"} : () -> ()


  // -----
  // erasing a terminator
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operation"() {attributeValueNames = [], opName = "func.return", operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK-NEXT: "terminator_erased"
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%0) ({
      "pdl.erase"(%0) : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "erasing_a_terminator"} : () -> ()


  // -----
  // erasing an unknown op
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK-NEXT: "terminator_erased"
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%0) ({
      "pdl.erase"(%0) : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "erasing_an_unknown_op"} : () -> ()

}) : () -> ()