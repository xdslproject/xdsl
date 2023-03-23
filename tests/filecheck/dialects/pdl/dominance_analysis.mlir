// RUN: xdsl-opt -t mlir %s -p pdl-analysis | filecheck %s

// "builtin.module"() ({
// "pdl.pattern"() ({
//     %0 = "pdl.type"() : () -> !pdl.type
//     %1 = "pdl.operand"() : () -> !pdl.value
//     %2 = "pdl.operand"() : () -> !pdl.value
//     %3 = "pdl.operation"(%1, %2, %0) {attributeValueNames = [], opName = "custom.add", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
//     %4 = "pdl.result"(%3) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
//     "pdl.rewrite"(%3) ({
//       %5 = "pdl.operation"(%4, %0) {attributeValueNames = [], opName = "custom.op", operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
//     }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
// }) {benefit = 5 : i16, sym_name = "required"} : () -> ()
// }) : () -> ()

// CHECK: "builtin.module"()
"builtin.module"() ({

  // erasing an op that is still used
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    // CHECK-NEXT: "pdl.type"()
    %0 = "pdl.type"() : () -> !pdl.type
    // CHECK-NEXT: "pdl.attribute"()
    %1 = "pdl.attribute"() : () -> !pdl.attribute
    // CHECK-NEXT: "pdl.operation"
    %2 = "pdl.operation"(%1, %0) {attributeValueNames = ["value"], opName = "arith.constant", operand_segment_sizes = array<i32: 0, 1, 1>} : (!pdl.attribute, !pdl.type) -> !pdl.operation
    // CHECK-NEXT: "pdl.result"
    %3 = "pdl.result"(%2) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    // CHECK-NEXT: "pdl.operand"
    %4 = "pdl.operand"() : () -> !pdl.value
    // CHECK-NEXT: "pdl.operation"
    %5 = "pdl.operation"(%3, %4, %0) {attributeValueNames = [], opName = "arith.addi", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%5) ({
    // CHECK-NEXT: "erased_op_still_in_use"
      "pdl.erase"(%2) : (!pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "erasing_an_op_that_is_still_used"} : () -> ()


  // -----
  // erasing something not in scope, e.g. double erasure
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    // CHECK-NEXT: "pdl.operation"()
    %0 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%0) ({
      // CHECK-NEXT: "pdl.erase"
      "pdl.erase"(%0) : (!pdl.operation) -> ()
      // CHECK-NEXT: "out_of_scope_erasure"
      "pdl.erase"(%0) : (!pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 2 : i16, sym_name = "erasing_something_not_in_scope"} : () -> ()

  // -----
  // getting the result of something out of scope
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    // CHECK-NEXT: "pdl.operation"()
    %0 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%0) ({
      // CHECK-NEXT: "pdl.erase"
      "pdl.erase"(%0) : (!pdl.operation) -> ()
      // CHECK-NEXT: "val_out_of_scope"
      %1 = "pdl.result"(%0) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 2 : i16, sym_name = "getting_the_result_of_something_out_of_scope"} : () -> ()

  // -----
  // replacing something outside of scope
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    // CHECK-NEXT: "pdl.type"()
    %0 = "pdl.type"() : () -> !pdl.type
    // CHECK-NEXT: "pdl.operand"()
    %1 = "pdl.operand"() : () -> !pdl.value
    // CHECK-NEXT: "pdl.attribute"()
    %2 = "pdl.attribute"() : () -> !pdl.attribute
    // CHECK-NEXT: "pdl.operation"
    %3 = "pdl.operation"(%2, %0) {attributeValueNames = ["value"], opName = "custom.const", operand_segment_sizes = array<i32: 0, 1, 1>} : (!pdl.attribute, !pdl.type) -> !pdl.operation
    // CHECK-NEXT: "pdl.result"
    %4 = "pdl.result"(%3) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    // CHECK-NEXT: "pdl.operation"
    %5 = "pdl.operation"(%1, %4, %0) {attributeValueNames = [], opName = "custom.add", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%5) ({
      // CHECK-NEXT: "pdl.erase"
      "pdl.erase"(%5) : (!pdl.operation) -> ()
      // CHECK-NEXT: "out_of_scope_replacement"
      "pdl.replace"(%5, %3) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "replacing_something_outside_of_scope"} : () -> ()


  // -----
  // using root op in rhs
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({  
    // CHECK-NEXT: "pdl.type"()
    %0 = "pdl.type"() : () -> !pdl.type
    // CHECK-NEXT: "pdl.operand"()
    %1 = "pdl.operand"() : () -> !pdl.value
    // CHECK-NEXT: "pdl.operation"
    %2 = "pdl.operation"(%1, %0) {attributeValueNames = [], opName = "custom.op", operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%2) ({
      // CHECK-NEXT: "pdl.result"
      %3 = "pdl.result"(%2) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
      // CHECK-NEXT: "root_op_used_in_rhs"
      %4 = "pdl.operation"(%3, %0) {attributeValueNames = [], opName = "custom.op2", operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
      "pdl.replace"(%2, %4) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "using_root_op_in_rhs"} : () -> ()


  // -----
  // replacement with itself
  //
  // CHECK: "pdl.pattern"() ({
  "pdl.pattern"() ({
    // CHECK-NEXT: "pdl.type"()
    %0 = "pdl.type"() : () -> !pdl.type
    // CHECK-NEXT: "pdl.operand"()
    %1 = "pdl.operand"() : () -> !pdl.value
    // CHECK-NEXT: "pdl.operation"
    %2 = "pdl.operation"(%1, %0) {attributeValueNames = [], opName = "custom.op", operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.value, !pdl.type) -> !pdl.operation
    // CHECK-NEXT: "pdl.rewrite"
    "pdl.rewrite"(%2) ({
      // CHECK-NEXT: "replacement_with_itself"
      "pdl.replace"(%2, %2) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "replacement_with_itself"} : () -> ()


}) {sym_name = "patterns"} : () -> ()