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
  }) {benefit = 2 : i16, sym_name = "erasing_an_op_that_is_still_used"} : () -> ()

}) {sym_name = "patterns"} : () -> ()

