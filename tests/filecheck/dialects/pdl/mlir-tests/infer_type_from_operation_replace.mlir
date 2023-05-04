// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.type"() : () -> !pdl.type
    %2 = "pdl.operation"(%0, %1) {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%2) ({
      %3 = "pdl.type"() : () -> !pdl.type
      %4 = "pdl.operation"(%0, %3) {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
      "pdl.replace"(%2, %4) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_operation_replace"} : () -> ()
}) : () -> ()



// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "pdl.pattern"() ({
// CHECK-NEXT:      %0 = "pdl.type"() {"constantType" = i32} : () -> !pdl.type
// CHECK-NEXT:      %1 = "pdl.type"() : () -> !pdl.type
// CHECK-NEXT:      %2 = "pdl.operation"(%0, %1) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
// CHECK-NEXT:      "pdl.rewrite"(%2) ({
// CHECK-NEXT:        %3 = "pdl.type"() : () -> !pdl.type
// CHECK-NEXT:        %4 = "pdl.operation"(%0, %3) {"attributeValueNames" = [], "opName" = "foo.op", "operand_segment_sizes" = array<i32: 0, 0, 2>} : (!pdl.type, !pdl.type) -> !pdl.operation
// CHECK-NEXT:        "pdl.replace"(%2, %4) {"operand_segment_sizes" = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
// CHECK-NEXT:      }) {"operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK-NEXT:    }) {"benefit" = 1 : i16, "sym_name" = "infer_type_from_operation_replace"} : () -> ()
// CHECK-NEXT:  }) : () -> ()
