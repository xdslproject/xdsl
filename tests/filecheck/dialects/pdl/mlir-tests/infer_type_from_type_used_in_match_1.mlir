// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.types"() : () -> !pdl.range<!pdl.type>
    %1 = "pdl.operation"(%0) {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 1>} : (!pdl.range<!pdl.type>) -> !pdl.operation
    "pdl.rewrite"(%1) ({
      %2 = "pdl.types"() {constantTypes = [i32, i64]} : () -> !pdl.range<!pdl.type>
      %3 = "pdl.operation"(%0, %2) {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 2>} : (!pdl.range<!pdl.type>, !pdl.range<!pdl.type>) -> !pdl.operation
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"} : () -> ()
}) : () -> ()



// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "pdl.pattern"() ({
// CHECK-NEXT:      %0 = "pdl.types"() : () -> !pdl.range<!pdl.type>
// CHECK-NEXT:      %1 = "pdl.operation"(%0) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 0, 0, 1>} : (!pdl.range<!pdl.type>) -> !pdl.operation
// CHECK-NEXT:      "pdl.rewrite"(%1) ({
// CHECK-NEXT:        %2 = "pdl.types"() {"constantTypes" = [i32, i64]} : () -> !pdl.range<!pdl.type>
// CHECK-NEXT:        %3 = "pdl.operation"(%0, %2) {"attributeValueNames" = [], "opName" = "foo.op", "operand_segment_sizes" = array<i32: 0, 0, 2>} : (!pdl.range<!pdl.type>, !pdl.range<!pdl.type>) -> !pdl.operation
// CHECK-NEXT:      }) {"operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK-NEXT:    }) {"benefit" = 1 : i16, "sym_name" = "infer_type_from_type_used_in_match"} : () -> ()
// CHECK-NEXT:  }) : () -> ()

