// RUN: xdsl-opt %s --print-op-generic | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.types"() : () -> !pdl.range<!pdl.type>
    %1 = "pdl.operands"(%0) : (!pdl.range<!pdl.type>) -> !pdl.range<!pdl.value>
    %2 = "pdl.operation"(%1) {attributeValueNames = [], operand_segment_sizes = array<i32: 1, 0, 0>} : (!pdl.range<!pdl.value>) -> !pdl.operation
    "pdl.rewrite"(%2) ({
      %3 = "pdl.operation"(%0) {attributeValueNames = [], opName = "foo.op", operand_segment_sizes = array<i32: 0, 0, 1>} : (!pdl.range<!pdl.type>) -> !pdl.operation
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "infer_type_from_type_used_in_match"} : () -> ()
}) : () -> ()



// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "pdl.pattern"() ({
// CHECK-NEXT:      %0 = "pdl.types"() : () -> !pdl.range<!pdl.type>
// CHECK-NEXT:      %1 = "pdl.operands"(%0) : (!pdl.range<!pdl.type>) -> !pdl.range<!pdl.value>
// CHECK-NEXT:      %2 = "pdl.operation"(%1) {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 1, 0, 0>} : (!pdl.range<!pdl.value>) -> !pdl.operation
// CHECK-NEXT:      "pdl.rewrite"(%2) ({
// CHECK-NEXT:        %3 = "pdl.operation"(%0) {"attributeValueNames" = [], "opName" = "foo.op", "operand_segment_sizes" = array<i32: 0, 0, 1>} : (!pdl.range<!pdl.type>) -> !pdl.operation
// CHECK-NEXT:      }) {"operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK-NEXT:    }) {"benefit" = 1 : i16, "sym_name" = "infer_type_from_type_used_in_match"} : () -> ()
// CHECK-NEXT:  }) : () -> ()
