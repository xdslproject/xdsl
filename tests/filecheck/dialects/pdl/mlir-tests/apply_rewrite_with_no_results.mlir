// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    "pdl.rewrite"(%0) ({
      "pdl.apply_native_rewrite"(%0) {name = "NativeRewrite"} : (!pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "apply_rewrite_with_no_results"} : () -> ()
}) : () -> ()



// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "pdl.pattern"() ({
// CHECK-NEXT:      %0 = "pdl.operation"() {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> !pdl.operation
// CHECK-NEXT:      "pdl.rewrite"(%0) ({
// CHECK-NEXT:        "pdl.apply_native_rewrite"(%0) {"name" = "NativeRewrite"} : (!pdl.operation) -> ()
// CHECK-NEXT:      }) {"operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK-NEXT:    }) {"benefit" = 1 : i16, "sym_name" = "apply_rewrite_with_no_results"} : () -> ()
// CHECK-NEXT:  }) : () -> ()

