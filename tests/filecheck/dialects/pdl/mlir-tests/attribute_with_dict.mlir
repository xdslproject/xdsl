// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.operation"() {attributeValueNames = [], operand_segment_sizes = array<i32: 0, 0, 0>} : () -> !pdl.operation
    "pdl.rewrite"(%0) ({
      %1 = "pdl.attribute"() {pdl.special_attribute, value = {some_unit_attr}} : () -> !pdl.attribute
      "pdl.apply_native_rewrite"(%1) {name = "NativeRewrite"} : (!pdl.attribute) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "attribute_with_dict"} : () -> ()
}) : () -> ()



// CHECK:       "builtin.module"() ({
// CHECK-NEXT:    "pdl.pattern"() ({
// CHECK-NEXT:      %0 = "pdl.operation"() {"attributeValueNames" = [], "operand_segment_sizes" = array<i32: 0, 0, 0>} : () -> !pdl.operation
// CHECK-NEXT:      "pdl.rewrite"(%0) ({
// CHECK-NEXT:        %1 = "pdl.attribute"() {"pdl.special_attribute", "value" = {"some_unit_attr"=}} : () -> !pdl.attribute
// CHECK-NEXT:        "pdl.apply_native_rewrite"(%1) {"name" = "NativeRewrite"} : (!pdl.attribute) -> ()
// CHECK-NEXT:      }) {"operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK-NEXT:    }) {"benefit" = 1 : i16, "sym_name" = "attribute_with_dict"} : () -> ()
// CHECK-NEXT:  }) : () -> ()
