// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  pdl.pattern @infer_type_from_operation_replace : benefit(1) {
    %0 = pdl.type : i32
    %1 = pdl.type
    %2 = pdl.operation -> (%0, %1 : !pdl.type, !pdl.type)
    pdl.rewrite %2 {
      %3 = pdl.type
      %4 = pdl.operation "foo.op" -> (%0, %3 : !pdl.type, !pdl.type)
      "pdl.replace"(%2, %4) {"operand_segment_sizes" = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }
  }
}) : () -> ()


// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @infer_type_from_operation_replace : benefit(1) {
// CHECK-NEXT:     %0 = pdl.type : i32
// CHECK-NEXT:     %1 = pdl.type
// CHECK-NEXT:     %2 = pdl.operation -> (%0, %1 : !pdl.type, !pdl.type)
// CHECK-NEXT:     pdl.rewrite %2 {
// CHECK-NEXT:       %3 = pdl.type
// CHECK-NEXT:       %4 = pdl.operation "foo.op" -> (%0, %3 : !pdl.type, !pdl.type)
// CHECK-NEXT:       "pdl.replace"(%2, %4) {"operand_segment_sizes" = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
