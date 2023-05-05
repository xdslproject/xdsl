// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %0 = pdl.type
    %1 = pdl.type
    %2 = pdl.operand : %0
    %3 = pdl.operand : %1
    %4 = pdl.operation (%2, %3 : !pdl.value, !pdl.value)
    pdl.rewrite %4 {
      %5 = pdl.operation "foo.op" -> (%0, %1 : !pdl.type, !pdl.type)
    }
  }
}) : () -> ()



// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
// CHECK-NEXT:     %0 = pdl.type
// CHECK-NEXT:     %1 = pdl.type
// CHECK-NEXT:     %2 = pdl.operand : %0
// CHECK-NEXT:     %3 = pdl.operand : %1
// CHECK-NEXT:     %4 = pdl.operation (%2, %3 : !pdl.value, !pdl.value)
// CHECK-NEXT:     pdl.rewrite %4 {
// CHECK-NEXT:       %5 = pdl.operation "foo.op" -> (%0, %1 : !pdl.type, !pdl.type)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
