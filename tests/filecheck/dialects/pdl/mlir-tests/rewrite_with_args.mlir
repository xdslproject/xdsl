// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  pdl.pattern @rewrite_with_args : benefit(1) {
    %0 = pdl.operand
    %1 = pdl.operation (%0 : !pdl.value)
    pdl.rewrite %1 {
    }
  }
}) : () -> ()



// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @rewrite_with_args : benefit(1) {
// CHECK-NEXT:     %0 = pdl.operand
// CHECK-NEXT:     %1 = pdl.operation (%0 : !pdl.value)
// CHECK-NEXT:     pdl.rewrite %1 {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
