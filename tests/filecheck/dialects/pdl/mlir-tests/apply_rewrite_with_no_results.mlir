// RUN: xdsl-opt %s --verify-diagnostics | xdsl-opt | filecheck %s

"builtin.module"() ({
  pdl.pattern @apply_rewrite_with_no_results : benefit(1) {
    %0 = pdl.operation
    pdl.rewrite %0 {
      pdl.apply_native_rewrite "NativeRewrite"(%0 : !pdl.operation)
    }
  }
}) : () -> ()


// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @apply_rewrite_with_no_results : benefit(1) {
// CHECK-NEXT:     %0 = pdl.operation
// CHECK-NEXT:     pdl.rewrite %0 {
// CHECK-NEXT:       pdl.apply_native_rewrite "NativeRewrite"(%0 : !pdl.operation)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
