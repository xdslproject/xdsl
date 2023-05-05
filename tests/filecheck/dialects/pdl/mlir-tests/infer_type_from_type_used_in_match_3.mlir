// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"() ({
  pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
    %0 = pdl.types
    %1 = pdl.operands : %0
    %2 = pdl.operation (%1 : !pdl.range<!pdl.value>)
    pdl.rewrite %2 {
      %3 = pdl.operation "foo.op" -> (%0 : !pdl.range<!pdl.type>)
    }
  }
}) : () -> ()



// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
// CHECK-NEXT:     %0 = pdl.types
// CHECK-NEXT:     %1 = pdl.operands : %0
// CHECK-NEXT:     %2 = pdl.operation (%1 : !pdl.range<!pdl.value>)
// CHECK-NEXT:     pdl.rewrite %2 {
// CHECK-NEXT:       %3 = pdl.operation "foo.op" -> (%0 : !pdl.range<!pdl.type>)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }) : () -> ()
