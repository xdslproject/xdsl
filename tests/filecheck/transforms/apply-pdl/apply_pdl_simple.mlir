// RUN: xdsl-opt %s -p apply-pdl | filecheck %s

"test.op"() {attr = 0} : () -> ()

pdl.pattern : benefit(1) {
    %zero_attr = pdl.attribute = 0
    %root = pdl.operation "test.op" {"attr" = %zero_attr}
    pdl.rewrite %root {
      %one_attr = pdl.attribute = 1
      %new_op = pdl.operation "test.op" {"attr" = %one_attr}
      pdl.replace %root with %new_op
    }
}

//CHECK:         builtin.module {
// CHECK-NEXT:      "test.op"() {attr = 1 : i64} : () -> ()
// CHECK-NEXT:      pdl.pattern : benefit(1) {
// CHECK-NEXT:         %zero_attr = pdl.attribute = 0 : i64
// CHECK-NEXT:         %root = pdl.operation "test.op" {"attr" = %zero_attr}
// CHECK-NEXT:         pdl.rewrite %root {
// CHECK-NEXT:            %one_attr = pdl.attribute = 1 : i64
// CHECK-NEXT:            %new_op = pdl.operation "test.op" {"attr" = %one_attr}
// CHECK-NEXT:            pdl.replace %root with %new_op
// CHECK-NEXT:         }
// CHECK-NEXT:      }
// CHECK-NEXT:   }
