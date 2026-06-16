// RUN: xdsl-opt %s -p apply-pdl | filecheck %s

%x = arith.constant 42: i32
%y = arith.constant 84: i64
"test.op"(%y) : (i64) -> ()

pdl.pattern : benefit(1) {
    %in_type = pdl.type: i32
    %value = pdl.attribute: %in_type
    %constant_op = pdl.operation "arith.constant" {"value" = %value} -> (%in_type: !pdl.type)
    pdl.rewrite %constant_op {
      pdl.erase %constant_op
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %y = arith.constant 84 : i64
// CHECK-NEXT:    "test.op"(%y) : (i64) -> ()
// CHECK-NEXT:    pdl.pattern : benefit(1) {
// CHECK-NEXT:      %in_type = pdl.type : i32
// CHECK-NEXT:      %value = pdl.attribute : %in_type
// CHECK-NEXT:      %constant_op = pdl.operation "arith.constant" {"value" = %value} -> (%in_type : !pdl.type)
// CHECK-NEXT:      pdl.rewrite %constant_op {
// CHECK-NEXT:        pdl.erase %constant_op
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
