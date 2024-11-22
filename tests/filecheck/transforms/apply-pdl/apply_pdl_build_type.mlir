// RUN: xdsl-opt %s -p apply-pdl | filecheck %s

%x = "test.op"() : () -> (i32) 

pdl.pattern : benefit(1) {
    %in_type = pdl.type: i32
    %root = pdl.operation "test.op" -> (%in_type: !pdl.type)
    pdl.rewrite %root {
      %out_type = pdl.type: i64
      %new_op = pdl.operation "test.op" -> (%out_type: !pdl.type)
      pdl.replace %root with %new_op
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %x = "test.op"() : () -> i64
// CHECK-NEXT:    pdl.pattern : benefit(1) {
// CHECK-NEXT:      %in_type = pdl.type : i32
// CHECK-NEXT:      %root = pdl.operation "test.op" -> (%in_type : !pdl.type)
// CHECK-NEXT:      pdl.rewrite %root {
// CHECK-NEXT:        %out_type = pdl.type : i64
// CHECK-NEXT:        %new_op = pdl.operation "test.op" -> (%out_type : !pdl.type)
// CHECK-NEXT:        pdl.replace %root with %new_op
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  
