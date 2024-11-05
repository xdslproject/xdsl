// RUN: xdsl-opt %s -p apply-pdl | filecheck %s

func.func @example(%x: i32, %y: i32, %z: i32) -> i32 {
    %a = arith.addi %x, %y : i32
    %b = arith.subi %a, %y : i32
    %c = arith.muli %b, %z : i32
    return %c : i32
}

pdl.pattern : benefit(1) {
    %in_type = pdl.type: i32
    %x = pdl.operand : %in_type
    %y = pdl.operand : %in_type
    %root_addi = pdl.operation "arith.addi" (%x,%y: !pdl.value, !pdl.value)  -> (%in_type: !pdl.type)
    %a = pdl.result 0 of %root_addi
    %root_subi = pdl.operation "arith.subi" (%a,%y: !pdl.value, !pdl.value)  -> (%in_type: !pdl.type)
    pdl.rewrite %root_subi {
      %zero_attr = pdl.attribute = 0
      %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%in_type: !pdl.type)
      %zero = pdl.result 0 of %zero_op
      %new_addi = pdl.operation "arith.addi" (%x,%zero: !pdl.value, !pdl.value)  -> (%in_type: !pdl.type)
      pdl.replace %root_subi with %new_addi
      pdl.erase %root_addi
    }
}

//CHECK:         builtin.module {
// CHECK-NEXT:     func.func @example(%x : i32, %y : i32, %z : i32) -> i32 {
// CHECK-NEXT:        %0 = arith.constant 0 : i64
// CHECK-NEXT:        %b = arith.addi %x, %0 : i32
// CHECK-NEXT:        %c = arith.muli %b, %z : i32
// CHECK-NEXT:        func.return %c : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     pdl.pattern : benefit(1) {
// CHECK-NEXT:        %in_type = pdl.type : i32
// CHECK-NEXT:        %x = pdl.operand : %in_type
// CHECK-NEXT:        %y = pdl.operand : %in_type
// CHECK-NEXT:        %root_addi = pdl.operation "arith.addi" (%x, %y : !pdl.value, !pdl.value) -> (%in_type : !pdl.type)
// CHECK-NEXT:        %a = pdl.result 0 of %root_addi
// CHECK-NEXT:        %root_subi = pdl.operation "arith.subi" (%a, %y : !pdl.value, !pdl.value) -> (%in_type : !pdl.type)
// CHECK-NEXT:        pdl.rewrite %root_subi {
// CHECK-NEXT:           %zero_attr = pdl.attribute = 0 : i64
// CHECK-NEXT:           %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%in_type : !pdl.type)
// CHECK-NEXT:           %zero = pdl.result 0 of %zero_op
// CHECK-NEXT:           %new_addi = pdl.operation "arith.addi" (%x, %zero : !pdl.value, !pdl.value) -> (%in_type : !pdl.type)
// CHECK-NEXT:           pdl.replace %root_subi with %new_addi
// CHECK-NEXT:           pdl.erase %root_addi
// CHECK-NEXT:        }
// CHECK-NEXT:     }
// CHECK-NEXT:  }
