// RUN: xdsl-opt %s -p apply-eqsat-pdl | filecheck %s

// CHECK:         func.func @impl() -> i32 {
// CHECK-NEXT:      %c4 = arith.constant 4 : i32
// CHECK-NEXT:      %c4_eq = eqsat.eclass %c4 : i32
// CHECK-NEXT:      %c2 = arith.constant 2 : i32
// CHECK-NEXT:      %c2_eq = eqsat.eclass %c2 : i32
// CHECK-NEXT:      %sum = arith.addi %c2_eq, %c4_eq : i32
// CHECK-NEXT:      %sum_1 = arith.addi %c4_eq, %c2_eq : i32
// CHECK-NEXT:      %sum_eq = eqsat.eclass %sum_1, %sum : i32
// CHECK-NEXT:      func.return %sum_eq : i32
// CHECK-NEXT:    }

func.func @impl() -> i32 {
    %c4 = arith.constant 4 : i32
    %c4_eq = eqsat.eclass %c4 : i32
    %c2 = arith.constant 2 : i32
    %c2_eq = eqsat.eclass %c2 : i32
    %sum = arith.addi %c4_eq, %c2_eq : i32
    %sum_eq = eqsat.eclass %sum : i32
    func.return %sum_eq : i32
}

pdl.pattern : benefit(1) {
    %x = pdl.operand
    %y = pdl.operand
    %type = pdl.type
    %x_y = pdl.operation "arith.addi" (%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    pdl.rewrite %x_y {
        %y_x = pdl.operation "arith.addi" (%y, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        pdl.replace %x_y with %y_x
    }
}
