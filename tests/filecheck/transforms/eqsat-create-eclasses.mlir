// RUN: xdsl-opt -p eqsat-create-eclasses %s | filecheck %s

func.func @test(%x : index) -> (index) {
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    func.return %res : index
}

// CHECK:       func.func @test(%x : index) -> (index) {
// CHECK-NEXT:      %x_eq = eqsat.eclass %x : index
// CHECK-NEXT:      %c2 = arith.constant 2 : index
// CHECK-NEXT:      %c2_eq = eqsat.eclass %c2 : index
// CHECK-NEXT:      %res = arith.muli %x_eq, %c2_eq : index
// CHECK-NEXT:      %res_eq = eqsat.eclass %res : index
// CHECK-NEXT:      func.return %res_eq : index
// CHECK-NEXT:  }
