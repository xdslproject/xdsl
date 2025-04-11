// RUN: xdsl-opt -p eqsat-create-eclasses %s | filecheck %s

func.func @test(%x : index) -> (index) {
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    func.return %res : index
}

// CHECK:      func.func @test(%x : index) -> index {
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %res = eqsat.egraph -> index {
// CHECK-NEXT:     %x_1 = eqsat.eclass %x : index
// CHECK-NEXT:     %res_1 = arith.muli %x_1, %c2 : index
// CHECK-NEXT:     %res_2 = eqsat.eclass %res_1 : index
// CHECK-NEXT:     eqsat.yield %res_2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %res : index
// CHECK-NEXT: }
