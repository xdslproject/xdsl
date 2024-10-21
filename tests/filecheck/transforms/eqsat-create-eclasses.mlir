// RUN: xdsl-opt -p eqsat-create-eclasses %s | filecheck %s

func.func @test(%x : index) -> (index) {
    %c2 = arith.constant 2 : index
    func.return %c2 : index
}

// CHECK:  func.func @test(%x : index) -> index {
// CHECK-NEXT:    %x_1 = eqsat.eclass %x : index
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c2_1 = eqsat.eclass %c2 : index
// CHECK-NEXT:    func.return %c2_1 : index
// CHECK-NEXT:  }
