// RUN: xdsl-opt -p eqsat-create-eclasses %s | filecheck %s

func.func @test(%x : index) -> (index) {
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    func.return %res : index
}

// CHECK:       func.func @test(%x : index) -> index {
// CHECK-NEXT:    %x_1 = equivalence.class %x : index
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c2_1 = equivalence.class %c2 : index
// CHECK-NEXT:    %res = arith.muli %x_1, %c2_1 : index
// CHECK-NEXT:    %res_1 = equivalence.class %res : index
// CHECK-NEXT:    func.return %res_1 : index
// CHECK-NEXT:  }
