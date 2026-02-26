// RUN: xdsl-opt -p 'eqsat-create-eclasses,eqsat-add-costs{default=1},eqsat-extract' %s | filecheck %s

func.func @test(%x : index) -> (index) {
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    func.return %res : index
}

// CHECK:  func.func @test(%x : index) -> index {
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %res = arith.muli %x, %c2 : index
// CHECK-NEXT:    func.return %res : index
// CHECK-NEXT:  }
