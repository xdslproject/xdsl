// RUN: xdsl-opt -p eqsat-create-eclasses %s | filecheck %s

func.func @test(%x : index) -> (index) {
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    func.return %res : index
}

// CHECK:     builtin.module {
// CHECK-NEXT:     func.func @test(%x : index) -> index {
// CHECK-NEXT:       %0 = "eqsat.egraph"() ({
// CHECK-NEXT:         %x_1 = eqsat.eclass %x : index
// CHECK-NEXT:         %c2 = arith.constant 2 : index
// CHECK-NEXT:         %c2_1 = eqsat.eclass %c2 : index
// CHECK-NEXT:         %res = arith.muli %x_1, %c2_1 : index
// CHECK-NEXT:         %res_1 = eqsat.eclass %res : index
// CHECK-NEXT:         "eqsat.yield"(%res_1) : (index) -> ()
// CHECK-NEXT:       }) : () -> index
// CHECK-NEXT:       func.return %0 : index
// CHECK-NEXT:     }
// CHECK-NEXT:   }
