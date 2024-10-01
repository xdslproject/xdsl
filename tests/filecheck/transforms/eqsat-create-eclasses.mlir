// RUN: xdsl-opt -p eqsat-create-eclasses --verify-diagnostics %s | filecheck %s

func.func @test(%x : index) -> (index) {
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    func.return %res : index
}

// CHECK: Error while applying pattern: Ops with non-single results not handled
