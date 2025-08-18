// RUN: xdsl-opt -p eqsat-create-egraphs %s | filecheck %s

// CHECK:      func.func @test(%x : index) -> index {
// CHECK-NEXT:   %res = eqsat.egraph -> index {
// CHECK-NEXT:     %x_1 = eqsat.eclass %x : index
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %c2_1 = eqsat.eclass %c2 : index
// CHECK-NEXT:     %res_1 = arith.muli %x_1, %c2_1 : index
// CHECK-NEXT:     %res_2 = eqsat.eclass %res_1 : index
// CHECK-NEXT:     eqsat.yield %res_2 : index
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %res : index
// CHECK-NEXT: }
func.func @test(%x : index) -> (index) {
  %c2 = arith.constant 2 : index
  %res = arith.muli %x, %c2 : index
  func.return %res : index
}

// CHECK:      func.func @test2(%lb : i32) -> i32 {
// CHECK-NEXT:   %sum = eqsat.egraph -> i32 {
// CHECK-NEXT:     %lb_1 = eqsat.eclass %lb : i32
// CHECK-NEXT:     %ub = arith.constant 42 : i32
// CHECK-NEXT:     %ub_1 = eqsat.eclass %ub : i32
// CHECK-NEXT:     %step = arith.constant 7 : i32
// CHECK-NEXT:     %step_1 = eqsat.eclass %step : i32
// CHECK-NEXT:     %sum_init = arith.constant 36 : i32
// CHECK-NEXT:     %sum_init_1 = eqsat.eclass %sum_init : i32
// CHECK-NEXT:     %sum_1 = scf.for %iv = %lb_1 to %ub_1 step %step_1 iter_args(%sum_iter = %sum_init_1) -> (i32)  : i32 {
// CHECK-NEXT:       %sum_new = arith.addi %sum_iter, %iv : i32
// CHECK-NEXT:       scf.yield %sum_new : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     %sum_2 = eqsat.eclass %sum_1 : i32
// CHECK-NEXT:     eqsat.yield %sum_2 : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %sum : i32
// CHECK-NEXT: }
func.func @test2(%lb: i32) -> (i32) {
  %ub = arith.constant 42 : i32
  %step = arith.constant 7 : i32
  %sum_init = arith.constant 36 : i32
  %sum = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_init) -> (i32) : i32 {
    %sum_new = arith.addi %sum_iter, %iv : i32
    scf.yield %sum_new : i32
  }
  func.return %sum : i32
}

// CHECK:      func.func @test3(%a : index) -> (index, index, index) {
// CHECK-NEXT:   %a_1, %b = eqsat.egraph -> index, index {
// CHECK-NEXT:     %a_2 = eqsat.eclass %a : index
// CHECK-NEXT:     %b_1 = "test.op"(%a_2) : (index) -> index
// CHECK-NEXT:     %b_2 = eqsat.eclass %b_1 : index
// CHECK-NEXT:     eqsat.yield %a_2, %b_2 : index, index
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %a_1, %b, %b : index, index, index
// CHECK-NEXT: }
func.func @test3(%a: index) -> (index, index, index) {
  %b = "test.op"(%a) : (index) -> index
  return %a, %b, %b : index, index, index
}
