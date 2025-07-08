// RUN: xdsl-opt %s -p licm | filecheck %s

// CHECK:       builtin.module {

// CHECK-NEXT:    func.func public @simple() {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c8 = arith.constant 8 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c5 = arith.constant 5 : index
// CHECK-NEXT:      scf.for %i = %c0 to %c8 step %c1 {
// CHECK-NEXT:        %l = arith.addi %i, %c5 : index
// CHECK-NEXT:        scf.for %j = %c0 to %c8 step %c1 {
// CHECK-NEXT:          %k = arith.addi %i, %j : index
// CHECK-NEXT:          "test.op"(%c5, %k, %l) : (index, index, index) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
func.func public @simple() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      %c5 = arith.constant 5 : index
      %k = arith.addi %i, %j : index
      %l = arith.addi %i, %c5 : index
      "test.op"(%c5, %k, %l) : (index, index, index) -> ()
    }
  }
  return
}

// CHECK-NEXT: }
