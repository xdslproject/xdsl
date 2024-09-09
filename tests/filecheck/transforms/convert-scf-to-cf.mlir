// RUN: xdsl-opt -p convert-scf-to-cf %s | filecheck %s

func.func @triangle(%n : i32) -> (i32) {
  // Initial sum set to 0.
  %sum_0 = arith.constant 0 : i32

  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  // iter_args binds initial values to the loop's region arguments.
  %sum = scf.for %iv = %zero to %n step %one
    iter_args(%sum_iter = %sum_0) -> (i32) : i32 {

    %sum_next = arith.addi %sum_iter, %iv : i32
    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    scf.yield %sum_next : i32
  }
  return %sum : i32
}

// CHECK:      func.func @triangle(%n : i32) -> i32 {
// CHECK-NEXT:   %sum = arith.constant 0 : i32
// CHECK-NEXT:   %zero = arith.constant 0 : i32
// CHECK-NEXT:   %one = arith.constant 1 : i32
// CHECK-NEXT:   cf.br ^[[#b0:]](%zero, %sum : i32, i32)
// CHECK-NEXT: ^[[#b0]](%iv : i32, %sum_iter : i32):
// CHECK-NEXT:   %[[#v0:]] = arith.cmpi slt, %iv, %n : i32
// CHECK-NEXT:   cf.cond_br %[[#v0]], ^[[#b1:]], ^[[#b2:]]
// CHECK-NEXT: ^[[#b1]]:
// CHECK-NEXT:   %sum_next = arith.addi %sum_iter, %iv : i32
// CHECK-NEXT:   %[[#v1:]] = arith.addi %iv, %one : i32
// CHECK-NEXT:   cf.br ^[[#b0]](%[[#v1]], %sum_next : i32, i32)
// CHECK-NEXT: ^[[#b2]]:
// CHECK-NEXT:   func.return %sum_iter : i32
// CHECK-NEXT: }

func.func @if(%b : i1) -> (i32) {
  %ret = scf.if %b -> (i32) {
    %one = arith.constant 1 : i32
    scf.yield %one : i32
  } else {
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
  }
  return %ret : i32
}

// CHECK:      func.func @if(%b : i1) -> i32 {
// CHECK-NEXT:   cf.cond_br %b, ^[[#b0:]], ^[[#b1:]]
// CHECK-NEXT: ^[[#b0]]:
// CHECK-NEXT:   %one = arith.constant 1 : i32
// CHECK-NEXT:   cf.br ^[[#b2:]](%one : i32)
// CHECK-NEXT: ^[[#b1]]:
// CHECK-NEXT:   %zero = arith.constant 0 : i32
// CHECK-NEXT:   cf.br ^[[#b2]](%zero : i32)
// CHECK-NEXT: ^[[#b2]](%ret : i32):
// CHECK-NEXT:   cf.br ^[[#b3:]]
// CHECK-NEXT: ^[[#b3]]:
// CHECK-NEXT:   func.return %ret : i32
// CHECK-NEXT: }

func.func @nested(%n : index) -> (index) {
  // Initial sum set to 0.
  %sum_0 = arith.constant 0 : index

  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index

  %sum = scf.for %iv = %zero to %n step %one
    iter_args(%sum_iter = %sum_0) -> (index) : index {

    %cond = arith.constant true

    %sum_next = scf.if %cond -> (index) {
      %0 = arith.addi %sum_iter, %iv : index
      scf.yield %0 : index
    } else {
      scf.yield %sum_iter : index
    }
    scf.yield %sum_next : index
  }
  return %sum : index
}

// CHECK:      func.func @nested(%n : index) -> index {
// CHECK-NEXT:   %sum = arith.constant 0 : index
// CHECK-NEXT:   %zero = arith.constant 0 : index
// CHECK-NEXT:   %one = arith.constant 1 : index
// CHECK-NEXT:   %two = arith.constant 2 : index
// CHECK-NEXT:   cf.br ^[[#b0:]](%zero, %sum : index, index)
// CHECK-NEXT: ^[[#b0]](%iv : index, %sum_iter : index):
// CHECK-NEXT:   %[[#v0:]] = arith.cmpi slt, %iv, %n : index
// CHECK-NEXT:   cf.cond_br %[[#v0]], ^[[#b1:]], ^[[#b2:]]
// CHECK-NEXT: ^[[#b1]]:
// CHECK-NEXT:   %cond = arith.constant true
// CHECK-NEXT:   cf.cond_br %cond, ^[[#b3:]], ^[[#b4:]]
// CHECK-NEXT: ^[[#b3]]:
// CHECK-NEXT:   %[[#v1:]] = arith.addi %sum_iter, %iv : index
// CHECK-NEXT:   cf.br ^[[#b5:]](%[[#v1]] : index)
// CHECK-NEXT: ^[[#b4]]:
// CHECK-NEXT:   cf.br ^[[#b5]](%sum_iter : index)
// CHECK-NEXT: ^[[#b5]](%sum_next : index):
// CHECK-NEXT:   cf.br ^[[#b6:]]
// CHECK-NEXT: ^[[#b6]]:
// CHECK-NEXT:   %[[#v2:]] = arith.addi %iv, %one : index
// CHECK-NEXT:   cf.br ^[[#b0]](%[[#v2]], %sum_next : index, index)
// CHECK-NEXT: ^[[#b2]]:
// CHECK-NEXT:   func.return %sum_iter : index
// CHECK-NEXT: }
