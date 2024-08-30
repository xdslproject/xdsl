// RUN: xdsl-opt -p convert-scf-to-cf %s | filecheck %s

func.func @triangle(%n : i32) -> (i32) {
  // Initial sum set to 0.
  %sum_0 = arith.constant 0 : i32

  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  // iter_args binds initial values to the loop's region arguments.
  %sum = scf.for %iv = %zero to %n step %one
    iter_args(%sum_iter = %sum_0) -> (i32) {

    %sum_next = arith.addi %sum_iter, %iv : i32
    // Yield current iteration sum to next iteration %sum_iter or to %sum
    // if final iteration.
    scf.yield %sum_next : i32
  }
  return %sum : i32
}
