// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

func.func @while_before_multiple_blocks(%arg0: i32) {
  %res = scf.while (%arg1 = %arg0) : (i32) -> i32 {
    "test.termop"() [^bb1] : () -> ()
  ^bb1:
    %cond = "test.op"() : () -> i1
    scf.condition(%cond) %arg1 : i32
  } do {
  ^bb0(%arg2: i32):
    scf.yield %arg2 : i32
  }
  func.return
}

// CHECK: Operation does not verify: Region 'before_region' at position 0 expected a single block, but got 2 blocks
