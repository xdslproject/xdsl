"builtin.module"() ({
  func.func @loopUnroll(%arg0: index, %arg1: index) -> index {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %0:2 = scf.for %arg2 = %c0 to %c32 step %c1
            iter_args(%arg3 = %arg0, %arg4 = %arg1)
            -> (index, index) {
        %8 = arith.addi %arg4, %arg4 : index
        %9 = arith.subi %arg4, %arg3 : index
        scf.yield %9, %8 : index, index
    }
    %1:2 = scf.for %arg2 = %c0 to %c32 step %c1
            iter_args(%arg3 = %arg0, %arg4 = %arg1)
            -> (index, index) {
        %8 = arith.addi %arg3, %arg4 : index
        %9 = arith.subi %arg4, %arg3 : index
        scf.yield %9, %8 : index, index
    }
    return %1#0: index
  }
}) : () -> ()
