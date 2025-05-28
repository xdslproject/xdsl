// RUN: xdsl-opt -t arm-asm %s --verify-diagnostics --split-input-file | filecheck %s

func.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 0 : i64}> ({
        "omp.loop_nest"(%arg0, %arg1, %arg2) ({
        ^0(%arg7 : i32):
            omp.yield
        }) : (i32, i32, i32) -> ()
        "omp.terminator"() : () -> ()
    }) : () -> ()
    return
}

// CHECK: Operation does not verify: Body of omp.wsloop operation body must consist of one loop nest
