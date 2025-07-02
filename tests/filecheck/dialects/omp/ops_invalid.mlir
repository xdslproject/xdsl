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

// CHECK: Operation does not verify: omp.wsloop is not a LoopWrapper: has 2 ops, expected 1

// -----

func.func @omp_ordered(%arg0 : i32, %arg1 : i32, %arg2 : i32, %arg3 : i64, %arg4 : i64, %arg5 : i64, %arg6 : i64) {
    "omp.wsloop"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, ordered = 0 : i64}> ({
        "omp.terminator"() : () -> ()
    }) : () -> ()
    return
}

// CHECK: omp.wsloop is not a LoopWrapper: should have a single operation which is either another LoopWrapper or omp.loop_nest

// -----

func.func @omp_simd_aligned(%ub : index, %lb : index, %step : index, %a1 : memref<1xi32>, %a2 : memref<10xf32>) {
  "omp.simd"(%a1, %a2) <{operandSegmentSizes = array<i32: 2, 0, 0, 0, 0, 0, 0>, alignments = [64, 8, 16]}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (memref<1xi32>, memref<10xf32>) -> ()
  func.return
}

// CHECK: integer 2 expected from int variable 'ALIGN_COUNT', but got 3

// -----

func.func @omp_simd_linear(%ub : index, %lb : index, %step : index, %l1 : memref<1xi32>, %lstep1 : i32, %lstep2 : i32) {
  "omp.simd"(%l1, %lstep1, %lstep2) <{operandSegmentSizes = array<i32: 0, 0, 1, 2, 0, 0, 0>}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : (memref<1xi32>, i32, i32) -> ()
  func.return
}

// CHECK: integer 1 expected from int variable 'LINEAR_COUNT', but got 2

// -----

func.func @omp_simd_simdlen(%ub : index, %lb : index, %step : index) {
  "omp.simd"() <{operandSegmentSizes = array<i32: 0, 0, 0, 0, 0, 0, 0>, simdlen=8, safelen=2}> ({
    "omp.loop_nest"(%lb, %ub, %step) ({
    ^0(%iter : index):
      omp.yield
    }) : (index, index, index) -> ()

  }) : () -> ()
  func.return
}

// CHECK: `safelen` must be greater than or equal to `simdlen`
