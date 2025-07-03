// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

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

// -----

func.func @omp_target_data_no_map_info(%m : memref<1xf32>) {
  "omp.target_data"(%m) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> ({
  ^0(%0 : memref<1xf32>, %1 : memref<1xf32>, %2 : memref<1xf32>):
    "omp.terminator"() : () -> ()
  }) : (memref<1xf32>) -> ()
  func.return
}

// CHECK: All mapped operands of omp.target_data must be results of a omp.map.info

// -----

func.func @omp_target_data_delete(%m : memref<1xf32>) {
  %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x08 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_data"(%m1) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> ({
  ^0(%0 : memref<1xf32>, %1 : memref<1xf32>, %2 : memref<1xf32>):
    "omp.terminator"() : () -> ()
  }) : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_data

// -----

func.func @omp_target_data_to_and_delete(%dev : i64, %if : i1, %m : memref<1xf32>, %d1 : memref<1xf32>, %d2 : memref<1xf32>) {
  %m1 = "omp.map.info"(%m) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>, var_type = memref<1xf32>, map_type = 0x9 : ui64, map_capture_type = #omp<variable_capture_kind(ByCopy)>}> : (memref<1xf32>) -> memref<1xf32>
  "omp.target_data"(%m1) <{operandSegmentSizes = array<i32: 0, 0, 1, 0, 0>}> ({
  ^0(%0 : memref<1xf32>, %1 : memref<1xf32>, %2 : memref<1xf32>):
    "omp.terminator"() : () -> ()
  }) : (memref<1xf32>) -> ()
  func.return
}

// CHECK: Cannot have map_type DELETE in omp.target_data
