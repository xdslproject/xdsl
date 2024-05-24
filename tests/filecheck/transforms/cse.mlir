// RUN: xdsl-opt --allow-unregistered-dialect %s -p cse | filecheck %s

// CHECK-DAG: #[[$MAP:.*]] = affine_map<(d0) -> (d0 mod 2)>
#map0 = affine_map<(d0) -> (d0 mod 2)>

func.func @simple_constant() -> (i32, i32) {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 1 : i32
    func.return %0, %1 : i32, i32
}

// CHECK:         func.func @simple_constant() -> (i32, i32) {
// CHECK-NEXT:      %0 = arith.constant 1 : i32
// CHECK-NEXT:      func.return %0, %0 : i32, i32
// CHECK-NEXT:    }

// CHECK-LABEL: @basic
  func.func @basic() -> (index, index) {
    %2 = arith.constant 0 : index
    %3 = arith.constant 0 : index
    %4 = "affine.apply"(%2) <{"map" = affine_map<(d0) -> ((d0 mod 2))>}> : (index) -> index
    %5 = "affine.apply"(%3) <{"map" = affine_map<(d0) -> ((d0 mod 2))>}> : (index) -> index
    func.return %4, %5 : index, index
  }

// CHECK:         func.func @basic() -> (index, index) {
// CHECK-NEXT:      %1 = arith.constant 0 : index
// CHECK-NEXT:      %2 = "affine.apply"(%1) <{"map" = affine_map<(d0) -> ((d0 mod 2))>}> : (index) -> index
// CHECK-NEXT:      func.return %2, %2 : index, index
// CHECK-NEXT:    }

// CHECK-LABEL: @many
  func.func @many(%arg0 : f32, %arg1 : f32) -> f32 {
    %6 = arith.addf %arg0, %arg1 : f32
    %7 = arith.addf %arg0, %arg1 : f32
    %8 = arith.addf %arg0, %arg1 : f32
    %9 = arith.addf %arg0, %arg1 : f32
    %10 = arith.addf %6, %7 : f32
    %11 = arith.addf %8, %9 : f32
    %12 = arith.addf %6, %8 : f32
    %13 = arith.addf %10, %11 : f32
    %14 = arith.addf %11, %12 : f32
    %15 = arith.addf %13, %14 : f32
    func.return %15 : f32
  }

// CHECK:         func.func @many(%arg0 : f32, %arg1 : f32) -> f32 {
// CHECK-NEXT:      %3 = arith.addf %arg0, %arg1 : f32
// CHECK-NEXT:      %4 = arith.addf %3, %3 : f32
// CHECK-NEXT:      %5 = arith.addf %4, %4 : f32
// CHECK-NEXT:      %6 = arith.addf %5, %5 : f32
// CHECK-NEXT:      func.return %6 : f32
// CHECK-NEXT:    }

/// Check that operations are not eliminated if they have different operands.
// CHECK-LABEL: @different_ops
func.func @different_ops() -> (i32, i32) {
    %16 = arith.constant 0 : i32
    %17 = arith.constant 1 : i32
    func.return %16, %17 : i32, i32
  }

// CHECK:         func.func @different_ops() -> (i32, i32) {
// CHECK-NEXT:      %7 = arith.constant 0 : i32
// CHECK-NEXT:      %8 = arith.constant 1 : i32
// CHECK-NEXT:      func.return %7, %8 : i32, i32
// CHECK-NEXT:    }

/// Check that operations are not eliminated if they have different result
/// types.
// CHECK-LABEL: @different_results
  func.func @different_results(%arg0_1 : memref<*xf32>) -> (memref<?x?xf32>, memref<4x?xf32>) {
    %18 = "memref.cast"(%arg0_1) : (memref<*xf32>) -> memref<?x?xf32>
    %19 = "memref.cast"(%arg0_1) : (memref<*xf32>) -> memref<4x?xf32>
    func.return %18, %19 : memref<?x?xf32>, memref<4x?xf32>
  }
// CHECK:         func.func @different_results(%arg0_1 : memref<*xf32>) -> (memref<?x?xf32>, memref<4x?xf32>) {
// CHECK-NEXT:      %9 = "memref.cast"(%arg0_1) : (memref<*xf32>) -> memref<?x?xf32>
// CHECK-NEXT:      %10 = "memref.cast"(%arg0_1) : (memref<*xf32>) -> memref<4x?xf32>
// CHECK-NEXT:      func.return %9, %10 : memref<?x?xf32>, memref<4x?xf32>
// CHECK-NEXT:    }

/// Check that operations are not eliminated if they have different attributes.
// CHECK-LABEL: @different_attributes
  func.func @different_attributes(%arg0_2 : index, %arg1_1 : index) -> (i1, i1, i1) {
    %20 = arith.cmpi slt, %arg0_2, %arg1_1 : index
    %21 = arith.cmpi ne, %arg0_2, %arg1_1 : index
    %22 = arith.cmpi ne, %arg0_2, %arg1_1 : index
    func.return %20, %21, %22 : i1, i1, i1
  }

// CHECK:         func.func @different_attributes(%arg0_2 : index, %arg1_1 : index) -> (i1, i1, i1) {
// CHECK-NEXT:      %11 = arith.cmpi slt, %arg0_2, %arg1_1 : index
// CHECK-NEXT:      %12 = arith.cmpi ne, %arg0_2, %arg1_1 : index
// CHECK-NEXT:      %13 = arith.cmpi ne, %arg0_2, %arg1_1 : index
// CHECK-NEXT:      func.return %11, %12, %13 : i1, i1, i1
// CHECK-NEXT:    }

/// Check that operations with side effects are not eliminated.
// CHECK-LABEL: @side_effect
  func.func @side_effect() -> (memref<2x1xf32>, memref<2x1xf32>) {
    %23 = memref.alloc() : memref<2x1xf32>
    %24 = memref.alloc() : memref<2x1xf32>
    func.return %23, %24 : memref<2x1xf32>, memref<2x1xf32>
  }
// CHECK:         func.func @side_effect() -> (memref<2x1xf32>, memref<2x1xf32>) {
// CHECK-NEXT:      %14 = memref.alloc() : memref<2x1xf32>
// CHECK-NEXT:      %15 = memref.alloc() : memref<2x1xf32>
// CHECK-NEXT:      func.return %14, %15 : memref<2x1xf32>, memref<2x1xf32>
// CHECK-NEXT:    }

/// Check that operation definitions are properly propagated down the dominance
/// tree.
// CHECK-LABEL: @down_propagate_for
  func.func @down_propagate_for() {
    %25 = arith.constant 1 : i32
    "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "operandSegmentSizes" = array<i32: 0, 0, 0>, "step" = 1 : index, "upperBoundMap" = affine_map<() -> (4)>}> ({
    ^0(%arg0_3 : index):
      %26 = arith.constant 1 : i32
      "foo"(%25, %26) : (i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    func.return
  }

// CHECK:         func.func @down_propagate_for() {
// CHECK-NEXT:      %16 = arith.constant 1 : i32
// CHECK-NEXT:      "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "operandSegmentSizes" = array<i32: 0, 0, 0>, "step" = 1 : index, "upperBoundMap" = affine_map<() -> (4)>}> ({
// CHECK-NEXT:      ^0(%arg0_3 : index):
// CHECK-NEXT:        "foo"(%16, %16) : (i32, i32) -> ()
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// This would be checking that the constant in the second block is cse'd with the first one
// MLIR has the notion of SSACFG regions (those) and graph regions.
// This works on SSACFG regions only - at least in MLIR implementation.
// We do not have this Region Kind disctinction; so everything here works on the pessimistic
// Graph Rewgion assumption.

// CHECK-LABEL: @down_propagate
func.func @down_propagate() -> i32 {
    %27 = arith.constant 1 : i32
    %28 = arith.constant true
    "cf.cond_br"(%28, %27) [^1, ^2] <{"operandSegmentSizes" = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
  ^1:
    %29 = arith.constant 1 : i32
    "cf.br"(%29) [^2] : (i32) -> ()
  ^2(%30 : i32):
    func.return %30 : i32
  }

// CHECK:         func.func @down_propagate() -> i32 {
// CHECK-NEXT:      %17 = arith.constant 1 : i32
// CHECK-NEXT:      %18 = arith.constant true
// CHECK-NEXT:      "cf.cond_br"(%18, %17) [^1, ^2] <{"operandSegmentSizes" = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
// CHECK-NEXT:    ^1:
// CHECK-NEXT:      %19 = arith.constant 1 : i32
// CHECK-NEXT:      "cf.br"(%19) [^2] : (i32) -> ()
// CHECK-NEXT:    ^2(%20 : i32):
// CHECK-NEXT:      func.return %20 : i32
// CHECK-NEXT:    }

/// Check that operation definitions are NOT propagated up the dominance tree.
// CHECK-LABEL: @up_propagate_for
 func.func @up_propagate_for() -> i32 {
    "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "operandSegmentSizes" = array<i32: 0, 0, 0>, "step" = 1 : index, "upperBoundMap" = affine_map<() -> (4)>}> ({
    ^3(%arg0_4 : index):
      %31 = arith.constant 1 : i32
      "foo"(%31) : (i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %32 = arith.constant 1 : i32
    func.return %32 : i32
  }

// CHECK:         func.func @up_propagate_for() -> i32 {
// CHECK-NEXT:      "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, "operandSegmentSizes" = array<i32: 0, 0, 0>, "step" = 1 : index, "upperBoundMap" = affine_map<() -> (4)>}> ({
// CHECK-NEXT:      ^3(%arg0_4 : index):
// CHECK-NEXT:        %21 = arith.constant 1 : i32
// CHECK-NEXT:        "foo"(%21) : (i32) -> ()
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      %22 = arith.constant 1 : i32
// CHECK-NEXT:      func.return %22 : i32
// CHECK-NEXT:    }

// CHECK-LABEL: func @up_propagate
func.func @up_propagate() -> i32 {
    %33 = arith.constant 0 : i32
    %34 = arith.constant true
    "cf.cond_br"(%34, %33) [^4, ^5] <{"operandSegmentSizes" = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
  ^4:
    %35 = arith.constant 1 : i32
    "cf.br"(%35) [^5] : (i32) -> ()
  ^5(%36 : i32):
    %37 = arith.constant 1 : i32
    %38 = arith.addi %36, %37 : i32
    func.return %38 : i32
  }

// CHECK:         func.func @up_propagate() -> i32 {
// CHECK-NEXT:      %23 = arith.constant 0 : i32
// CHECK-NEXT:      %24 = arith.constant true
// CHECK-NEXT:      "cf.cond_br"(%24, %23) [^4, ^5] <{"operandSegmentSizes" = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
// CHECK-NEXT:    ^4:
// CHECK-NEXT:      %25 = arith.constant 1 : i32
// CHECK-NEXT:      "cf.br"(%25) [^5] : (i32) -> ()
// CHECK-NEXT:    ^5(%26 : i32):
// CHECK-NEXT:      %27 = arith.constant 1 : i32
// CHECK-NEXT:      %28 = arith.addi %26, %27 : i32
// CHECK-NEXT:      func.return %28 : i32
// CHECK-NEXT:    }

/// The same test as above except that we are testing on a cfg embedded within
/// an operation region.
// CHECK-LABEL: func @up_propagate_region
func.func @up_propagate_region() -> i32 {
    %39 = "foo.region"() ({
      %40 = arith.constant 0 : i32
      %41 = arith.constant true
      "cf.cond_br"(%41, %40) [^6, ^7] <{"operandSegmentSizes" = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
    ^6:
      %42 = arith.constant 1 : i32
      "cf.br"(%42) [^7] : (i32) -> ()
    ^7(%43 : i32):
      %44 = arith.constant 1 : i32
      %45 = arith.addi %43, %44 : i32
      "foo.yield"(%45) : (i32) -> ()
    }) : () -> i32
    func.return %39 : i32
  }

// CHECK:         func.func @up_propagate_region() -> i32 {
// CHECK-NEXT:      %29 = "foo.region"() ({
// CHECK-NEXT:        %30 = arith.constant 0 : i32
// CHECK-NEXT:        %31 = arith.constant true
// CHECK-NEXT:        "cf.cond_br"(%31, %30) [^6, ^7] <{"operandSegmentSizes" = array<i32: 1, 0, 1>}> : (i1, i32) -> ()
// CHECK-NEXT:      ^6:
// CHECK-NEXT:        %32 = arith.constant 1 : i32
// CHECK-NEXT:        "cf.br"(%32) [^7] : (i32) -> ()
// CHECK-NEXT:      ^7(%33 : i32):
// CHECK-NEXT:        %34 = arith.constant 1 : i32
// CHECK-NEXT:        %35 = arith.addi %33, %34 : i32
// CHECK-NEXT:        "foo.yield"(%35) : (i32) -> ()
// CHECK-NEXT:      }) : () -> i32
// CHECK-NEXT:      func.return %29 : i32
// CHECK-NEXT:    }

/// This test checks that nested regions that are isolated from above are
/// properly handled.
// CHECK-LABEL: @nested_isolated
func.func @nested_isolated() -> i32 {
    %46 = arith.constant 1 : i32
    func.func @nested_func() {
      %47 = arith.constant 1 : i32
      "foo.yield"(%47) : (i32) -> ()
    }
    "foo.region"() ({
      %48 = arith.constant 1 : i32
      "foo.yield"(%48) : (i32) -> ()
    }) : () -> ()
    func.return %46 : i32
  }

// CHECK:         func.func @nested_isolated() -> i32 {
// CHECK-NEXT:      %36 = arith.constant 1 : i32
// CHECK-NEXT:      func.func @nested_func() {
// CHECK-NEXT:        %37 = arith.constant 1 : i32
// CHECK-NEXT:        "foo.yield"(%37) : (i32) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      "foo.region"() ({
// CHECK-NEXT:        %38 = arith.constant 1 : i32
// CHECK-NEXT:        "foo.yield"(%38) : (i32) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return %36 : i32
// CHECK-NEXT:    }

/// This test is checking that CSE gracefully handles values in graph regions
/// where the use occurs before the def, and one of the defs could be CSE'd with
/// the other.
// CHECK-LABEL: @use_before_def
func.func @use_before_def() {
    "test.graph_region"() ({
      %49 = arith.addi %50, %51 : i32
      %50 = arith.constant 1 : i32
      %51 = arith.constant 1 : i32
      "foo.yield"(%49) : (i32) -> ()
    }) : () -> ()
    func.return
  }

// CHECK:         func.func @use_before_def() {
// CHECK-NEXT:      "test.graph_region"() ({
// CHECK-NEXT:        %39 = arith.addi %40, %41 : i32
// CHECK-NEXT:        %40 = arith.constant 1 : i32
// CHECK-NEXT:        %41 = arith.constant 1 : i32
// CHECK-NEXT:        "foo.yield"(%39) : (i32) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

/// This test is checking that CSE is removing duplicated read op that follow
/// other.
/// NB: xDSL doesn't, we don't have the notion of "read" ops.
// CHECK-LABEL: @remove_direct_duplicated_read_op
  func.func @remove_direct_duplicated_read_op() -> i32 {
    %52 = "test.op_with_memread"() : () -> i32
    %53 = "test.op_with_memread"() : () -> i32
    %54 = arith.addi %52, %53 : i32
    func.return %54 : i32
  }

// CHECK:         func.func @remove_direct_duplicated_read_op() -> i32 {
// CHECK-NEXT:      %42 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      %43 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      %44 = arith.addi %42, %43 : i32
// CHECK-NEXT:      func.return %44 : i32
// CHECK-NEXT:    }


/// This test is checking that CSE is removing duplicated read op that follow
/// other.
/// NB: xDSL doesn't, we don't have the notion of "read" ops.
// CHECK-LABEL: @remove_multiple_duplicated_read_op
  func.func @remove_multiple_duplicated_read_op() -> i64 {
    %55 = "test.op_with_memread"() : () -> i64
    %56 = "test.op_with_memread"() : () -> i64
    %57 = arith.addi %55, %56 : i64
    %58 = "test.op_with_memread"() : () -> i64
    %59 = arith.addi %57, %58 : i64
    %60 = "test.op_with_memread"() : () -> i64
    %61 = arith.addi %59, %60 : i64
    func.return %61 : i64
  }

// CHECK:         func.func @remove_multiple_duplicated_read_op() -> i64 {
// CHECK-NEXT:      %45 = "test.op_with_memread"() : () -> i64
// CHECK-NEXT:      %46 = "test.op_with_memread"() : () -> i64
// CHECK-NEXT:      %47 = arith.addi %45, %46 : i64
// CHECK-NEXT:      %48 = "test.op_with_memread"() : () -> i64
// CHECK-NEXT:      %49 = arith.addi %47, %48 : i64
// CHECK-NEXT:      %50 = "test.op_with_memread"() : () -> i64
// CHECK-NEXT:      %51 = arith.addi %49, %50 : i64
// CHECK-NEXT:      func.return %51 : i64
// CHECK-NEXT:    

/// This test is checking that CSE is not removing duplicated read op that
/// have write op in between.
/// NB: xDSL doesn't, we don't have the notion of "read" ops.
// CHECK-LABEL: @dont_remove_duplicated_read_op_with_sideeffecting
func.func @dont_remove_duplicated_read_op_with_sideeffecting() -> i32 {
    %62 = "test.op_with_memread"() : () -> i32
    "test.op_with_memwrite"() : () -> ()
    %63 = "test.op_with_memread"() : () -> i32
    %64 = arith.addi %62, %63 : i32
    func.return %64 : i32
  }

// CHECK:         func.func @dont_remove_duplicated_read_op_with_sideeffecting() -> i32 {
// CHECK-NEXT:      %52 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      "test.op_with_memwrite"() : () -> ()
// CHECK-NEXT:      %53 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      %54 = arith.addi %52, %53 : i32
// CHECK-NEXT:      func.return %54 : i32
// CHECK-NEXT:    }

// Check that an operation with a single region can CSE.
  func.func @cse_single_block_ops(%arg0_5 : tensor<?x?xf32>, %arg1_2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %65 = "test.pureop"(%arg0_5, %arg1_2) ({
    ^8(%arg2 : f32):
      "test.region_yield"(%arg2) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %66 = "test.pureop"(%arg0_5, %arg1_2) ({
    ^9(%arg2_1 : f32):
      "test.region_yield"(%arg2_1) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    func.return %65, %66 : tensor<?x?xf32>, tensor<?x?xf32>
  }

// CHECK:         func.func @cse_single_block_ops(%arg0_5 : tensor<?x?xf32>, %arg1_2 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %55 = "test.pureop"(%arg0_5, %arg1_2) ({
// CHECK-NEXT:      ^8(%arg2 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg2) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %55, %55 : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK-NEXT:    }

// Operations with different number of bbArgs dont CSE.
func.func @no_cse_varied_bbargs(%arg0_6 : tensor<?x?xf32>, %arg1_3 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %67 = "test.pureop"(%arg0_6, %arg1_3) ({
    ^10(%arg2_2 : f32, %arg3 : f32):
      "test.region_yield"(%arg2_2) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %68 = "test.pureop"(%arg0_6, %arg1_3) ({
    ^11(%arg2_3 : f32):
      "test.region_yield"(%arg2_3) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    func.return %67, %68 : tensor<?x?xf32>, tensor<?x?xf32>
  }

// CHECK:         func.func @no_cse_varied_bbargs(%arg0_6 : tensor<?x?xf32>, %arg1_3 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %56 = "test.pureop"(%arg0_6, %arg1_3) ({
// CHECK-NEXT:      ^9(%arg2_1 : f32, %arg3 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg2_1) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      %57 = "test.pureop"(%arg0_6, %arg1_3) ({
// CHECK-NEXT:      ^10(%arg2_2 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg2_2) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %56, %57 : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK-NEXT:    }

// Operations with different regions dont CSE
func.func @no_cse_region_difference_simple(%arg0_7 : tensor<?x?xf32>, %arg1_4 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %69 = "test.pureop"(%arg0_7, %arg1_4) ({
    ^12(%arg2_4 : f32, %arg3_1 : f32):
      "test.region_yield"(%arg2_4) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %70 = "test.pureop"(%arg0_7, %arg1_4) ({
    ^13(%arg2_5 : f32, %arg3_2 : f32):
      "test.region_yield"(%arg3_2) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    func.return %69, %70 : tensor<?x?xf32>, tensor<?x?xf32>
  }
// CHECK:         func.func @no_cse_region_difference_simple(%arg0_7 : tensor<?x?xf32>, %arg1_4 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %58 = "test.pureop"(%arg0_7, %arg1_4) ({
// CHECK-NEXT:      ^11(%arg2_3 : f32, %arg3_1 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg2_3) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      %59 = "test.pureop"(%arg0_7, %arg1_4) ({
// CHECK-NEXT:      ^12(%arg2_4 : f32, %arg3_2 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg3_2) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %58, %59 : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK-NEXT:    }

// Operation with identical region with multiple statements CSE.
func.func @cse_single_block_ops_identical_bodies(%arg0_8 : tensor<?x?xf32>, %arg1_5 : tensor<?x?xf32>, %arg2_6 : f32, %arg3_3 : i1) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %71 = "test.pureop"(%arg0_8, %arg1_5) ({
    ^14(%arg4 : f32, %arg5 : f32):
      %72 = arith.divf %arg4, %arg5 : f32
      %73 = "arith.remf"(%arg4, %arg2_6) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
      %74 = arith.select %arg3_3, %72, %73 : f32
      "test.region_yield"(%74) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %75 = "test.pureop"(%arg0_8, %arg1_5) ({
    ^15(%arg4_1 : f32, %arg5_1 : f32):
      %76 = arith.divf %arg4_1, %arg5_1 : f32
      %77 = "arith.remf"(%arg4_1, %arg2_6) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
      %78 = arith.select %arg3_3, %76, %77 : f32
      "test.region_yield"(%78) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    func.return %71, %75 : tensor<?x?xf32>, tensor<?x?xf32>
}

// CHECK:         func.func @cse_single_block_ops_identical_bodies(%arg0_8 : tensor<?x?xf32>, %arg1_5 : tensor<?x?xf32>, %arg2_5 : f32, %arg3_3 : i1) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %60 = "test.pureop"(%arg0_8, %arg1_5) ({
// CHECK-NEXT:      ^13(%arg4 : f32, %arg5 : f32):
// CHECK-NEXT:        %61 = arith.divf %arg4, %arg5 : f32
// CHECK-NEXT:        %62 = "arith.remf"(%arg4, %arg2_5) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:        %63 = arith.select %arg3_3, %61, %62 : f32
// CHECK-NEXT:        "test.region_yield"(%63) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %60, %60 : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK-NEXT:    }

// Operation with non-identical regions dont CSE.
func.func @no_cse_single_block_ops_different_bodies(%arg0_9 : tensor<?x?xf32>, %arg1_6 : tensor<?x?xf32>, %arg2_7 : f32, %arg3_4 : i1) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    %79 = "test.pureop"(%arg0_9, %arg1_6) ({
    ^16(%arg4_2 : f32, %arg5_2 : f32):
      %80 = arith.divf %arg4_2, %arg5_2 : f32
      %81 = "arith.remf"(%arg4_2, %arg2_7) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
      %82 = arith.select %arg3_4, %80, %81 : f32
      "test.region_yield"(%82) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %83 = "test.pureop"(%arg0_9, %arg1_6) ({
    ^17(%arg4_3 : f32, %arg5_3 : f32):
      %84 = arith.divf %arg4_3, %arg5_3 : f32
      %85 = "arith.remf"(%arg4_3, %arg2_7) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
      %86 = arith.select %arg3_4, %85, %84 : f32
      "test.region_yield"(%86) : (f32) -> ()
    }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    func.return %79, %83 : tensor<?x?xf32>, tensor<?x?xf32>
  }

// CHECK:         func.func @no_cse_single_block_ops_different_bodies(%arg0_9 : tensor<?x?xf32>, %arg1_6 : tensor<?x?xf32>, %arg2_6 : f32, %arg3_4 : i1) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %64 = "test.pureop"(%arg0_9, %arg1_6) ({
// CHECK-NEXT:      ^14(%arg4_1 : f32, %arg5_1 : f32):
// CHECK-NEXT:        %65 = arith.divf %arg4_1, %arg5_1 : f32
// CHECK-NEXT:        %66 = "arith.remf"(%arg4_1, %arg2_6) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:        %67 = arith.select %arg3_4, %65, %66 : f32
// CHECK-NEXT:        "test.region_yield"(%67) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      %68 = "test.pureop"(%arg0_9, %arg1_6) ({
// CHECK-NEXT:      ^15(%arg4_2 : f32, %arg5_2 : f32):
// CHECK-NEXT:        %69 = arith.divf %arg4_2, %arg5_2 : f32
// CHECK-NEXT:        %70 = "arith.remf"(%arg4_2, %arg2_6) <{"fastmath" = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:        %71 = arith.select %arg3_4, %70, %69 : f32
// CHECK-NEXT:        "test.region_yield"(%71) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %64, %68 : tensor<?x?xf32>, tensor<?x?xf32>
// CHECK-NEXT:    }

func.func @failing_issue_59135(%arg0_10 : tensor<2x2xi1>, %arg1_7 : f32, %arg2_8 : tensor<2xi1>) -> (tensor<2xi1>, tensor<2xi1>) {
    %87 = arith.constant false
    %88 = arith.constant true
    %89 = "test.pureop"(%arg2_8) ({
    ^18(%arg3_5 : i1):
      %90 = arith.constant true
      "test.region_yield"(%90) : (i1) -> ()
    }) : (tensor<2xi1>) -> tensor<2xi1>
    %91 = "test.pureop"(%arg2_8) ({
    ^19(%arg3_6 : i1):
      %92 = arith.constant true
      "test.region_yield"(%92) : (i1) -> ()
    }) : (tensor<2xi1>) -> tensor<2xi1>
    %93 = arith.maxsi %87, %88 : i1
    func.return %89, %91 : tensor<2xi1>, tensor<2xi1>
  }

// CHECK-NEXT:    func.func @failing_issue_59135(%arg0_10 : tensor<2x2xi1>, %arg1_7 : f32, %arg2_7 : tensor<2xi1>) -> (tensor<2xi1>, tensor<2xi1>) {
// CHECK-NEXT:      %72 = arith.constant false
// CHECK-NEXT:      %73 = arith.constant true
// CHECK-NEXT:      %74 = "test.pureop"(%arg2_7) ({
// CHECK-NEXT:      ^16(%arg3_5 : i1):
// CHECK-NEXT:        "test.region_yield"(%73) : (i1) -> ()
// CHECK-NEXT:      }) : (tensor<2xi1>) -> tensor<2xi1>
// CHECK-NEXT:      %75 = arith.maxsi %72, %73 : i1
// CHECK-NEXT:      func.return %74, %74 : tensor<2xi1>, tensor<2xi1>
// CHECK-NEXT:    }

func.func @cse_multiple_regions(%arg0_11 : i1, %arg1_8 : tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
    %94 = "scf.if"(%arg0_11) ({
      %95 = tensor.empty() : tensor<5xf32>
      scf.yield %95 : tensor<5xf32>
    }, {
      scf.yield %arg1_8 : tensor<5xf32>
    }) : (i1) -> tensor<5xf32>
    %96 = "scf.if"(%arg0_11) ({
      %97 = tensor.empty() : tensor<5xf32>
      scf.yield %97 : tensor<5xf32>
    }, {
      scf.yield %arg1_8 : tensor<5xf32>
    }) : (i1) -> tensor<5xf32>
    func.return %94, %96 : tensor<5xf32>, tensor<5xf32>
  }

// CHECK:         func.func @cse_multiple_regions(%arg0_11 : i1, %arg1_8 : tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// CHECK-NEXT:      %76 = "scf.if"(%arg0_11) ({
// CHECK-NEXT:        %77 = tensor.empty() : tensor<5xf32>
// CHECK-NEXT:        scf.yield %77 : tensor<5xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:        scf.yield %arg1_8 : tensor<5xf32>
// CHECK-NEXT:      }) : (i1) -> tensor<5xf32>
// CHECK-NEXT:      func.return %76, %76 : tensor<5xf32>, tensor<5xf32>
// CHECK-NEXT:    }

// xDSL doesn't have the notion of sideffects.
 func.func @cse_recursive_effects_success() -> (i32, i32, i32) {
    %98 = "test.op_with_memread"() : () -> i32
    %99 = arith.constant true
    %100 = "scf.if"(%99) ({
      %101 = arith.constant 42 : i32
      scf.yield %101 : i32
    }, {
      %102 = arith.constant 24 : i32
      scf.yield %102 : i32
    }) : (i1) -> i32
    %103 = "test.op_with_memread"() : () -> i32
    func.return %98, %103, %100 : i32, i32, i32
  }

// CHECK:         func.func @cse_recursive_effects_success() -> (i32, i32, i32) {
// CHECK-NEXT:      %78 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      %79 = arith.constant true
// CHECK-NEXT:      %80 = "scf.if"(%79) ({
// CHECK-NEXT:        %81 = arith.constant 42 : i32
// CHECK-NEXT:        scf.yield %81 : i32
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %82 = arith.constant 24 : i32
// CHECK-NEXT:        scf.yield %82 : i32
// CHECK-NEXT:      }) : (i1) -> i32
// CHECK-NEXT:      %83 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      func.return %78, %83, %80 : i32, i32, i32
// CHECK-NEXT:    }

// xDSL doesn't have the notion of sideffects.
func.func @cse_recursive_effects_failure() -> (i32, i32, i32) {
    %104 = "test.op_with_memread"() : () -> i32
    %105 = arith.constant true
    %106 = "scf.if"(%105) ({
      "test.op_with_memwrite"() : () -> ()
      %107 = arith.constant 42 : i32
      scf.yield %107 : i32
    }, {
      %108 = arith.constant 24 : i32
      scf.yield %108 : i32
    }) : (i1) -> i32
    %109 = "test.op_with_memread"() : () -> i32
    func.return %104, %109, %106 : i32, i32, i32
  }

// CHECK:         func.func @cse_recursive_effects_failure() -> (i32, i32, i32) {
// CHECK-NEXT:      %84 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      %85 = arith.constant true
// CHECK-NEXT:      %86 = "scf.if"(%85) ({
// CHECK-NEXT:        "test.op_with_memwrite"() : () -> ()
// CHECK-NEXT:        %87 = arith.constant 42 : i32
// CHECK-NEXT:        scf.yield %87 : i32
// CHECK-NEXT:      }, {
// CHECK-NEXT:        %88 = arith.constant 24 : i32
// CHECK-NEXT:        scf.yield %88 : i32
// CHECK-NEXT:      }) : (i1) -> i32
// CHECK-NEXT:      %89 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      func.return %84, %89, %86 : i32, i32, i32
// CHECK-NEXT:    }

// Check that no CSE happens on a recursively side-effecting ops containing side-effects.
func.func @no_cse_multiple_regions_side_effect(%arg0_12 : i1, %arg1_9 : memref<5xf32>) -> (memref<5xf32>, memref<5xf32>) {
    %90 = "scf.if"(%arg0_12) ({
      %91 = memref.alloc() : memref<5xf32>
      scf.yield %91 : memref<5xf32>
    }, {
      scf.yield %arg1_9 : memref<5xf32>
    }) : (i1) -> memref<5xf32>
    %92 = "scf.if"(%arg0_12) ({
      %93 = memref.alloc() : memref<5xf32>
      scf.yield %93 : memref<5xf32>
    }, {
      scf.yield %arg1_9 : memref<5xf32>
    }) : (i1) -> memref<5xf32>
    func.return %90, %92 : memref<5xf32>, memref<5xf32>
}

// CHECK:          func.func @no_cse_multiple_regions_side_effect(%arg0_12 : i1, %arg1_9 : memref<5xf32>) -> (memref<5xf32>, memref<5xf32>) {
// CHECK-NEXT:       %90 = "scf.if"(%arg0_12) ({
// CHECK-NEXT:         %91 = memref.alloc() : memref<5xf32>
// CHECK-NEXT:         scf.yield %91 : memref<5xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:         scf.yield %arg1_9 : memref<5xf32>
// CHECK-NEXT:       }) : (i1) -> memref<5xf32>
// CHECK-NEXT:       %92 = "scf.if"(%arg0_12) ({
// CHECK-NEXT:         %93 = memref.alloc() : memref<5xf32>
// CHECK-NEXT:         scf.yield %93 : memref<5xf32>
// CHECK-NEXT:       }, {
// CHECK-NEXT:         scf.yield %arg1_9 : memref<5xf32>
// CHECK-NEXT:       }) : (i1) -> memref<5xf32>
// CHECK-NEXT:       func.return %90, %92 : memref<5xf32>, memref<5xf32>
// CHECK-NEXT:     }
