// RUN: xdsl-opt --allow-unregistered-dialect %s -p cse | filecheck %s

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

func.func @simple_float_constant() -> (f32, f32) {
    %0 = arith.constant 1.0 : f32
    %1 = arith.constant 1.0 : f32
    func.return %0, %1 : f32, f32
}

// CHECK:         func.func @simple_float_constant() -> (f32, f32) {
// CHECK-NEXT:      %0 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:      func.return %0, %0 : f32, f32
// CHECK-NEXT:    }

// CHECK-LABEL: @basic
  func.func @basic() -> (index, index) {
    %2 = arith.constant 0 : index
    %3 = arith.constant 0 : index
    %4 = affine.apply affine_map<(d0) -> ((d0 mod 2))>(%2)
    %5 = affine.apply affine_map<(d0) -> ((d0 mod 2))>(%3)
    func.return %4, %5 : index, index
  }

// CHECK:         %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = affine.apply affine_map<(d0) -> ((d0 mod 2))> (%0)
// CHECK-NEXT:      func.return %1, %1 : index, index
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

// CHECK:      %0 = arith.addf %arg0, %arg1 : f32
// CHECK-NEXT:      %1 = arith.addf %0, %0 : f32
// CHECK-NEXT:      %2 = arith.addf %1, %1 : f32
// CHECK-NEXT:      %3 = arith.addf %2, %2 : f32
// CHECK-NEXT:      func.return %3 : f32
// CHECK-NEXT:    }

/// Check that operations are not eliminated if they have different operands.
// CHECK-LABEL: @different_ops
func.func @different_ops() -> (i32, i32) {
    %16 = arith.constant 0 : i32
    %17 = arith.constant 1 : i32
    func.return %16, %17 : i32, i32
  }

// CHECK:      %0 = arith.constant 0 : i32
// CHECK-NEXT:      %1 = arith.constant 1 : i32
// CHECK-NEXT:      func.return %0, %1 : i32, i32
// CHECK-NEXT:    }

/// Check that operations are not eliminated if they have different result
/// types.
// CHECK-LABEL: @different_results
  func.func @different_results(%arg0_1 : memref<*xf32>) -> (memref<?x?xf32>, memref<4x?xf32>) {
    %18 = "memref.cast"(%arg0_1) : (memref<*xf32>) -> memref<?x?xf32>
    %19 = "memref.cast"(%arg0_1) : (memref<*xf32>) -> memref<4x?xf32>
    func.return %18, %19 : memref<?x?xf32>, memref<4x?xf32>
  }
// CHECK:      %0 = "memref.cast"(%arg0) : (memref<*xf32>) -> memref<?x?xf32>
// CHECK-NEXT:      %1 = "memref.cast"(%arg0) : (memref<*xf32>) -> memref<4x?xf32>
// CHECK-NEXT:      func.return %0, %1 : memref<?x?xf32>, memref<4x?xf32>
// CHECK-NEXT:    }

/// Check that operations are not eliminated if they have different attributes.
// CHECK-LABEL: @different_attributes
  func.func @different_attributes(%arg0_2 : index, %arg1_1 : index) -> (i1, i1, i1) {
    %20 = arith.cmpi slt, %arg0_2, %arg1_1 : index
    %21 = arith.cmpi ne, %arg0_2, %arg1_1 : index
    %22 = arith.cmpi ne, %arg0_2, %arg1_1 : index
    func.return %20, %21, %22 : i1, i1, i1
  }

// CHECK:      %0 = arith.cmpi slt, %arg0, %arg1 : index
// CHECK-NEXT:      %1 = arith.cmpi ne, %arg0, %arg1 : index
// CHECK-NEXT:      func.return %0, %1, %1 : i1, i1, i1
// CHECK-NEXT:    }

/// Check that operations with side effects are not eliminated.
// CHECK-LABEL: @side_effect
  func.func @side_effect() -> (memref<2x1xf32>, memref<2x1xf32>) {
    %23 = memref.alloc() : memref<2x1xf32>
    %24 = memref.alloc() : memref<2x1xf32>
    func.return %23, %24 : memref<2x1xf32>, memref<2x1xf32>
  }
// CHECK:      %0 = memref.alloc() : memref<2x1xf32>
// CHECK-NEXT:      %1 = memref.alloc() : memref<2x1xf32>
// CHECK-NEXT:      func.return %0, %1 : memref<2x1xf32>, memref<2x1xf32>
// CHECK-NEXT:    }

/// Check that operation definitions are properly propagated down the dominance
/// tree.
// CHECK-LABEL: @down_propagate_for
  func.func @down_propagate_for() {
    %25 = arith.constant 1 : i32
    "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, "step" = 1 : index, "upperBoundMap" = affine_map<() -> (4)>}> ({
    ^0(%arg0_3 : index):
      %26 = arith.constant 1 : i32
      "foo"(%25, %26) : (i32, i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    func.return
  }

// CHECK:      %0 = arith.constant 1 : i32
// CHECK-NEXT:      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
// CHECK-NEXT:      ^0(%arg0 : index):
// CHECK-NEXT:        "foo"(%0, %0) : (i32, i32) -> ()
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// This would be checking that the constant in the second block is cse'd with the first one
// MLIR has the notion of SSACFG regions (those) and graph regions.
// This works on SSACFG regions only - at least in MLIR implementation.
// We do not have this Region Kind disctinction; so everything here works on the pessimistic
// Graph Rewgion assumption.

// CHECK-LABEL: @down_propagate()
func.func @down_propagate() -> i32 {
    %27 = arith.constant 1 : i32
    %28 = arith.constant true
    cf.cond_br %28, ^1, ^2(%27 : i32)
  ^1:
    %29 = arith.constant 1 : i32
    cf.br ^2(%29 : i32)
  ^2(%30 : i32):
    func.return %30 : i32
  }

// CHECK:      %0 = arith.constant 1 : i32
// CHECK-NEXT:      %1 = arith.constant true
// CHECK-NEXT:      cf.cond_br %1, ^0, ^1(%0 : i32)
// CHECK-NEXT:    ^0:
// CHECK-NEXT:      %2 = arith.constant 1 : i32
// CHECK-NEXT:      cf.br ^1(%2 : i32)
// CHECK-NEXT:    ^1(%3 : i32):
// CHECK-NEXT:      func.return %3 : i32
// CHECK-NEXT:    }

/// Check that operation definitions are NOT propagated up the dominance tree.
// CHECK-LABEL: @up_propagate_for
 func.func @up_propagate_for() -> i32 {
    "affine.for"() <{"lowerBoundMap" = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, "step" = 1 : index, "upperBoundMap" = affine_map<() -> (4)>}> ({
    ^3(%arg0_4 : index):
      %31 = arith.constant 1 : i32
      "foo"(%31) : (i32) -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    %32 = arith.constant 1 : i32
    func.return %32 : i32
  }

// CHECK:      "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap = affine_map<() -> (4)>}> ({
// CHECK-NEXT:      ^0(%arg0 : index):
// CHECK-NEXT:        %0 = arith.constant 1 : i32
// CHECK-NEXT:        "foo"(%0) : (i32) -> ()
// CHECK-NEXT:        "affine.yield"() : () -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      %1 = arith.constant 1 : i32
// CHECK-NEXT:      func.return %1 : i32
// CHECK-NEXT:    }

// CHECK-LABEL: func @up_propagate()
func.func @up_propagate() -> i32 {
    %33 = arith.constant 0 : i32
    %34 = arith.constant true
    cf.cond_br %34, ^4, ^5(%33 : i32)
  ^4:
    %35 = arith.constant 1 : i32
    cf.br ^5(%35 : i32)
  ^5(%36 : i32):
    %37 = arith.constant 1 : i32
    %38 = arith.addi %36, %37 : i32
    func.return %38 : i32
  }

// CHECK:      %0 = arith.constant 0 : i32
// CHECK-NEXT:      %1 = arith.constant true
// CHECK-NEXT:      cf.cond_br %1, ^0, ^1(%0 : i32)
// CHECK-NEXT:    ^0:
// CHECK-NEXT:      %2 = arith.constant 1 : i32
// CHECK-NEXT:      cf.br ^1(%2 : i32)
// CHECK-NEXT:    ^1(%3 : i32):
// CHECK-NEXT:      %4 = arith.constant 1 : i32
// CHECK-NEXT:      %5 = arith.addi %3, %4 : i32
// CHECK-NEXT:      func.return %5 : i32
// CHECK-NEXT:    }

/// The same test as above except that we are testing on a cfg embedded within
/// an operation region.
// CHECK-LABEL: func @up_propagate_region
func.func @up_propagate_region() -> i32 {
    %39 = "foo.region"() ({
      %40 = arith.constant 0 : i32
      %41 = arith.constant true
      cf.cond_br %41, ^6, ^7(%40 : i32)
    ^6:
      %42 = arith.constant 1 : i32
      cf.br ^7(%42 : i32)
    ^7(%43 : i32):
      %44 = arith.constant 1 : i32
      %45 = arith.addi %43, %44 : i32
      "foo.yield"(%45) : (i32) -> ()
    }) : () -> i32
    func.return %39 : i32
  }

// CHECK:      %0 = "foo.region"() ({
// CHECK-NEXT:        %1 = arith.constant 0 : i32
// CHECK-NEXT:        %2 = arith.constant true
// CHECK-NEXT:        cf.cond_br %2, ^0, ^1(%1 : i32)
// CHECK-NEXT:      ^0:
// CHECK-NEXT:        %3 = arith.constant 1 : i32
// CHECK-NEXT:        cf.br ^1(%3 : i32)
// CHECK-NEXT:      ^1(%4 : i32):
// CHECK-NEXT:        %5 = arith.constant 1 : i32
// CHECK-NEXT:        %6 = arith.addi %4, %5 : i32
// CHECK-NEXT:        "foo.yield"(%6) : (i32) -> ()
// CHECK-NEXT:      }) : () -> i32
// CHECK-NEXT:      func.return %0 : i32
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

// CHECK:      %0 = arith.constant 1 : i32
// CHECK-NEXT:      func.func @nested_func() {
// CHECK-NEXT:        %1 = arith.constant 1 : i32
// CHECK-NEXT:        "foo.yield"(%1) : (i32) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      "foo.region"() ({
// CHECK-NEXT:        %1 = arith.constant 1 : i32
// CHECK-NEXT:        "foo.yield"(%1) : (i32) -> ()
// CHECK-NEXT:      }) : () -> ()
// CHECK-NEXT:      func.return %0 : i32
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

// CHECK:      "test.graph_region"() ({
// CHECK-NEXT:        %0 = arith.addi %1, %2 : i32
// CHECK-NEXT:        %1 = arith.constant 1 : i32
// CHECK-NEXT:        %2 = arith.constant 1 : i32
// CHECK-NEXT:        "foo.yield"(%0) : (i32) -> ()
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

// CHECK:         %0 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      %1 = arith.addi %0, %0 : i32
// CHECK-NEXT:      func.return %1 : i32
// CHECK-NEXT:    }


/// This test is checking that CSE is removing duplicated read op that follow
/// other.
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

// CHECK:        %0 = "test.op_with_memread"() : () -> i64
// CHECK-NEXT:      %1 = arith.addi %0, %0 : i64
// CHECK-NEXT:      %2 = arith.addi %1, %0 : i64
// CHECK-NEXT:      %3 = arith.addi %2, %0 : i64
// CHECK-NEXT:      func.return %3 : i64
// CHECK-NEXT:    }

/// This test is checking that CSE is not removing duplicated read op that
/// have write op in between.
func.func @dont_remove_duplicated_read_op_with_sideeffecting() -> i32 {
    %62 = "test.op_with_memread"() : () -> i32
    "test.op_with_memwrite"() : () -> ()
    %63 = "test.op_with_memread"() : () -> i32
    %64 = arith.addi %62, %63 : i32
    func.return %64 : i32
  }

// CHECK:      %0 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      "test.op_with_memwrite"() : () -> ()
// CHECK-NEXT:      %1 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      %2 = arith.addi %0, %1 : i32
// CHECK-NEXT:      func.return %2 : i32
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

// CHECK:         func.func @cse_single_block_ops(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %0 = "test.pureop"(%arg0, %arg1) ({
// CHECK-NEXT:      ^0(%arg2 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg2) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %0, %0 : tensor<?x?xf32>, tensor<?x?xf32>
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

// CHECK:         func.func @no_cse_varied_bbargs(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %0 = "test.pureop"(%arg0, %arg1) ({
// CHECK-NEXT:      ^0(%arg2 : f32, %arg3 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg2) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      %1 = "test.pureop"(%arg0, %arg1) ({
// CHECK-NEXT:      ^1(%arg2_1 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg2_1) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
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
// CHECK:         func.func @no_cse_region_difference_simple(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %0 = "test.pureop"(%arg0, %arg1) ({
// CHECK-NEXT:      ^0(%arg2 : f32, %arg3 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg2) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      %1 = "test.pureop"(%arg0, %arg1) ({
// CHECK-NEXT:      ^1(%arg2_1 : f32, %arg3_1 : f32):
// CHECK-NEXT:        "test.region_yield"(%arg3_1) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %0, %1 : tensor<?x?xf32>, tensor<?x?xf32>
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

// CHECK:         func.func @cse_single_block_ops_identical_bodies(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : f32, %arg3 : i1) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %0 = "test.pureop"(%arg0, %arg1) ({
// CHECK-NEXT:      ^0(%arg4 : f32, %arg5 : f32):
// CHECK-NEXT:        %1 = arith.divf %arg4, %arg5 : f32
// CHECK-NEXT:        %2 = "arith.remf"(%arg4, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:        %3 = arith.select %arg3, %1, %2 : f32
// CHECK-NEXT:        "test.region_yield"(%3) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %0, %0 : tensor<?x?xf32>, tensor<?x?xf32>
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

// CHECK:         func.func @no_cse_single_block_ops_different_bodies(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : f32, %arg3 : i1) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
// CHECK-NEXT:      %0 = "test.pureop"(%arg0, %arg1) ({
// CHECK-NEXT:      ^0(%arg4 : f32, %arg5 : f32):
// CHECK-NEXT:        %1 = arith.divf %arg4, %arg5 : f32
// CHECK-NEXT:        %2 = "arith.remf"(%arg4, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:        %3 = arith.select %arg3, %1, %2 : f32
// CHECK-NEXT:        "test.region_yield"(%3) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      %4 = "test.pureop"(%arg0, %arg1) ({
// CHECK-NEXT:      ^1(%arg4_1 : f32, %arg5_1 : f32):
// CHECK-NEXT:        %5 = arith.divf %arg4_1, %arg5_1 : f32
// CHECK-NEXT:        %6 = "arith.remf"(%arg4_1, %arg2) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
// CHECK-NEXT:        %7 = arith.select %arg3, %6, %5 : f32
// CHECK-NEXT:        "test.region_yield"(%7) : (f32) -> ()
// CHECK-NEXT:      }) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:      func.return %0, %4 : tensor<?x?xf32>, tensor<?x?xf32>
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

// CHECK:         func.func @failing_issue_59135(%arg0 : tensor<2x2xi1>, %arg1 : f32, %arg2 : tensor<2xi1>) -> (tensor<2xi1>, tensor<2xi1>) {
// CHECK-NEXT:      %0 = arith.constant false
// CHECK-NEXT:      %1 = arith.constant true
// CHECK-NEXT:      %2 = "test.pureop"(%arg2) ({
// CHECK-NEXT:      ^0(%arg3 : i1):
// CHECK-NEXT:        "test.region_yield"(%1) : (i1) -> ()
// CHECK-NEXT:      }) : (tensor<2xi1>) -> tensor<2xi1>
// CHECK-NEXT:      func.return %2, %2 : tensor<2xi1>, tensor<2xi1>
// CHECK-NEXT:    }

func.func @cse_multiple_regions(%arg0_11 : i1, %arg1_8 : tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
    %94 = scf.if %arg0_11 -> (tensor<5xf32>) {
      %95 = tensor.empty() : tensor<5xf32>
      scf.yield %95 : tensor<5xf32>
    } else {
      scf.yield %arg1_8 : tensor<5xf32>
    }
    %96 = scf.if %arg0_11 -> (tensor<5xf32>) {
      %97 = tensor.empty() : tensor<5xf32>
      scf.yield %97 : tensor<5xf32>
    } else {
      scf.yield %arg1_8 : tensor<5xf32>
    }
    func.return %94, %96 : tensor<5xf32>, tensor<5xf32>
  }

// CHECK:         func.func @cse_multiple_regions(%arg0 : i1, %arg1 : tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// CHECK-NEXT:      %0 = scf.if %arg0 -> (tensor<5xf32>) {
// CHECK-NEXT:        %1 = tensor.empty() : tensor<5xf32>
// CHECK-NEXT:        scf.yield %1 : tensor<5xf32>
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %arg1 : tensor<5xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %0, %0 : tensor<5xf32>, tensor<5xf32>
// CHECK-NEXT:    }

// Check that no CSE happens on a recursively side-effecting ops containing side-effects.
func.func @no_cse_multiple_regions_side_effect(%arg0_12 : i1, %arg1_9 : memref<5xf32>) -> (memref<5xf32>, memref<5xf32>) {
    %90 = scf.if %arg0_12 -> (memref<5xf32>) {
      %91 = memref.alloc() : memref<5xf32>
      scf.yield %91 : memref<5xf32>
    } else {
      scf.yield %arg1_9 : memref<5xf32>
    }
    %92 = scf.if %arg0_12 -> (memref<5xf32>) {
      %93 = memref.alloc() : memref<5xf32>
      scf.yield %93 : memref<5xf32>
    } else {
      scf.yield %arg1_9 : memref<5xf32>
    }
    func.return %90, %92 : memref<5xf32>, memref<5xf32>
}

// CHECK:         func.func @no_cse_multiple_regions_side_effect(%arg0 : i1, %arg1 : memref<5xf32>) -> (memref<5xf32>, memref<5xf32>) {
// CHECK-NEXT:      %0 = scf.if %arg0 -> (memref<5xf32>) {
// CHECK-NEXT:        %1 = memref.alloc() : memref<5xf32>
// CHECK-NEXT:        scf.yield %1 : memref<5xf32>
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %arg1 : memref<5xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %2 = scf.if %arg0 -> (memref<5xf32>) {
// CHECK-NEXT:        %3 = memref.alloc() : memref<5xf32>
// CHECK-NEXT:        scf.yield %3 : memref<5xf32>
// CHECK-NEXT:      } else {
// CHECK-NEXT:        scf.yield %arg1 : memref<5xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %0, %2 : memref<5xf32>, memref<5xf32>
// CHECK-NEXT:    }

 func.func @cse_recursive_effects_success() -> (i32, i32, i32) {
    %98 = "test.op_with_memread"() : () -> i32
    %99 = arith.constant true
    %100 = scf.if %99 -> (i32) {
      %101 = arith.constant 42 : i32
      scf.yield %101 : i32
    } else {
      %102 = arith.constant 24 : i32
      scf.yield %102 : i32
    }
    %103 = "test.op_with_memread"() : () -> i32
    func.return %98, %103, %100 : i32, i32, i32
  }

// CHECK:         func.func @cse_recursive_effects_success() -> (i32, i32, i32) {
// CHECK-NEXT:      %0 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      %1 = arith.constant true
// CHECK-NEXT:      %2 = scf.if %1 -> (i32) {
// CHECK-NEXT:        %3 = arith.constant 42 : i32
// CHECK-NEXT:        scf.yield %3 : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %4 = arith.constant 24 : i32
// CHECK-NEXT:        scf.yield %4 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %0, %0, %2 : i32, i32, i32
// CHECK-NEXT:    }

// xDSL doesn't have the notion of sideffects.
func.func @cse_recursive_effects_failure() -> (i32, i32, i32) {
    %104 = "test.op_with_memread"() : () -> i32
    %105 = arith.constant true
    %106 = scf.if %105 -> (i32) {
      "test.op_with_memwrite"() : () -> ()
      %107 = arith.constant 42 : i32
      scf.yield %107 : i32
    } else {
      %108 = arith.constant 24 : i32
      scf.yield %108 : i32
    }
    %109 = "test.op_with_memread"() : () -> i32
    func.return %104, %109, %106 : i32, i32, i32
  }

// CHECK:         func.func @cse_recursive_effects_failure() -> (i32, i32, i32) {
// CHECK-NEXT:      %0 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      %1 = arith.constant true
// CHECK-NEXT:      %2 = scf.if %1 -> (i32) {
// CHECK-NEXT:        "test.op_with_memwrite"() : () -> ()
// CHECK-NEXT:        %3 = arith.constant 42 : i32
// CHECK-NEXT:        scf.yield %3 : i32
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %4 = arith.constant 24 : i32
// CHECK-NEXT:        scf.yield %4 : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      %5 = "test.op_with_memread"() : () -> i32
// CHECK-NEXT:      func.return %0, %5, %2 : i32, i32, i32
// CHECK-NEXT:    }
