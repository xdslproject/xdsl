// RUN: xdsl-opt %s -p dce | filecheck %s

"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  %1 = "test.op"() : () -> i32

  %a = "arith.addi"(%0, %1) : (i32, i32) -> i32
  %b = "arith.addi"(%0, %a) : (i32, i32) -> i32
  %c = "arith.addi"(%0, %b) : (i32, i32) -> i32
  %d = "arith.addi"(%0, %c) : (i32, i32) -> i32
  %e = "arith.addi"(%0, %d) : (i32, i32) -> i32

  "test.op"(%0) : (i32) -> ()

  // CHECK:       %0 = "test.op"() : () -> i32
  // CHECK-NEXT:  %1 = "test.op"() : () -> i32
  // CHECK-NOT: addi
  // CHECK-NEXT:  "test.op"(%0) : (i32) -> ()

  %10 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<memref<?xf32>>
  %11 = memref.load %10[] : memref<memref<?xf32>>
  %12 = arith.constant 1 : index
  %13 = "memref.dim"(%11, %12) : (memref<?xf32>, index) -> index

  // CHECK:       %2 = memref.alloc() : memref<memref<?xf32>>
  // CHECK-NOT: memref.load
  // CHECK-NOT: arith.constant
  // CHECK-NOT: memref.dim

  %20 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<memref<?xf32>>
  %21 = memref.load %20[] : memref<memref<?xf32>>
  %22 = arith.constant 0 : index
  %23 = arith.constant 9.1 : f32
  memref.store %23, %21[%22] : memref<?xf32>

  // CHECK:       %3 = memref.alloc() : memref<memref<?xf32>>
  // CHECK-NEXT:  %4 = memref.load %3[] : memref<memref<?xf32>>
  // CHECK-NEXT:  %5 = arith.constant 0 : index
  // CHECK-NEXT:  %6 = arith.constant 9.100000e+00 : f32
  // CHECK-NEXT:  memref.store %6, %4[%5] : memref<?xf32>

}) : () -> ()
