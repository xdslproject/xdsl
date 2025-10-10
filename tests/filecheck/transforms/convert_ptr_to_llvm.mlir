// RUN: xdsl-opt -p convert-ptr-to-llvm,reconcile-unrealized-casts --parsing-diagnostics --verify-diagnostics  %s | filecheck %s

%0 = "test.op"() : () -> !ptr_xdsl.ptr
%1 = "test.op"() : () -> index

// CHECK: %2 = "llvm.load"(%0) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
%2 = ptr_xdsl.load %0 : !ptr_xdsl.ptr -> i32

// CHECK-NEXT: "llvm.store"(%2, %0) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
ptr_xdsl.store %2, %0  : i32, !ptr_xdsl.ptr

// CHECK-NEXT: %3 = arith.index_cast %1 : index to i64
// CHECK-NEXT: %4 = "llvm.ptrtoint"(%0) : (!llvm.ptr) -> i64
// CHECK-NEXT: %5 = arith.addi %4, %3 : i64
// CHECK-NEXT: %6 = "llvm.inttoptr"(%5) : (i64) -> !llvm.ptr
%3 = ptr_xdsl.ptradd %0, %1 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr

// CHECK-NEXT: func.func private @example(!llvm.ptr) -> !llvm.ptr
func.func private @example(!ptr_xdsl.ptr) -> !ptr_xdsl.ptr

// CHECK-NEXT: %memref, %ptr = "test.op"() : () -> (memref<1xf32>, !llvm.ptr)
%memref, %ptr = "test.op"() : () -> (memref<1xf32>, !ptr_xdsl.ptr)

// CHECK-NEXT: "test.op"(%memref) : (memref<1xf32>) -> ()
%to_ptr = ptr_xdsl.to_ptr %memref : memref<1xf32> -> !ptr_xdsl.ptr
"test.op"(%to_ptr) : (!ptr_xdsl.ptr) -> ()

// CHECK-NEXT: "test.op"(%ptr) : (!llvm.ptr) -> ()
%from_ptr = ptr_xdsl.from_ptr %ptr : !ptr_xdsl.ptr -> memref<1xf32>
"test.op"(%from_ptr) : (memref<1xf32>) -> ()

// CHECK-NEXT: }
