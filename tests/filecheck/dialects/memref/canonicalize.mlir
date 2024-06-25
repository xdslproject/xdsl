// RUN: xdsl-opt -p canonicalize %s | filecheck %s

// CHECK:       builtin.module {


%alloc0 = memref.alloc() : memref<1xf64>
%alloc1 = memref.alloc() : memref<2xf64>

"test.op"(%alloc1) : (memref<2xf64>) -> ()

memref.dealloc %alloc0 : memref<1xf64>
memref.dealloc %alloc1 : memref<2xf64>
// CHECK-NEXT:    %alloc1 = memref.alloc() : memref<2xf64>
// CHECK-NEXT:    "test.op"(%alloc1) : (memref<2xf64>) -> ()
// CHECK-NEXT:    memref.dealloc %alloc1 : memref<2xf64>


// CHECK-NEXT:  }
