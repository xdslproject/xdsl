// RUN: xdsl-opt %s | xdsl-opt | $XDSL_MLIR_OPT --allow-unregistered-dialect --mlir-print-local-scope | filecheck %s

func.func @memref_alloca_scope() -> (i32, f32) {
  memref.alloca_scope {
  }
  memref.alloca_scope {
    "test.op"() : () -> ()
  } {foo = 1 : i32}
  %a, %b = memref.alloca_scope -> (i32, f32) {
    %x, %y = "test.op"() : () -> (i32, f32)
    memref.alloca_scope.return %x, %y : i32, f32
  }
  func.return %a, %b : i32, f32
}

%v0 = "test.op"() : () -> (i32)
%i0 = "test.op"() : () -> (index)
%i1 = "test.op"() : () -> (index)
%m = "test.op"() : () -> (memref<2x3xi32>)
%r = "test.op"() : () -> (memref<10x3xi32>)
memref.store %v0, %m[%i0, %i1] : memref<2x3xi32>
memref.store %v0, %m[%i0, %i1] {"nontemporal" = false} : memref<2x3xi32>
memref.store %v0, %m[%i0, %i1] {"nontemporal" = true} : memref<2x3xi32>
%v1 = memref.load %m[%i0, %i1] : memref<2x3xi32>
%v2 = memref.load %m[%i0, %i1] {"nontemporal" = false} : memref<2x3xi32>
%v3 = memref.load %m[%i0, %i1] {"nontemporal" = true} : memref<2x3xi32>
%r1 = memref.expand_shape %r [[0, 1], [2]] output_shape [5, 2, 3] : memref<10x3xi32> into memref<5x2x3xi32>
%r2 = memref.collapse_shape %r [[0, 1]] : memref<10x3xi32> into memref<30xi32>
%a1 = memref.alloc() : memref<2x3xf32>
%a2 = memref.alloc()[%i1] {alignment = 8}: memref<2x3xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>, 1>
memref.dealloc %a1 : memref<2x3xf32>
%s0 = memref.subview %r[0, 0] [1, 3] [1, 1] : memref<10x3xi32> to memref<3xi32>
%s2 = memref.subview %r[%i0, 0] [1, 3] [1, 1] : memref<10x3xi32> to memref<3xi32, strided<[1], offset: ?>>

%dm = "test.op"() : () -> (memref<?x?xi32>)
%di = "test.op"() : () -> (index)
%dv = memref.dim %dm, %di : memref<?x?xi32>

// CHECK:       module {
// CHECK-NEXT:    func.func @memref_alloca_scope() -> (i32, f32) {
// CHECK-NEXT:    memref.alloca_scope  {
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.alloca_scope  {
// CHECK-NEXT:      "test.op"() : () -> ()
// CHECK-NEXT:    } {foo = 1 : i32}
// CHECK-NEXT:    %10:2 = memref.alloca_scope  -> (i32, f32) {
// CHECK-NEXT:      %11:2 = "test.op"() : () -> (i32, f32)
// CHECK-NEXT:      memref.alloca_scope.return %11#0, %11#1 : i32, f32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %10#0, %10#1 : i32, f32
// CHECK-NEXT:  }
// CHECK-NEXT:   %0 = "test.op"() : () -> i32
// CHECK-NEXT:   %1 = "test.op"() : () -> index
// CHECK-NEXT:   %2 = "test.op"() : () -> index
// CHECK-NEXT:   %3 = "test.op"() : () -> memref<2x3xi32>
// CHECK-NEXT:   %4 = "test.op"() : () -> memref<10x3xi32>
// CHECK-NEXT:   memref.store %0, %3[%1, %2] : memref<2x3xi32>
// CHECK-NEXT:   memref.store %0, %3[%1, %2] : memref<2x3xi32>
// CHECK-NEXT:   memref.store %0, %3[%1, %2] {nontemporal = true} : memref<2x3xi32>
// CHECK-NEXT:   %{{.*}} = memref.load %3[%1, %2] : memref<2x3xi32>
// CHECK-NEXT:   %{{.*}} = memref.load %3[%1, %2] : memref<2x3xi32>
// CHECK-NEXT:   %{{.*}} = memref.load %3[%1, %2] {nontemporal = true} : memref<2x3xi32>
// CHECK-NEXT:   %{{.*}} = memref.expand_shape %4
// CHECK-SAME{LITERAL}: [[0, 1], [2]] output_shape [5, 2, 3] : memref<10x3xi32> into memref<5x2x3xi32>
// CHECK-NEXT:   %{{.*}} = memref.collapse_shape %4
// CHECK-SAME{LITERAL}: [[0, 1]] : memref<10x3xi32> into memref<30xi32>
// CHECK-NEXT:   %{{.*}} = memref.alloc() : memref<2x3xf32>
// CHECK-NEXT:   %{{.*}} = memref.alloc()[%{{.*}}] {alignment = 8 : i64} : memref<2x3xf32, affine_map<(d0, d1)[s0] -> (d0 + s0, d1)>, 1>
// CHECK-NEXT:   memref.dealloc %{{.*}} : memref<2x3xf32>
// CHECK-NEXT:   %{{.*}} = memref.subview %{{.*}}[0, 0] [1, 3] [1, 1] : memref<10x3xi32> to memref<3xi32>
// CHECK-NEXT:   %{{.*}} = memref.subview %{{.*}}[%{{.*}}, 0] [1, 3] [1, 1] : memref<10x3xi32> to memref<3xi32, strided<[1], offset: ?>>
// CHECK-NEXT:   %{{.*}} = "test.op"() : () -> memref<?x?xi32>
// CHECK-NEXT:   %{{.*}} = "test.op"() : () -> index
// CHECK-NEXT:   %{{.*}} = memref.dim %{{.*}}, %{{.*}} : memref<?x?xi32>

// CHECK-NEXT: }
