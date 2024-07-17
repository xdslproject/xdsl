// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect --mlir-print-local-scope | filecheck %s

func.func @memref_alloca_scope() {
"memref.alloca_scope"() ({
    "memref.alloca_scope.return"() : () -> ()
}) : () -> ()
func.return
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

// CHECK:       module {
// CHECK-NEXT:    func.func @memref_alloca_scope() {
// CHECK-NEXT:      memref.alloca_scope  {
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
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

// CHECK-NEXT: }
