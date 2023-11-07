// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

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
memref.store %v0, %m[%i0, %i1] : memref<2x3xi32>
memref.store %v0, %m[%i0, %i1] {"nontemporal" = false} : memref<2x3xi32>
memref.store %v0, %m[%i0, %i1] {"nontemporal" = true} : memref<2x3xi32>
%v1 = memref.load %m[%i0, %i1] : memref<2x3xi32>
%v2 = memref.load %m[%i0, %i1] {"nontemporal" = false} : memref<2x3xi32>
%v3 = memref.load %m[%i0, %i1] {"nontemporal" = true} : memref<2x3xi32>

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
// CHECK-NEXT:   memref.store %0, %3[%1, %2] : memref<2x3xi32>
// CHECK-NEXT:   memref.store %0, %3[%1, %2] : memref<2x3xi32>
// CHECK-NEXT:   memref.store %0, %3[%1, %2] {nontemporal = true} : memref<2x3xi32>
// CHECK-NEXT:   %{{.*}} = memref.load %3[%1, %2] : memref<2x3xi32>
// CHECK-NEXT:   %{{.*}} = memref.load %3[%1, %2] : memref<2x3xi32>
// CHECK-NEXT:   %{{.*}} = memref.load %3[%1, %2] {nontemporal = true} : memref<2x3xi32>
// CHECK-NEXT: }
