// RUN: xdsl-opt %s | xdsl-opt | mlir-opt | filecheck %s

func.func @while() {
    %init = arith.constant 0 : i32
    %res = scf.while (%arg = %init) : (i32) -> i32 {
        %zero = arith.constant 0 : i32
        %c = "arith.cmpi"(%zero, %arg) {"predicate" = 1 : i64} : (i32, i32) -> i1
        scf.condition(%c) %zero : i32
    } do {
    ^1(%arg2: i32):
        scf.yield %arg2 : i32
    }
    return
}

// CHECK:      func.func @while() {
// CHECK-NEXT:   %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:   %{{.*}} = scf.while (%{{.*}} = %{{.*}}) : (i32) -> i32 {
// CHECK-NEXT:     %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:     %{{.*}} = arith.cmpi ne, %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:     scf.condition(%{{.*}}) %{{.*}} : i32
// CHECK-NEXT:   } do {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: i32):
// CHECK-NEXT:     scf.yield %{{.*}} : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
