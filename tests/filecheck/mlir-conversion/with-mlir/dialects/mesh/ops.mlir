// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

mesh.mesh @mesh0(shape = 2x2x4)
mesh.mesh @mesh1(shape = 4x?)
mesh.mesh @mesh2(shape = ?x4)
mesh.mesh @mesh3(shape = ?x?)
mesh.mesh @mesh4(shape = 3)
mesh.mesh @mesh5(shape = ?)

// CHECK:      mesh.mesh @mesh0(shape = 2x2x4)
// CHECK-NEXT: mesh.mesh @mesh1(shape = 4x?)
// CHECK-NEXT: mesh.mesh @mesh2(shape = ?x4)
// CHECK-NEXT: mesh.mesh @mesh3(shape = ?x?)
// CHECK-NEXT: mesh.mesh @mesh4(shape = 3)
// CHECK-NEXT: mesh.mesh @mesh5(shape = ?)
