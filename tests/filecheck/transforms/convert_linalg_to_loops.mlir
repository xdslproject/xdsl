// RUN: xdsl-opt -p convert-linalg-to-loops %s | filecheck %s

%A, %B, %C = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)
%D, %E, %F = "test.op"() : () -> (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>)
%G, %H, %I = "test.op"() : () -> (memref<4xf64>, memref<2xf64>, memref<3xf64>)
%J = "test.op"() : () -> memref<3x2xf64>

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<2x3xf64>, memref<3x4xf64>, memref<2x4xf64>)
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<4xf64>, memref<2xf64>, memref<3xf64>)
// CHECK-NEXT:    %{{.*}} = "test.op"() : () -> memref<3x2xf64>

linalg.generic {
    indexing_maps = [
        affine_map<() -> ()>,
        affine_map<() -> ()>,
        affine_map<() -> ()>
    ],
    iterator_types = []
} ins(%A, %B : memref<f64>, memref<f64>) outs(%C : memref<f64>) {
^bb0(%a: f64, %b: f64, %acc_old: f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}
// CHECK-NEXT:    %{{.*}} = memref.load %{{.*}}[] : memref<f64>
// CHECK-NEXT:    %{{.*}} = memref.load %{{.*}}[] : memref<f64>
// CHECK-NEXT:    %{{.*}} = memref.load %{{.*}}[] : memref<f64>
// CHECK-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    memref.store %{{.*}}, %{{.*}}[] : memref<f64>


linalg.generic {
    indexing_maps = [
        affine_map<(i, j, k) -> (i, j)>,
        affine_map<(i, j, k) -> (j, k)>,
        affine_map<(i, j, k) -> (i, k)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%D, %E : memref<2x3xf64>, memref<3x4xf64>) outs(%F : memref<2x4xf64>) {
^bb0(%d: f64, %e: f64, %acc_old: f64):
    %prod = arith.mulf %d, %e : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}

// CHECK-NEXT:    %0 = arith.constant 2 : index
// CHECK-NEXT:    %1 = arith.constant 3 : index
// CHECK-NEXT:    %2 = arith.constant 4 : index
// CHECK-NEXT:    %3 = arith.constant 0 : index
// CHECK-NEXT:    %4 = arith.constant 1 : index
// CHECK-NEXT:    scf.for %5 = %3 to %0 step %4 {
// CHECK-NEXT:      scf.for %6 = %3 to %1 step %4 {
// CHECK-NEXT:        scf.for %7 = %3 to %2 step %4 {
// CHECK-NEXT:          %d = memref.load %D[%5, %6] : memref<2x3xf64>
// CHECK-NEXT:          %e = memref.load %E[%6, %7] : memref<3x4xf64>
// CHECK-NEXT:          %acc_old_1 = memref.load %F[%5, %7] : memref<2x4xf64>
// CHECK-NEXT:          %prod_1 = arith.mulf %d, %e : f64
// CHECK-NEXT:          %acc_new_1 = arith.addf %acc_old_1, %prod_1 : f64
// CHECK-NEXT:          memref.store %acc_new_1, %F[%5, %7] : memref<2x4xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }


linalg.generic {
    indexing_maps = [
        affine_map<(i, j) -> (i + j)>,
        affine_map<(i, j) -> (j)>,
        affine_map<(i, j) -> (i)>
    ],
    iterator_types = ["parallel", "reduction"]
} ins(%G, %H : memref<4xf64>, memref<2xf64>) outs(%I : memref<3xf64>) {
^bb0(%g: f64, %h: f64, %acc_old: f64):
    %prod = arith.mulf %g, %h : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}

// CHECK-NEXT:    %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = affine.apply affine_map<(d0, d1) -> ((d0 + d1))>
// CHECK-NEXT:        %{{.*}} = memref.load %G[%{{.*}}] : memref<4xf64>
// CHECK-NEXT:        %{{.*}} = memref.load %H[%{{.*}}] : memref<2xf64>
// CHECK-NEXT:        %{{.*}} = memref.load %I[%{{.*}}] : memref<3xf64>
// CHECK-NEXT:        %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        memref.store %{{.*}}, %I[%{{.*}}] : memref<3xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Scalar argument
%zero = arith.constant 0.0 : f64
linalg.generic {
    indexing_maps = [
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%zero : f64) outs(%D : memref<2x3xf64>) {
^bb0(%in: f64, %out: f64):
    linalg.yield %in : f64
}

// CHECK-NEXT:    %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<2x3xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Named op: add
linalg.add ins(%D, %D : memref<2x3xf64>, memref<2x3xf64>) outs(%D : memref<2x3xf64>)

// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = memref.load %D[%{{.*}}, %{{.*}}] : memref<2x3xf64>
// CHECK-NEXT:        %{{.*}} = memref.load %D[%{{.*}}, %{{.*}}] : memref<2x3xf64>
// CHECK-NEXT:        %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:        memref.store %{{.*}}, %D[%{{.*}}, %{{.*}}] : memref<2x3xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Named op: transpose
linalg.transpose ins(%D : memref<2x3xf64>) outs(%J : memref<3x2xf64>) permutation = [1, 0]

// CHECK-NEXT:    %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    scf.for [[I:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for [[J:%.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = memref.load %D[[[J]], [[I]]] : memref<2x3xf64>
// CHECK-NEXT:        memref.store %{{.*}}, %J[[[I]], [[J]]] : memref<3x2xf64>
// CHECK-NEXT:      }
// CHECK-NEXT:    }


// Named op: matmul
linalg.matmul ins(%D, %E : memref<2x3xf64>, memref<3x4xf64>) outs(%F : memref<2x4xf64>)

// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 4 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:    %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          %{{.*}} = memref.load %D[%{{.*}}, %{{.*}}] : memref<2x3xf64>
// CHECK-NEXT:          %{{.*}} = memref.load %E[%{{.*}}, %{{.*}}] : memref<3x4xf64>
// CHECK-NEXT:          %{{.*}} = memref.load %F[%{{.*}}, %{{.*}}] : memref<2x4xf64>
// CHECK-NEXT:          %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:          memref.store %{{.*}}, %F[%{{.*}}, %{{.*}}] : memref<2x4xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Dynamic shape: add
%LHS, %RHS, %OUT = "test.op"() : () -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
linalg.add ins(%LHS, %RHS : memref<?x?xf32>, memref<?x?xf32>) outs(%OUT : memref<?x?xf32>)

// CHECK-NEXT:    [[LHS_DYN:%.*]], [[RHS_DYN:%.*]], [[OUT_DYN:%.*]] = "test.op"() : () -> (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>)
// CHECK-NEXT:    [[DIM0_IDX:%.*]] = arith.constant 0 : index
// CHECK-NEXT:    [[DIM0:%.*]] = memref.dim [[LHS_DYN]], [[DIM0_IDX]] : memref<?x?xf32>
// CHECK-NEXT:    [[DIM1_IDX:%.*]] = arith.constant 1 : index
// CHECK-NEXT:    [[DIM1:%.*]] = memref.dim [[LHS_DYN]], [[DIM1_IDX]] : memref<?x?xf32>
// CHECK-NEXT:    [[LB:%.*]] = arith.constant 0 : index
// CHECK-NEXT:    [[STEP:%.*]] = arith.constant 1 : index
// CHECK-NEXT:    scf.for [[I:%.*]] = [[LB]] to [[DIM0]] step [[STEP]] {
// CHECK-NEXT:      scf.for [[J:%.*]] = [[LB]] to [[DIM1]] step [[STEP]] {
// CHECK-NEXT:        [[LHS_VAL:%.*]] = memref.load [[LHS_DYN]][[[I]], [[J]]] : memref<?x?xf32>
// CHECK-NEXT:        [[RHS_VAL:%.*]] = memref.load [[RHS_DYN]][[[I]], [[J]]] : memref<?x?xf32>
// CHECK-NEXT:        [[SUM:%.*]] = arith.addf [[LHS_VAL]], [[RHS_VAL]] : f32
// CHECK-NEXT:        memref.store [[SUM]], [[OUT_DYN]][[[I]], [[J]]] : memref<?x?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Index op lowering
%IDX_OUT = "test.op"() : () -> memref<2x3xindex>
linalg.generic {
    indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
} outs(%IDX_OUT : memref<2x3xindex>) {
^bb0(%out : index):
    %i = linalg.index 0 : index
    %j = linalg.index 1 : index
    %sum = arith.addi %i, %j : index
    linalg.yield %sum : index
}

// CHECK-NEXT:    [[IDX_OUT:%.*]] = "test.op"() : () -> memref<2x3xindex>
// CHECK-NEXT:    [[UB0:%.*]] = arith.constant 2 : index
// CHECK-NEXT:    [[UB1:%.*]] = arith.constant 3 : index
// CHECK-NEXT:    [[LB:%.*]] = arith.constant 0 : index
// CHECK-NEXT:    [[STEP:%.*]] = arith.constant 1 : index
// CHECK-NEXT:    scf.for [[I:%.*]] = [[LB]] to [[UB0]] step [[STEP]] {
// CHECK-NEXT:      scf.for [[J:%.*]] = [[LB]] to [[UB1]] step [[STEP]] {
// CHECK-NEXT:        [[SUM:%.*]] = arith.addi [[I]], [[J]] : index
// CHECK-NEXT:        memref.store [[SUM]], [[IDX_OUT]][[[I]], [[J]]] : memref<2x3xindex>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// CHECK-NEXT:  }
