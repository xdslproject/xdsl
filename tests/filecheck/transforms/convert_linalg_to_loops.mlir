
%A, %B, %C = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)

linalg.generic {
    indexing_maps = [
        affine_map<() -> ()>,
        affine_map<() -> ()>,
        affine_map<() -> ()>
    ],
    iterator_types = []
} ins(%A, %B : memref<f64>, memref<f64>) outs(%C : memref<f64>) {
^0(%a : f64, %b : f64, %acc_old : f64):
    %prod = arith.mulf %a, %b : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (memref<f64>, memref<f64>, memref<f64>)
// CHECK-NEXT:    %%{{.*}} = memref.load %0[] : memref<f64>
// CHECK-NEXT:    %%{{.*}} = memref.load %0[] : memref<f64>
// CHECK-NEXT:    %%{{.*}} = memref.load %0[] : memref<f64>
// CHECK-NEXT:    %%{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    %%{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    memref.store %{{.*}}, %{{.*}}[] : memref<f64>
// CHECK-NEXT:  }
