// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:               builtin.module {
// CHECK-GENERIC:       "builtin.module"() ({

%readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)

%val = memref_stream.read from %readable : f32
memref_stream.write %val to %writable : f32

// CHECK-NEXT:             %readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)
// CHECK-NEXT:             %val = memref_stream.read from %readable : f32
// CHECK-NEXT:             memref_stream.write %val to %writable : f32

// CHECK-GENERIC-NEXT:    %readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)
// CHECK-GENERIC-NEXT:    %val = "memref_stream.read"(%readable) : (!stream.readable<f32>) -> f32
// CHECK-GENERIC-NEXT:    "memref_stream.write"(%val, %writable) : (f32, !stream.writable<f32>) -> ()

%A, %B, %C, %D = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>, f64)

memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>,
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>,
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>
    ]
} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs = {hello = "world"} {
^bb0(%a: !stream.readable<f32>, %b: !stream.readable<f32>, %c: !stream.writable<f64>):
    "test.op"(%a, %b, %c) : (!stream.readable<f32>, !stream.readable<f32>, !stream.writable<f64>) -> ()
}

// CHECK-NEXT:            %A, %B, %C, %D = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>, f64)
// CHECK-NEXT:            memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>]} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs =  {"hello" = "world"} {
// CHECK-NEXT:            ^0(%a : !stream.readable<f32>, %b : !stream.readable<f32>, %c : !stream.writable<f64>):
// CHECK-NEXT:              "test.op"(%a, %b, %c) : (!stream.readable<f32>, !stream.readable<f32>, !stream.writable<f64>) -> ()
// CHECK-NEXT:            }

// CHECK-GENERIC-NEXT:    %A, %B, %C, %D = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>, f64)
// CHECK-GENERIC-NEXT:    "memref_stream.streaming_region"(%A, %B, %C) <{"patterns" = [#memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>], "operandSegmentSizes" = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^0(%a : !stream.readable<f32>, %b : !stream.readable<f32>, %c : !stream.writable<f64>):
// CHECK-GENERIC-NEXT:      "test.op"(%a, %b, %c) : (!stream.readable<f32>, !stream.readable<f32>, !stream.writable<f64>) -> ()
// CHECK-GENERIC-NEXT:    }) {"hello" = "world"} : (memref<2xf32>, memref<3xf32>, memref<3x2xf64>) -> ()


memref_stream.generic {
    bounds = [3, 2],
    indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs = {hello = "world"} {
^bb0(%arg3: f32, %arg4: f32):
    memref_stream.yield %arg3 : f32
}

// CHECK-NEXT:          memref_stream.generic {bounds = [3, 2], indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs =  {"hello" = "world"} {
// CHECK-NEXT:          ^1(%arg3 : f32, %arg4 : f32):
// CHECK-NEXT:            memref_stream.yield %arg3 : f32
// CHECK-NEXT:          }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%A, %B, %C) <{"bounds" = [3 : index, 2 : index], "init_indices" = [], "indexing_maps" = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 2, 1, 0>}> ({
// CHECK-GENERIC-NEXT:    ^1(%arg3 : f32, %arg4 : f32):
// CHECK-GENERIC-NEXT:      "memref_stream.yield"(%arg3) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) {"hello" = "world"} : (memref<2xf32>, memref<3xf32>, memref<3x2xf64>) -> ()

memref_stream.generic {
    bounds = [3, 2],
    indexing_maps = [
        affine_map<(d0, d1) -> ()>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
} ins(%D : f64) outs(%C : memref<3x2xf64>) {
^bb0(%d : f64, %c : f64):
    memref_stream.yield %d : f64
}

// CHECK-NEXT:    memref_stream.generic {bounds = [3, 2], indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%D : f64) outs(%C : memref<3x2xf64>) {
// CHECK-NEXT:    ^2(%d : f64, %c_1 : f64):
// CHECK-NEXT:      memref_stream.yield %d : f64
// CHECK-NEXT:    }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%D, %C) <{"bounds" = [3 : index, 2 : index], "init_indices" = [], "indexing_maps" = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>], "operandSegmentSizes" = array<i32: 1, 1, 0>}> ({
// CHECK-GENERIC-NEXT:    ^2(%d : f64, %c_1 : f64):
// CHECK-GENERIC-NEXT:      "memref_stream.yield"(%d) : (f64) -> ()
// CHECK-GENERIC-NEXT:    }) : (f64, memref<3x2xf64>) -> ()

%E, %F, %G, %H = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, f64)
// CHECK-NEXT:            %E, %F, %G, %H = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, f64)
// CHECK-GENERIC-NEXT:    %E, %F, %G, %H = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, f64)

memref_stream.generic {
    bounds = [4, 2, 3],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%E, %F : memref<4x2xf64>, memref<2x3xf64>) outs(%G : memref<4x3xf64>) inits(%H : f64) {
^0(%e : f64, %f : f64, %acc_old : f64):
    %prod = arith.mulf %e, %f : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}

// CHECK-NEXT:    memref_stream.generic {bounds = [4, 2, 3], indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%{{.*}}, %{{.*}} : memref<4x2xf64>, memref<2x3xf64>) outs(%{{.*}} : memref<4x3xf64>) inits(%H : f64) {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:      linalg.yield %{{.*}} : f64
// CHECK-NEXT:    }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%{{.*}}, %{{.*}}, %{{.*}}, %H) <{"bounds" = [4 : index, 2 : index, 3 : index], "init_indices" = [#builtin.int<0>], "indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<reduction>], "operandSegmentSizes" = array<i32: 2, 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{"fastmath" = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{"fastmath" = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f64) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, f64) -> ()

%I = "test.op"() : () -> memref<4x3xf64>
// CHECK-NEXT:            %I = "test.op"() : () -> memref<4x3xf64>
// CHECK-GENERIC-NEXT:    %I = "test.op"() : () -> memref<4x3xf64>

memref_stream.generic {
    bounds = [4, 2, 3],
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
} ins(%E, %F : memref<4x2xf64>, memref<2x3xf64>) outs(%G, %I : memref<4x3xf64>, memref<4x3xf64>) inits(%H : f64, None) {
^0(%e : f64, %f : f64, %acc_old_0 : f64, %acc_old_1 : f64):
    %prod = arith.mulf %e, %f : f64
    %acc_new_0 = arith.addf %acc_old_0, %prod : f64
    %acc_new_1 = arith.addf %acc_old_1, %prod : f64
    linalg.yield %acc_new_0, %acc_new_1 : f64, f64
}

// CHECK-NEXT:    memref_stream.generic {bounds = [4, 2, 3], indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%{{.*}}, %{{.*}} : memref<4x2xf64>, memref<2x3xf64>) outs(%{{.*}}, %{{.*}} : memref<4x3xf64>, memref<4x3xf64>) inits(%{{.*}} : f64, None) {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:      linalg.yield %{{.*}}, %{{.*}} : f64, f64
// CHECK-NEXT:    }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{"bounds" = [4 : index, 2 : index, 3 : index], "init_indices" = [#builtin.int<0>], "indexing_maps" = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "iterator_types" = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<reduction>], "operandSegmentSizes" = array<i32: 2, 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{"fastmath" = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{"fastmath" = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{"fastmath" = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}, %{{.*}}) : (f64, f64) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, memref<4x3xf64>, f64) -> ()


memref_stream.fill %C with %D : memref<3x2xf64>

// CHECK-NEXT: memref_stream.fill %C with %D : memref<3x2xf64>
// CHECK-GENERIC-NEXT: "memref_stream.fill"(%C, %D) : (memref<3x2xf64>, f64) -> ()


// CHECK-NEXT:          }
// CHECK-GENERIC-NEXT:  }) : () -> ()
