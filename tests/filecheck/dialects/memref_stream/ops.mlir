// RUN: XDSL_ROUNDTRIP --match-full-lines --strict-whitespace
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:builtin.module {
// CHECK-GENERIC:       "builtin.module"() ({

%readable, %writable = "test.op"() : () -> (!memref_stream.readable<f32>, !memref_stream.writable<f32>)

%val = memref_stream.read from %readable : f32
memref_stream.write %val to %writable : f32

// CHECK-NEXT:  %readable, %writable = "test.op"() : () -> (!memref_stream.readable<f32>, !memref_stream.writable<f32>)
// CHECK-NEXT:  %val = memref_stream.read from %readable : f32
// CHECK-NEXT:  memref_stream.write %val to %writable : f32

// CHECK-GENERIC-NEXT:    %readable, %writable = "test.op"() : () -> (!memref_stream.readable<f32>, !memref_stream.writable<f32>)
// CHECK-GENERIC-NEXT:    %val = "memref_stream.read"(%readable) : (!memref_stream.readable<f32>) -> f32
// CHECK-GENERIC-NEXT:    "memref_stream.write"(%val, %writable) : (f32, !memref_stream.writable<f32>) -> ()

%A, %B, %C, %D = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>, f64)

memref_stream.streaming_region {
    patterns = [
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>,
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>,
        #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>
    ]
} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs = {hello = "world"} {
^bb0(%a: !memref_stream.readable<f32>, %b: !memref_stream.readable<f32>, %c: !memref_stream.writable<f64>):
    "test.op"(%a, %b, %c) : (!memref_stream.readable<f32>, !memref_stream.readable<f32>, !memref_stream.writable<f64>) -> ()
}

// CHECK-NEXT:  %A, %B, %C, %D = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>, f64)
// CHECK-NEXT:  memref_stream.streaming_region {
// CHECK-NEXT:    patterns = [
// CHECK-NEXT:      #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>,
// CHECK-NEXT:      #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>,
// CHECK-NEXT:      #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>
// CHECK-NEXT:    ]
// CHECK-NEXT:  } ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs =  {hello = "world"} {
// CHECK-NEXT:  ^bb0(%a : !memref_stream.readable<f32>, %b : !memref_stream.readable<f32>, %c : !memref_stream.writable<f64>):
// CHECK-NEXT:    "test.op"(%a, %b, %c) : (!memref_stream.readable<f32>, !memref_stream.readable<f32>, !memref_stream.writable<f64>) -> ()
// CHECK-NEXT:  }

// CHECK-GENERIC-NEXT:    %A, %B, %C, %D = "test.op"() : () -> (memref<2xf32>, memref<3xf32>, memref<3x2xf64>, f64)
// CHECK-GENERIC-NEXT:    "memref_stream.streaming_region"(%A, %B, %C) <{patterns = [#memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d1)>, #memref_stream.stride_pattern<ub = [3, 2], index_map = (d0, d1) -> (d0, d1)>], operandSegmentSizes = array<i32: 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^bb0(%a : !memref_stream.readable<f32>, %b : !memref_stream.readable<f32>, %c : !memref_stream.writable<f64>):
// CHECK-GENERIC-NEXT:      "test.op"(%a, %b, %c) : (!memref_stream.readable<f32>, !memref_stream.readable<f32>, !memref_stream.writable<f64>) -> ()
// CHECK-GENERIC-NEXT:    }) {hello = "world"} : (memref<2xf32>, memref<3xf32>, memref<3x2xf64>) -> ()


memref_stream.generic {
    bounds = [3, 2],
    indexing_maps = [
        affine_map<(d0, d1) -> (d0)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"],
    doc = "documentation string",
    library_call = "library call"
} ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs = {hello = "world"} {
^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    memref_stream.yield %arg3 : f32
}

// CHECK-NEXT:  memref_stream.generic {
// CHECK-NEXT:    bounds = [3, 2],
// CHECK-NEXT:    indexing_maps = [
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d1)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:    ],
// CHECK-NEXT:    iterator_types = ["parallel", "parallel"],
// CHECK-NEXT:    doc = "documentation string",
// CHECK-NEXT:    library_call = "library call"
// CHECK-NEXT:  } ins(%A, %B : memref<2xf32>, memref<3xf32>) outs(%C : memref<3x2xf64>) attrs =  {hello = "world"} {
// CHECK-NEXT:  ^bb1(%arg3 : f32, %arg4 : f32, %arg5 : f32):
// CHECK-NEXT:    memref_stream.yield %arg3 : f32
// CHECK-NEXT:  }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%A, %B, %C) <{bounds = [3 : index, 2 : index], init_indices = [], indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>], doc = "documentation string", library_call = "library call", operandSegmentSizes = array<i32: 2, 1, 0>}> ({
// CHECK-GENERIC-NEXT:    ^bb1(%arg3 : f32, %arg4 : f32, %arg5 : f32):
// CHECK-GENERIC-NEXT:      "memref_stream.yield"(%arg3) : (f32) -> ()
// CHECK-GENERIC-NEXT:    }) {hello = "world"} : (memref<2xf32>, memref<3xf32>, memref<3x2xf64>) -> ()

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

// CHECK-NEXT:  memref_stream.generic {
// CHECK-NEXT:    bounds = [3, 2],
// CHECK-NEXT:    indexing_maps = [
// CHECK-NEXT:      affine_map<(d0, d1) -> ()>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:    ],
// CHECK-NEXT:    iterator_types = ["parallel", "parallel"]
// CHECK-NEXT:  } ins(%D : f64) outs(%C : memref<3x2xf64>) {
// CHECK-NEXT:  ^bb2(%d : f64, %c_1 : f64):
// CHECK-NEXT:    memref_stream.yield %d : f64
// CHECK-NEXT:  }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%D, %C) <{bounds = [3 : index, 2 : index], init_indices = [], indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1, 0>}> ({
// CHECK-GENERIC-NEXT:    ^bb2(%d : f64, %c_1 : f64):
// CHECK-GENERIC-NEXT:      "memref_stream.yield"(%d) : (f64) -> ()
// CHECK-GENERIC-NEXT:    }) : (f64, memref<3x2xf64>) -> ()

%E, %F, %G, %H = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, f64)
// CHECK-NEXT:  %E, %F, %G, %H = "test.op"() : () -> (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, f64)
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
^bb0(%e : f64, %f : f64, %acc_old : f64):
    %prod = arith.mulf %e, %f : f64
    %acc_new = arith.addf %acc_old, %prod : f64
    linalg.yield %acc_new : f64
}

// CHECK-NEXT:  memref_stream.generic {
// CHECK-NEXT:    bounds = [4, 2, 3],
// CHECK-NEXT:    indexing_maps = [
// CHECK-NEXT:      affine_map<(d0, d1, d2) -> (d0, d2)>,
// CHECK-NEXT:      affine_map<(d0, d1, d2) -> (d2, d1)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:    ],
// CHECK-NEXT:    iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:  } ins(%{{.*}}, %{{.*}} : memref<4x2xf64>, memref<2x3xf64>) outs(%{{.*}} : memref<4x3xf64>) inits(%H : f64) {
// CHECK-NEXT:  ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    linalg.yield %{{.*}} : f64
// CHECK-NEXT:  }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%{{.*}}, %{{.*}}, %{{.*}}, %H) <{bounds = [4 : index, 2 : index, 3 : index], init_indices = [#builtin.int<0>], indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}) : (f64) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, f64) -> ()

%I = "test.op"() : () -> memref<4x3xf64>
// CHECK-NEXT:  %I = "test.op"() : () -> memref<4x3xf64>
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
^bb0(%e : f64, %f : f64, %acc_old_0 : f64, %acc_old_1 : f64):
    %prod = arith.mulf %e, %f : f64
    %acc_new_0 = arith.addf %acc_old_0, %prod : f64
    %acc_new_1 = arith.addf %acc_old_1, %prod : f64
    linalg.yield %acc_new_0, %acc_new_1 : f64, f64
}

// CHECK-NEXT:  memref_stream.generic {
// CHECK-NEXT:    bounds = [4, 2, 3],
// CHECK-NEXT:    indexing_maps = [
// CHECK-NEXT:      affine_map<(d0, d1, d2) -> (d0, d2)>,
// CHECK-NEXT:      affine_map<(d0, d1, d2) -> (d2, d1)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:      affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:    ],
// CHECK-NEXT:    iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:  } ins(%{{.*}}, %{{.*}} : memref<4x2xf64>, memref<2x3xf64>) outs(%{{.*}}, %{{.*}} : memref<4x3xf64>, memref<4x3xf64>) inits(%{{.*}} : f64, None) {
// CHECK-NEXT:  ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:    %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:    linalg.yield %{{.*}}, %{{.*}} : f64, f64
// CHECK-NEXT:  }

// CHECK-GENERIC-NEXT:    "memref_stream.generic"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{bounds = [4 : index, 2 : index, 3 : index], init_indices = [#builtin.int<0>], indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 2, 1>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<none>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:      "linalg.yield"(%{{.*}}, %{{.*}}) : (f64, f64) -> ()
// CHECK-GENERIC-NEXT:    }) : (memref<4x2xf64>, memref<2x3xf64>, memref<4x3xf64>, memref<4x3xf64>, f64) -> ()



func.func @interleaved_no_init(%A0 : memref<3x5xf64>, %B0 : memref<5x8xf64>, %C0 : memref<3x8xf64>) -> memref<3x8xf64> {
    memref_stream.generic {
        bounds = [3, 2, 5, 4],
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
            affine_map<(d0, d1, d2, d3) -> (d2, d1 * 4 + d3)>,
            affine_map<(d0, d1, d3) -> (d0, d1 * 4 + d3)>
        ],
        iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
    } ins(%A0, %B0 : memref<3x5xf64>, memref<5x8xf64>) outs(%C0 : memref<3x8xf64>) {
    ^bb1(
        %a0 : f64, %a1 : f64, %a2 : f64, %a3 : f64,
        %b0 : f64, %b1 : f64, %b2 : f64, %b3 : f64,
        %c0 : f64, %c1 : f64, %c2 : f64, %c3 : f64
    ):
        %prod0 = arith.mulf %a0, %b0 fastmath<fast> : f64
        %prod1 = arith.mulf %a1, %b1 fastmath<fast> : f64
        %prod2 = arith.mulf %a2, %b2 fastmath<fast> : f64
        %prod3 = arith.mulf %a3, %b3 fastmath<fast> : f64

        %res0 = arith.addf %prod0, %c0 fastmath<fast> : f64
        %res1 = arith.addf %prod1, %c1 fastmath<fast> : f64
        %res2 = arith.addf %prod2, %c2 fastmath<fast> : f64
        %res3 = arith.addf %prod3, %c3 fastmath<fast> : f64

        memref_stream.yield %res0, %res1, %res2, %res3 : f64, f64, f64, f64
    }
    func.return %C0 : memref<3x8xf64>
}

// CHECK-NEXT:  func.func @interleaved_no_init(%{{.*}} : memref<3x5xf64>, %{{.*}} : memref<5x8xf64>, %{{.*}} : memref<3x8xf64>) -> memref<3x8xf64> {
// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [3, 2, 5, 4],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// CHECK-NEXT:    } ins(%{{.*}}, %{{.*}} : memref<3x5xf64>, memref<5x8xf64>) outs(%{{.*}} : memref<3x8xf64>) {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      memref_stream.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %{{.*}} : memref<3x8xf64>
// CHECK-NEXT:  }

// CHECK-GENERIC-NEXT:    "func.func"() <{sym_name = "interleaved_no_init", function_type = (memref<3x5xf64>, memref<5x8xf64>, memref<3x8xf64>) -> memref<3x8xf64>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : memref<3x5xf64>, %{{.*}} : memref<5x8xf64>, %{{.*}} : memref<3x8xf64>):
// CHECK-GENERIC-NEXT:      "memref_stream.generic"(%{{.*}}, %{{.*}}, %{{.*}}) <{bounds = [3 : index, 2 : index, 5 : index, 4 : index], init_indices = [], indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>, affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>, affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>], iterator_types = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<reduction>, #memref_stream.iterator_type<interleaved>], operandSegmentSizes = array<i32: 2, 1, 0>}> ({
// CHECK-GENERIC-NEXT:      ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        "memref_stream.yield"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f64, f64, f64, f64) -> ()
// CHECK-GENERIC-NEXT:      }) : (memref<3x5xf64>, memref<5x8xf64>, memref<3x8xf64>) -> ()
// CHECK-GENERIC-NEXT:      "func.return"(%{{.*}}) : (memref<3x8xf64>) -> ()
// CHECK-GENERIC-NEXT:    }) : () -> ()


func.func @interleaved_init(%A1 : memref<3x5xf64>, %B1 : memref<5x8xf64>, %C1 : memref<3x8xf64>) -> memref<3x8xf64> {
    %zero_float = arith.constant 0.000000e+00 : f64
    memref_stream.generic {
        bounds = [3, 2, 5, 4],
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
            affine_map<(d0, d1, d2, d3) -> (d2, d1 * 4 + d3)>,
            affine_map<(d0, d1, d3) -> (d0, d1 * 4 + d3)>
        ],
        iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
    } ins(%A1, %B1 : memref<3x5xf64>, memref<5x8xf64>) outs(%C1 : memref<3x8xf64>) inits(%zero_float : f64) {
    ^bb1(
        %a0 : f64, %a1 : f64, %a2 : f64, %a3 : f64,
        %b0 : f64, %b1 : f64, %b2 : f64, %b3 : f64,
        %c0 : f64, %c1 : f64, %c2 : f64, %c3 : f64
    ):
        %prod0 = arith.mulf %a0, %b0 fastmath<fast> : f64
        %prod1 = arith.mulf %a1, %b1 fastmath<fast> : f64
        %prod2 = arith.mulf %a2, %b2 fastmath<fast> : f64
        %prod3 = arith.mulf %a3, %b3 fastmath<fast> : f64

        %res0 = arith.addf %prod0, %c0 fastmath<fast> : f64
        %res1 = arith.addf %prod1, %c1 fastmath<fast> : f64
        %res2 = arith.addf %prod2, %c2 fastmath<fast> : f64
        %res3 = arith.addf %prod3, %c3 fastmath<fast> : f64

        memref_stream.yield %res0, %res1, %res2, %res3 : f64, f64, f64, f64
    }
    func.return %C1 : memref<3x8xf64>
}

// CHECK-NEXT:  func.func @interleaved_init(%{{.*}} : memref<3x5xf64>, %{{.*}} : memref<5x8xf64>, %{{.*}} : memref<3x8xf64>) -> memref<3x8xf64> {
// CHECK-NEXT:    %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:    memref_stream.generic {
// CHECK-NEXT:      bounds = [3, 2, 5, 4],
// CHECK-NEXT:      indexing_maps = [
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// CHECK-NEXT:        affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// CHECK-NEXT:        affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// CHECK-NEXT:      ],
// CHECK-NEXT:      iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// CHECK-NEXT:    } ins(%{{.*}}, %{{.*}} : memref<3x5xf64>, memref<5x8xf64>) outs(%{{.*}} : memref<3x8xf64>) inits(%{{.*}} : f64) {
// CHECK-NEXT:    ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:      memref_stream.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.return %{{.*}} : memref<3x8xf64>
// CHECK-NEXT:  }

// CHECK-GENERIC-NEXT:    "func.func"() <{sym_name = "interleaved_init", function_type = (memref<3x5xf64>, memref<5x8xf64>, memref<3x8xf64>) -> memref<3x8xf64>}> ({
// CHECK-GENERIC-NEXT:    ^{{.*}}(%{{.*}} : memref<3x5xf64>, %{{.*}} : memref<5x8xf64>, %{{.*}} : memref<3x8xf64>):
// CHECK-GENERIC-NEXT:      %{{.*}} = "arith.constant"() <{value = 0.000000e+00 : f64}> : () -> f64
// CHECK-GENERIC-NEXT:      "memref_stream.generic"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) <{bounds = [3 : index, 2 : index, 5 : index, 4 : index], init_indices = [#builtin.int<0>], indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>, affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>, affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>], iterator_types = [#memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<parallel>, #memref_stream.iterator_type<reduction>, #memref_stream.iterator_type<interleaved>], operandSegmentSizes = array<i32: 2, 1, 1>}> ({
// CHECK-GENERIC-NEXT:      ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) <{fastmath = #arith.fastmath<fast>}> : (f64, f64) -> f64
// CHECK-GENERIC-NEXT:        "memref_stream.yield"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f64, f64, f64, f64) -> ()
// CHECK-GENERIC-NEXT:      }) : (memref<3x5xf64>, memref<5x8xf64>, memref<3x8xf64>, f64) -> ()
// CHECK-GENERIC-NEXT:      "func.return"(%{{.*}}) : (memref<3x8xf64>) -> ()
// CHECK-GENERIC-NEXT:    }) : () -> ()

memref_stream.fill %C with %D : memref<3x2xf64>

// CHECK-NEXT:  memref_stream.fill %C with %D : memref<3x2xf64>
// CHECK-GENERIC-NEXT: "memref_stream.fill"(%C, %D) : (memref<3x2xf64>, f64) -> ()


// CHECK-NEXT:}
// CHECK-GENERIC-NEXT:  }) : () -> ()
