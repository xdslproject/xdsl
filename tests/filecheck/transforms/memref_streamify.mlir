// RUN: xdsl-opt -p memref-streamify %s | filecheck %s

// Check that streamfying twice does not make further changes
// RUN: xdsl-opt -p memref-streamify,memref-streamify %s | filecheck %s


// CHECK:       builtin.module {

func.func public @dsum(%arg0 : memref<8x16xf64>, %arg1 : memref<8x16xf64>, %arg2 : memref<8x16xf64>) -> memref<8x16xf64> {
    memref_stream.generic {
        bounds = [#builtin.int<8>, #builtin.int<16>],
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : memref<8x16xf64>, memref<8x16xf64>) outs(%arg2 : memref<8x16xf64>) {
    ^0(%in : f64, %in_0 : f64, %out : f64):
        %0 = arith.addf %in, %in_0 : f64
        memref_stream.yield %0 : f64
    }
    func.return %arg2 : memref<8x16xf64>
}

// CHECK-NEXT:    func.func public @dsum(%arg0 : memref<8x16xf64>, %arg1 : memref<8x16xf64>, %arg2 : memref<8x16xf64>) -> memref<8x16xf64> {
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>, #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>, #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>]} ins(%arg0, %arg1 : memref<8x16xf64>, memref<8x16xf64>) outs(%arg2 : memref<8x16xf64>) {
// CHECK-NEXT:      ^0(%0 : !stream.readable<f64>, %1 : !stream.readable<f64>, %2 : !stream.writable<f64>):
// CHECK-NEXT:        memref_stream.generic {bounds = [#builtin.int<8>, #builtin.int<16>], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : !stream.readable<f64>, !stream.readable<f64>) outs(%2 : !stream.writable<f64>) {
// CHECK-NEXT:        ^1(%in : f64, %in_1 : f64, %out : f64):
// CHECK-NEXT:          %3 = arith.addf %in, %in_1 : f64
// CHECK-NEXT:          memref_stream.yield %3 : f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %arg2 : memref<8x16xf64>
// CHECK-NEXT:    }

func.func public @relu(%arg0 : memref<16x16xf64>, %arg1 : memref<16x16xf64>) -> memref<16x16xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    memref_stream.generic {
        bounds = [#builtin.int<16>, #builtin.int<16>],
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : memref<16x16xf64>) outs(%arg1 : memref<16x16xf64>) {
    ^1(%in_1 : f64, %out_1 : f64):
        %1 = arith.maximumf %in_1, %cst : f64
        memref_stream.yield %1 : f64
    }
    func.return %arg1 : memref<16x16xf64>
}

// CHECK-NEXT:    func.func public @relu(%arg0 : memref<16x16xf64>, %arg1 : memref<16x16xf64>) -> memref<16x16xf64> {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>, #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>]} ins(%arg0 : memref<16x16xf64>) outs(%arg1 : memref<16x16xf64>) {
// CHECK-NEXT:      ^0(%0 : !stream.readable<f64>, %1 : !stream.writable<f64>):
// CHECK-NEXT:        memref_stream.generic {bounds = [#builtin.int<16>, #builtin.int<16>], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : !stream.readable<f64>) outs(%1 : !stream.writable<f64>) {
// CHECK-NEXT:        ^1(%in : f64, %out : f64):
// CHECK-NEXT:          %2 = arith.maximumf %in, %cst : f64
// CHECK-NEXT:          memref_stream.yield %2 : f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %arg1 : memref<16x16xf64>
// CHECK-NEXT:    }

  func.func @fill(
    %X : f64,
    %Y : memref<16x16xf64>
  ) {
    memref_stream.generic {
        bounds = [#builtin.int<16>, #builtin.int<16>],
        indexing_maps = [
            affine_map<(d0, d1) -> ()>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%X : f64) outs(%Y : memref<16x16xf64>) {
    ^bb0(%d : f64, %c : f64):
        memref_stream.yield %d : f64
    }

    func.return
  }

// CHECK-NEXT:    func.func @fill(%{{.*}} : f64, %{{.*}} : memref<16x16xf64>) {
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>]} outs(%{{.*}} : memref<16x16xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stream.writable<f64>):
// CHECK-NEXT:        memref_stream.generic {bounds = [#builtin.int<16>, #builtin.int<16>], indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : f64) outs(%{{.*}} : !stream.writable<f64>) {
// CHECK-NEXT:        ^{{.*}}(%{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:          memref_stream.yield %{{.*}} : f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// CHECK-NEXT:  }
