// RUN: xdsl-opt -p convert-memref-stream-to-loops %s | filecheck %s

// CHECK:       builtin.module {

builtin.module {
  func.func public @dsum(%arg0 : memref<8x16xf64>, %arg1 : memref<8x16xf64>, %arg2 : memref<8x16xf64>) -> memref<8x16xf64> {
    memref_stream.streaming_region {bounds = [8, 16], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]} ins(%arg0, %arg1 : memref<8x16xf64>, memref<8x16xf64>) outs(%arg2 : memref<8x16xf64>) {
    ^0(%0 : !stream.readable<f64>, %1 : !stream.readable<f64>, %2 : !stream.writable<f64>):
      memref_stream.generic {bounds = [#builtin.int<8>, #builtin.int<16>], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : !stream.readable<f64>, !stream.readable<f64>) outs(%2 : !stream.writable<f64>) {
      ^1(%in : f64, %in_0 : f64, %out : f64):
        %3 = arith.addf %in, %in_0 : f64
        memref_stream.yield %3 : f64
      }
    }
    func.return %arg2 : memref<8x16xf64>
  }
// CHECK-NEXT:    func.func public @dsum(%{{.*}} : memref<8x16xf64>, %{{.*}} : memref<8x16xf64>, %{{.*}} : memref<8x16xf64>) -> memref<8x16xf64> {
// CHECK-NEXT:      memref_stream.streaming_region {bounds = [8, 16], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]} ins(%{{.*}}, %{{.*}} : memref<8x16xf64>, memref<8x16xf64>) outs(%{{.*}} : memref<8x16xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stream.readable<f64>, %{{.*}} : !stream.readable<f64>, %{{.*}} : !stream.writable<f64>):
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 16 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:            %{{.*}} = memref_stream.read from %{{.*}} : f64
// CHECK-NEXT:            %{{.*}} = memref_stream.read from %{{.*}} : f64
// CHECK-NEXT:            %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:            memref_stream.write %{{.*}} to %{{.*}} : f64
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<8x16xf64>
// CHECK-NEXT:    }

  func.func public @relu(%arg0_1 : memref<16x16xf64>, %arg1_1 : memref<16x16xf64>) -> memref<16x16xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    memref_stream.streaming_region {bounds = [16, 16], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]} ins(%arg0_1 : memref<16x16xf64>) outs(%arg1_1 : memref<16x16xf64>) {
    ^2(%4 : !stream.readable<f64>, %5 : !stream.writable<f64>):
      memref_stream.generic {bounds = [#builtin.int<16>, #builtin.int<16>], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : !stream.readable<f64>) outs(%5 : !stream.writable<f64>) {
      ^3(%in_1 : f64, %out_1 : f64):
        %6 = arith.maximumf %in_1, %cst : f64
        memref_stream.yield %6 : f64
      }
    }
    func.return %arg1_1 : memref<16x16xf64>
  }
// CHECK-NEXT:    func.func public @relu(%{{.*}} : memref<16x16xf64>, %{{.*}} : memref<16x16xf64>) -> memref<16x16xf64> {
// CHECK-NEXT:      %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {bounds = [16, 16], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]} ins(%{{.*}} : memref<16x16xf64>) outs(%{{.*}} : memref<16x16xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stream.readable<f64>, %{{.*}} : !stream.writable<f64>):
// CHECK-NEXT:        %{{.*}} = arith.constant 16 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 16 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:            %{{.*}} = memref_stream.read from %{{.*}} : f64
// CHECK-NEXT:            %{{.*}} = arith.maximumf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:            memref_stream.write %{{.*}} to %{{.*}} : f64
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<16x16xf64>
// CHECK-NEXT:    }

}
// CHECK-NEXT:  }
