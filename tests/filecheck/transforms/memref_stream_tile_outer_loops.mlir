// RUN: xdsl-opt --split-input-file --verify-diagnostics -p memref-stream-tile-outer-loops{target-rank=4} %s | filecheck %s

func.func public @pooling_nchw_max_d1_s2_3x3(%X : memref<1x1x18x18xf64> {"llvm.noalias"}, %Y : memref<1x1x8x8xf64> {"llvm.noalias"}) -> memref<1x1x8x8xf64> {
    %cst = arith.constant -1.000000e+04 : f64
    memref_stream.generic {
        bounds = [1, 1, 8, 2, 3, 3, 4],
        indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, ((d2 * 2) + d4), ((((d3 * 4) + d6) * 2) + d5))>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, ((d3 * 4) + d4))>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "interleaved"]
    } ins(%X : memref<1x1x18x18xf64>) outs(%Y : memref<1x1x8x8xf64>) inits(%cst : f64) {
    ^0(%in : f64, %in_1 : f64, %in_2 : f64, %in_3 : f64, %out : f64, %out_1 : f64, %out_2 : f64, %out_3 : f64):
        %res_0 = arith.maximumf %out, %in fastmath<fast> : f64
        %res_1 = arith.maximumf %out_1, %in_1 fastmath<fast> : f64
        %res_2 = arith.maximumf %out_2, %in_2 fastmath<fast> : f64
        %res_3 = arith.maximumf %out_3, %in_3 fastmath<fast> : f64
        memref_stream.yield %res_0, %res_1, %res_2, %res_3 : f64, f64, f64, f64
    }
    func.return %Y : memref<1x1x8x8xf64>
}


// CHECK:       builtin.module {
// CHECK-NEXT:    func.func public @pooling_nchw_max_d1_s2_3x3(%X : memref<1x1x18x18xf64> {llvm.noalias}, %Y : memref<1x1x8x8xf64> {llvm.noalias}) -> memref<1x1x8x8xf64> {
// CHECK-NEXT:      %cst = arith.constant -1.000000e+04 : f64
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %ub = arith.constant 8 : index
// CHECK-NEXT:      scf.for %i = %c0 to %ub step %c1 {
// CHECK-NEXT:        %X_offset = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0)> (%c0, %c0, %i, %c0, %c0, %c0, %c0)
// CHECK-NEXT:        %X_offset_1 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1)> (%c0, %c0, %i, %c0, %c0, %c0, %c0)
// CHECK-NEXT:        %X_offset_2 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 2) + d4))> (%c0, %c0, %i, %c0, %c0, %c0, %c0)
// CHECK-NEXT:        %X_offset_3 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((((d3 * 8) + (d6 * 2)) + d5))> (%c0, %c0, %i, %c0, %c0, %c0, %c0)
// CHECK-NEXT:        %X_subview = memref.subview %X[%X_offset, %X_offset_1, %X_offset_2, %X_offset_3] [1, 1, 3, 17] [1, 1, 1, 1] : memref<1x1x18x18xf64> to memref<1x1x3x17xf64, strided<[324, 324, 18, 1], offset: ?>>
// CHECK-NEXT:        %Y_offset = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0)> (%c0, %c0, %i, %c0, %c0)
// CHECK-NEXT:        %Y_offset_1 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d1)> (%c0, %c0, %i, %c0, %c0)
// CHECK-NEXT:        %Y_offset_2 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d2)> (%c0, %c0, %i, %c0, %c0)
// CHECK-NEXT:        %Y_offset_3 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (((d3 * 4) + d4))> (%c0, %c0, %i, %c0, %c0)
// CHECK-NEXT:        %Y_subview = memref.subview %Y[%Y_offset, %Y_offset_1, %Y_offset_2, %Y_offset_3] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x1x8x8xf64> to memref<1x1x1x8xf64, strided<[64, 64, 8, 1], offset: ?>>
// CHECK-NEXT:        memref_stream.generic {
// CHECK-NEXT:          bounds = [1, 1, 1, 2, 3, 3, 4],
// CHECK-NEXT:          indexing_maps = [
// CHECK-NEXT:            affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, ((d2 * 2) + d4), (((d3 * 8) + (d6 * 2)) + d5))>
// CHECK-NEXT:            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, ((d3 * 4) + d4))>
// CHECK-NEXT:          ],
// CHECK-NEXT:          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "interleaved"]
// CHECK-NEXT:        } ins(%X_subview : memref<1x1x3x17xf64, strided<[324, 324, 18, 1], offset: ?>>) outs(%Y_subview : memref<1x1x1x8xf64, strided<[64, 64, 8, 1], offset: ?>>) inits(%cst : f64) {
// CHECK-NEXT:        ^0(%in : f64, %in_1 : f64, %in_2 : f64, %in_3 : f64, %out : f64, %out_1 : f64, %out_2 : f64, %out_3 : f64):
// CHECK-NEXT:          %res = arith.maximumf %out, %in fastmath<fast> : f64
// CHECK-NEXT:          %res_1 = arith.maximumf %out_1, %in_1 fastmath<fast> : f64
// CHECK-NEXT:          %res_2 = arith.maximumf %out_2, %in_2 fastmath<fast> : f64
// CHECK-NEXT:          %res_3 = arith.maximumf %out_3, %in_3 fastmath<fast> : f64
// CHECK-NEXT:          memref_stream.yield %res, %res_1, %res_2, %res_3 : f64, f64, f64, f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %Y : memref<1x1x8x8xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// -----

func.func public @pooling_nchw_max_d1_s2_3x3(%X : memref<1x2x18x18xf64> {"llvm.noalias"}, %Y : memref<1x2x8x8xf64> {"llvm.noalias"}) -> memref<1x2x8x8xf64> {
    %cst = arith.constant -1.000000e+04 : f64
    memref_stream.generic {
        bounds = [1, 2, 8, 2, 3, 3, 4],
        indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, ((d2 * 2) + d4), ((((d3 * 4) + d6) * 2) + d5))>,
            affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, ((d3 * 4) + d4))>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "interleaved"]
    } ins(%X : memref<1x2x18x18xf64>) outs(%Y : memref<1x2x8x8xf64>) inits(%cst : f64) {
    ^0(%in : f64, %in_1 : f64, %in_2 : f64, %in_3 : f64, %out : f64, %out_1 : f64, %out_2 : f64, %out_3 : f64):
        %res_0 = arith.maximumf %out, %in fastmath<fast> : f64
        %res_1 = arith.maximumf %out_1, %in_1 fastmath<fast> : f64
        %res_2 = arith.maximumf %out_2, %in_2 fastmath<fast> : f64
        %res_3 = arith.maximumf %out_3, %in_3 fastmath<fast> : f64
        memref_stream.yield %res_0, %res_1, %res_2, %res_3 : f64, f64, f64, f64
    }
    func.return %Y : memref<1x2x8x8xf64>
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func public @pooling_nchw_max_d1_s2_3x3(%X : memref<1x2x18x18xf64> {llvm.noalias}, %Y : memref<1x2x8x8xf64> {llvm.noalias}) -> memref<1x2x8x8xf64> {
// CHECK-NEXT:      %cst = arith.constant -1.000000e+04 : f64
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %ub = arith.constant 2 : index
// CHECK-NEXT:      scf.for %i = %c0 to %ub step %c1 {
// CHECK-NEXT:        %X_offset = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0)> (%c0, %i, %c0, %c0, %c0, %c0, %c0)
// CHECK-NEXT:        %X_offset_1 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1)> (%c0, %i, %c0, %c0, %c0, %c0, %c0)
// CHECK-NEXT:        %X_offset_2 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 2) + d4))> (%c0, %i, %c0, %c0, %c0, %c0, %c0)
// CHECK-NEXT:        %X_offset_3 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((((d3 * 8) + (d6 * 2)) + d5))> (%c0, %i, %c0, %c0, %c0, %c0, %c0)
// CHECK-NEXT:        %X_subview = memref.subview %X[%X_offset, %X_offset_1, %X_offset_2, %X_offset_3] [1, 1, 17, 17] [1, 1, 1, 1] : memref<1x2x18x18xf64> to memref<1x1x17x17xf64, strided<[648, 324, 18, 1], offset: ?>>
// CHECK-NEXT:        %Y_offset = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0)> (%c0, %i, %c0, %c0, %c0)
// CHECK-NEXT:        %Y_offset_1 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d1)> (%c0, %i, %c0, %c0, %c0)
// CHECK-NEXT:        %Y_offset_2 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d2)> (%c0, %i, %c0, %c0, %c0)
// CHECK-NEXT:        %Y_offset_3 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (((d3 * 4) + d4))> (%c0, %i, %c0, %c0, %c0)
// CHECK-NEXT:        %Y_subview = memref.subview %Y[%Y_offset, %Y_offset_1, %Y_offset_2, %Y_offset_3] [1, 1, 8, 8] [1, 1, 1, 1] : memref<1x2x8x8xf64> to memref<1x1x8x8xf64, strided<[128, 64, 8, 1], offset: ?>>
// CHECK-NEXT:        %c0_1 = arith.constant 0 : index
// CHECK-NEXT:        %c1_1 = arith.constant 1 : index
// CHECK-NEXT:        %ub_1 = arith.constant 8 : index
// CHECK-NEXT:        scf.for %i_1 = %c0_1 to %ub_1 step %c1_1 {
// CHECK-NEXT:          %X_subview_offset = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0)> (%c0_1, %c0_1, %i_1, %c0_1, %c0_1, %c0_1, %c0_1)
// CHECK-NEXT:          %X_subview_offset_1 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1)> (%c0_1, %c0_1, %i_1, %c0_1, %c0_1, %c0_1, %c0_1)
// CHECK-NEXT:          %X_subview_offset_2 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d2 * 2) + d4))> (%c0_1, %c0_1, %i_1, %c0_1, %c0_1, %c0_1, %c0_1)
// CHECK-NEXT:          %X_subview_offset_3 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((((d3 * 8) + (d6 * 2)) + d5))> (%c0_1, %c0_1, %i_1, %c0_1, %c0_1, %c0_1, %c0_1)
// CHECK-NEXT:          %X_subview_subview = memref.subview %X_subview[%X_subview_offset, %X_subview_offset_1, %X_subview_offset_2, %X_subview_offset_3] [1, 1, 3, 17] [1, 1, 1, 1] : memref<1x1x17x17xf64, strided<[648, 324, 18, 1], offset: ?>> to memref<1x1x17x17xf64, strided<[648, 324, 18, 1], offset: ?>>
// CHECK-NEXT:          %Y_subview_offset = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0)> (%c0_1, %c0_1, %i_1, %c0_1, %c0_1)
// CHECK-NEXT:          %Y_subview_offset_1 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d1)> (%c0_1, %c0_1, %i_1, %c0_1, %c0_1)
// CHECK-NEXT:          %Y_subview_offset_2 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d2)> (%c0_1, %c0_1, %i_1, %c0_1, %c0_1)
// CHECK-NEXT:          %Y_subview_offset_3 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (((d3 * 4) + d4))> (%c0_1, %c0_1, %i_1, %c0_1, %c0_1)
// CHECK-NEXT:          %Y_subview_subview = memref.subview %Y_subview[%Y_subview_offset, %Y_subview_offset_1, %Y_subview_offset_2, %Y_subview_offset_3] [1, 1, 1, 8] [1, 1, 1, 1] : memref<1x1x8x8xf64, strided<[128, 64, 8, 1], offset: ?>> to memref<1x1x8x8xf64, strided<[128, 64, 8, 1], offset: ?>>
// CHECK-NEXT:          memref_stream.generic {
// CHECK-NEXT:            bounds = [1, 1, 1, 2, 3, 3, 4],
// CHECK-NEXT:            indexing_maps = [
// CHECK-NEXT:              affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, ((d2 * 2) + d4), (((d3 * 8) + (d6 * 2)) + d5))>,
// CHECK-NEXT:              affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, ((d3 * 4) + d4))>
// CHECK-NEXT:            ],
// CHECK-NEXT:            iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "interleaved"]
// CHECK-NEXT:          } ins(%X_subview_subview : memref<1x1x17x17xf64, strided<[648, 324, 18, 1], offset: ?>>) outs(%Y_subview_subview : memref<1x1x8x8xf64, strided<[128, 64, 8, 1], offset: ?>>) inits(%cst : f64) {
// CHECK-NEXT:          ^0(%in : f64, %in_1 : f64, %in_2 : f64, %in_3 : f64, %out : f64, %out_1 : f64, %out_2 : f64, %out_3 : f64):
// CHECK-NEXT:            %res = arith.maximumf %out, %in fastmath<fast> : f64
// CHECK-NEXT:            %res_1 = arith.maximumf %out_1, %in_1 fastmath<fast> : f64
// CHECK-NEXT:            %res_2 = arith.maximumf %out_2, %in_2 fastmath<fast> : f64
// CHECK-NEXT:            %res_3 = arith.maximumf %out_3, %in_3 fastmath<fast> : f64
// CHECK-NEXT:            memref_stream.yield %res, %res_1, %res_2, %res_3 : f64, f64, f64, f64
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %Y : memref<1x2x8x8xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
