// RUN: xdsl-opt -p memref-streamify %s | filecheck %s

// Check that streamfying twice does not make further changes
// RUN: xdsl-opt -p memref-streamify,memref-streamify %s | filecheck %s


// CHECK:       builtin.module {

func.func @fill_empty_shape(%scalar: memref<f64>) {
  %zero_float = arith.constant 0.000000e+00 : f64
  memref_stream.generic {
    bounds = [],
    indexing_maps = [
      affine_map<() -> ()>,
      affine_map<() -> ()>
    ],
    iterator_types = []
  } ins(%zero_float : f64) outs(%scalar : memref<f64>) {
  ^bb0(%in: f64, %out: f64):
    linalg.yield %in : f64
  }
  return
}

// CHECK-NEXT:    func.func @fill_empty_shape(%scalar : memref<f64>) {
// CHECK-NEXT:      %zero_float = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.generic {
// CHECK-NEXT:        bounds = [],
// CHECK-NEXT:        indexing_maps = [
// CHECK-NEXT:          affine_map<() -> ()>,
// CHECK-NEXT:          affine_map<() -> ()>
// CHECK-NEXT:        ],
// CHECK-NEXT:        iterator_types = []
// CHECK-NEXT:      } ins(%zero_float : f64) outs(%scalar : memref<f64>) {
// CHECK-NEXT:      ^bb0(%in : f64, %out : f64):
// CHECK-NEXT:        linalg.yield %in : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

func.func public @dsum(%arg0 : memref<8x16xf64>, %arg1 : memref<8x16xf64>, %arg2 : memref<8x16xf64>) -> memref<8x16xf64> {
    memref_stream.generic {
        bounds = [8, 16],
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : memref<8x16xf64>, memref<8x16xf64>) outs(%arg2 : memref<8x16xf64>) {
    ^bb0(%in : f64, %in_0 : f64, %out : f64):
        %0 = arith.addf %in, %in_0 : f64
        memref_stream.yield %0 : f64
    }
    func.return %arg2 : memref<8x16xf64>
}

// CHECK-NEXT:    func.func public @dsum(%{{.*}} : memref<8x16xf64>, %{{.*}} : memref<8x16xf64>, %{{.*}} : memref<8x16xf64>) -> memref<8x16xf64> {
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%{{.*}}, %{{.*}} : memref<8x16xf64>, memref<8x16xf64>) outs(%{{.*}} : memref<8x16xf64>) {
// CHECK-NEXT:      ^bb{{.*}}(%{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.writable<f64>):
// CHECK-NEXT:        memref_stream.generic {
// CHECK-NEXT:          bounds = [8, 16],
// CHECK-NEXT:          indexing_maps = [
// CHECK-NEXT:            affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:            affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:            affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:          ],
// CHECK-NEXT:          iterator_types = ["parallel", "parallel"]
// CHECK-NEXT:        } ins(%{{.*}}, %{{.*}} : !memref_stream.readable<f64>, !memref_stream.readable<f64>) outs(%{{.*}} : !memref_stream.writable<f64>) {
// CHECK-NEXT:        ^bb{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:          memref_stream.yield %{{.*}} : f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<8x16xf64>
// CHECK-NEXT:    }

func.func public @relu(%arg0 : memref<16x16xf64>, %arg1 : memref<16x16xf64>) -> memref<16x16xf64> {
    %cst = arith.constant 0.000000e+00 : f64
    memref_stream.generic {
        bounds = [16, 16],
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : memref<16x16xf64>) outs(%arg1 : memref<16x16xf64>) {
    ^bb1(%in_1 : f64, %out_1 : f64):
        %1 = arith.maximumf %in_1, %cst : f64
        memref_stream.yield %1 : f64
    }
    func.return %arg1 : memref<16x16xf64>
}

// CHECK-NEXT:    func.func public @relu(%arg0 : memref<16x16xf64>, %arg1 : memref<16x16xf64>) -> memref<16x16xf64> {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%arg0 : memref<16x16xf64>) outs(%arg1 : memref<16x16xf64>) {
// CHECK-NEXT:      ^bb0(%0 : !memref_stream.readable<f64>, %1 : !memref_stream.writable<f64>):
// CHECK-NEXT:        memref_stream.generic {
// CHECK-NEXT:          bounds = [16, 16],
// CHECK-NEXT:          indexing_maps = [
// CHECK-NEXT:            affine_map<(d0, d1) -> (d0, d1)>,
// CHECK-NEXT:            affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:          ],
// CHECK-NEXT:          iterator_types = ["parallel", "parallel"]
// CHECK-NEXT:        } ins(%0 : !memref_stream.readable<f64>) outs(%1 : !memref_stream.writable<f64>) {
// CHECK-NEXT:        ^bb1(%in : f64, %out : f64):
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
        bounds = [16, 16],
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
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } outs(%{{.*}} : memref<16x16xf64>) {
// CHECK-NEXT:      ^bb{{.*}}(%{{.*}} : !memref_stream.writable<f64>):
// CHECK-NEXT:        memref_stream.generic {
// CHECK-NEXT:          bounds = [16, 16],
// CHECK-NEXT:          indexing_maps = [
// CHECK-NEXT:            affine_map<(d0, d1) -> ()>,
// CHECK-NEXT:            affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:          ],
// CHECK-NEXT:          iterator_types = ["parallel", "parallel"]
// CHECK-NEXT:        } ins(%{{.*}} : f64) outs(%{{.*}} : !memref_stream.writable<f64>) {
// CHECK-NEXT:        ^bb{{.*}}(%{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:          memref_stream.yield %{{.*}} : f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

    func.func public @conv_2d_nchw_fchw_d1_s1_3x3(
        %X : memref<1x1x8x8xf64>,
        %Y : memref<1x1x3x3xf64>,
        %Z : memref<1x1x6x6xf64>
    ) {
    %zero_float = arith.constant 0.000000e+00 : f64
    memref_stream.generic {
      bounds = [1, 1, 6, 6, 1, 3, 3],
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>,
        affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
    } ins(%X, %Y : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) outs(%Z : memref<1x1x6x6xf64>) inits(%zero_float : f64) {
    ^bb0(%x : f64, %y : f64, %acc : f64):
      %prod = arith.mulf %x, %y fastmath<fast> : f64
      %res = arith.addf %prod, %acc fastmath<fast> : f64
      memref_stream.yield %res : f64
    }

    func.return
  }

// CHECK-NEXT:    func.func public @conv_2d_nchw_fchw_d1_s1_3x3(%X : memref<1x1x8x8xf64>, %Y : memref<1x1x3x3xf64>, %Z : memref<1x1x6x6xf64>) {
// CHECK-NEXT:      %zero_float = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [1, 1, 6, 6, 1, 3, 3], index_map = (d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, (d2 + d5), (d3 + d6))>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [1, 1, 6, 6, 1, 3, 3], index_map = (d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [1, 1, 6, 6], index_map = (d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%X, %Y : memref<1x1x8x8xf64>, memref<1x1x3x3xf64>) outs(%Z : memref<1x1x6x6xf64>) {
// CHECK-NEXT:      ^bb0(%{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.writable<f64>):
// CHECK-NEXT:        memref_stream.generic {
// CHECK-NEXT:          bounds = [1, 1, 6, 6, 1, 3, 3],
// CHECK-NEXT:          indexing_maps = [
// CHECK-NEXT:            affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, (d2 + d5), (d3 + d6))>,
// CHECK-NEXT:            affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>,
// CHECK-NEXT:            affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT:          ],
// CHECK-NEXT:          iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
// CHECK-NEXT:        } ins(%{{.*}}, %{{.*}} : !memref_stream.readable<f64>, !memref_stream.readable<f64>) outs(%{{.*}} : !memref_stream.writable<f64>) inits(%zero_float : f64) {
// CHECK-NEXT:        ^bb{{\d+}}(%x : f64, %y : f64, %acc : f64):
// CHECK-NEXT:          %prod = arith.mulf %x, %y fastmath<fast> : f64
// CHECK-NEXT:          %res = arith.addf %prod, %acc fastmath<fast> : f64
// CHECK-NEXT:          memref_stream.yield %res : f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

    func.func public @used_only(
        %X : memref<2xf64>,
        %Y : memref<2xf64>,
        %Z : memref<2xf64>
    ) {
    memref_stream.generic {
      bounds = [2],
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%X, %Y : memref<2xf64>, memref<2xf64>) outs(%Z : memref<2xf64>) {
    ^bb0(%x : f64, %y : f64, %z : f64):
      memref_stream.yield %x : f64
    }

    func.return
  }

// CHECK-NEXT:    func.func public @used_only(%{{.*}} : memref<2xf64>, %{{.*}} : memref<2xf64>, %{{.*}} : memref<2xf64>) {
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [2], index_map = (d0) -> (d0)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [2], index_map = (d0) -> (d0)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%{{.*}} : memref<2xf64>) outs(%{{.*}} : memref<2xf64>) {
// CHECK-NEXT:      ^bb{{.*}}(%{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.writable<f64>):
// CHECK-NEXT:        memref_stream.generic {
// CHECK-NEXT:          bounds = [2],
// CHECK-NEXT:          indexing_maps = [
// CHECK-NEXT:            affine_map<(d0) -> (d0)>,
// CHECK-NEXT:            affine_map<(d0) -> (d0)>,
// CHECK-NEXT:            affine_map<(d0) -> (d0)>
// CHECK-NEXT:          ],
// CHECK-NEXT:          iterator_types = ["parallel"]
// CHECK-NEXT:        } ins(%{{.*}}, %{{.*}} : !memref_stream.readable<f64>, memref<2xf64>) outs(%{{.*}} : !memref_stream.writable<f64>) {
// CHECK-NEXT:        ^bb{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:          memref_stream.yield %{{.*}} : f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }


func.func @interleaved_no_init(%A : memref<3x5xf64>, %B : memref<5x8xf64>, %C : memref<3x8xf64>) -> memref<3x8xf64> {
    memref_stream.generic {
        bounds = [3, 2, 5, 4],
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
            affine_map<(d0, d1, d2, d3) -> (d2, d1 * 4 + d3)>,
            affine_map<(d0, d1, d3) -> (d0, d1 * 4 + d3)>
        ],
        iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
    } ins(%A, %B : memref<3x5xf64>, memref<5x8xf64>) outs(%C : memref<3x8xf64>) {
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
    func.return %C : memref<3x8xf64>
}

// CHECK-NEXT:    func.func @interleaved_no_init(%{{.*}} : memref<3x5xf64>, %{{.*}} : memref<5x8xf64>, %{{.*}} : memref<3x8xf64>) -> memref<3x8xf64> {
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [3, 2, 5, 4], index_map = (d0, d1, d2, d3) -> (d0, d2)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [3, 2, 5, 4], index_map = (d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%{{.*}}, %{{.*}} : memref<3x5xf64>, memref<5x8xf64>) {
// CHECK-NEXT:      ^bb{{.*}}(%{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.readable<f64>):
// CHECK-NEXT:        memref_stream.generic {
// CHECK-NEXT:          bounds = [3, 2, 5, 4],
// CHECK-NEXT:          indexing_maps = [
// CHECK-NEXT:            affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// CHECK-NEXT:            affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// CHECK-NEXT:            affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// CHECK-NEXT:          ],
// CHECK-NEXT:          iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// CHECK-NEXT:        } ins(%{{.*}}, %{{.*}} : !memref_stream.readable<f64>, !memref_stream.readable<f64>) outs(%{{.*}} : memref<3x8xf64>) {
// CHECK-NEXT:        ^bb{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:          %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          memref_stream.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<3x8xf64>
// CHECK-NEXT:    }

func.func @interleaved_init(%A : memref<3x5xf64>, %B : memref<5x8xf64>, %C : memref<3x8xf64>) -> memref<3x8xf64> {
    %zero_float = arith.constant 0.000000e+00 : f64
    memref_stream.generic {
        bounds = [3, 2, 5, 4],
        indexing_maps = [
            affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
            affine_map<(d0, d1, d2, d3) -> (d2, d1 * 4 + d3)>,
            affine_map<(d0, d1, d3) -> (d0, d1 * 4 + d3)>
        ],
        iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
    } ins(%A, %B : memref<3x5xf64>, memref<5x8xf64>) outs(%C : memref<3x8xf64>) inits(%zero_float : f64) {
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
    func.return %C : memref<3x8xf64>
}

// CHECK-NEXT:    func.func @interleaved_init(%{{.*}} : memref<3x5xf64>, %{{.*}} : memref<5x8xf64>, %{{.*}} : memref<3x8xf64>) -> memref<3x8xf64> {
// CHECK-NEXT:      %zero_float = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [3, 2, 5, 4], index_map = (d0, d1, d2, d3) -> (d0, d2)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [3, 2, 5, 4], index_map = (d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [3, 2, 4], index_map = (d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%{{.*}}, %{{.*}} : memref<3x5xf64>, memref<5x8xf64>) outs(%{{.*}} : memref<3x8xf64>) {
// CHECK-NEXT:      ^bb{{.*}}(%{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.writable<f64>):
// CHECK-NEXT:        memref_stream.generic {
// CHECK-NEXT:          bounds = [3, 2, 5, 4],
// CHECK-NEXT:          indexing_maps = [
// CHECK-NEXT:            affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
// CHECK-NEXT:            affine_map<(d0, d1, d2, d3) -> (d2, ((d1 * 4) + d3))>,
// CHECK-NEXT:            affine_map<(d0, d1, d2) -> (d0, ((d1 * 4) + d2))>
// CHECK-NEXT:          ],
// CHECK-NEXT:          iterator_types = ["parallel", "parallel", "reduction", "interleaved"]
// CHECK-NEXT:        } ins(%{{.*}}, %{{.*}} : !memref_stream.readable<f64>, !memref_stream.readable<f64>) outs(%{{.*}} : !memref_stream.writable<f64>) inits(%{{.*}} : f64) {
// CHECK-NEXT:        ^bb{{.*}}(%{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64, %{{.*}} : f64):
// CHECK-NEXT:          %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.mulf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          %{{.*}} = arith.addf %{{.*}}, %{{.*}} fastmath<fast> : f64
// CHECK-NEXT:          memref_stream.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f64, f64, f64, f64
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<3x8xf64>
// CHECK-NEXT:    }

func.func public @ssum(
  %X: memref<8x16xf32>,
  %Y: memref<8x16xf32>,
  %Z: memref<8x16xf32>
) {
  memref_stream.generic {
    bounds = [8, 8],
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, 2 * d1)>,
      affine_map<(d0, d1) -> (d0, 2 * d1)>,
      affine_map<(d0, d1) -> (d0, 2 * d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%X, %Y : memref<8x16xf32>, memref<8x16xf32>) outs(%Z : memref<8x16xf32>) {
  ^bb1(%in : vector<2xf32>, %in_1 : vector<2xf32>, %out : vector<2xf32>):
    %3 = arith.addf %in, %in_1 : vector<2xf32>
    memref_stream.yield %3 : vector<2xf32>
  }
  func.return
}

// CHECK-NEXT:    func.func public @ssum(%X : memref<8x16xf32>, %Y : memref<8x16xf32>, %Z : memref<8x16xf32>) {
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [8, 8], index_map = (d0, d1) -> (d0, (d1 * 2))>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [8, 8], index_map = (d0, d1) -> (d0, (d1 * 2))>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [8, 8], index_map = (d0, d1) -> (d0, (d1 * 2))>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%X, %Y : memref<8x16xf32>, memref<8x16xf32>) outs(%Z : memref<8x16xf32>) {
// CHECK-NEXT:      ^bb0(%0 : !memref_stream.readable<vector<2xf32>>, %1 : !memref_stream.readable<vector<2xf32>>, %2 : !memref_stream.writable<vector<2xf32>>):
// CHECK-NEXT:        memref_stream.generic {
// CHECK-NEXT:          bounds = [8, 8],
// CHECK-NEXT:          indexing_maps = [
// CHECK-NEXT:            affine_map<(d0, d1) -> (d0, (d1 * 2))>,
// CHECK-NEXT:            affine_map<(d0, d1) -> (d0, (d1 * 2))>,
// CHECK-NEXT:            affine_map<(d0, d1) -> (d0, (d1 * 2))>
// CHECK-NEXT:          ],
// CHECK-NEXT:          iterator_types = ["parallel", "parallel"]
// CHECK-NEXT:        } ins(%0, %1 : !memref_stream.readable<vector<2xf32>>, !memref_stream.readable<vector<2xf32>>) outs(%2 : !memref_stream.writable<vector<2xf32>>) {
// CHECK-NEXT:        ^bb1(%in : vector<2xf32>, %in_1 : vector<2xf32>, %out : vector<2xf32>):
// CHECK-NEXT:          %3 = arith.addf %in, %in_1 : vector<2xf32>
// CHECK-NEXT:          memref_stream.yield %3 : vector<2xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }

// CHECK-NEXT:  }
