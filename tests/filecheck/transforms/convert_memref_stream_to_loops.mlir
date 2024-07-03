// RUN: xdsl-opt -p convert-memref-stream-to-loops --split-input-file --verify-diagnostics %s | filecheck %s

// CHECK:       builtin.module {

  func.func public @dsum(%arg0 : memref<8x16xf64>, %arg1 : memref<8x16xf64>, %arg2 : memref<8x16xf64>) -> memref<8x16xf64> {
    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>,
        #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>,
        #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>
    ]
  } ins(%arg0, %arg1 : memref<8x16xf64>, memref<8x16xf64>) outs(%arg2 : memref<8x16xf64>) {
    ^0(%0 : !stream.readable<f64>, %1 : !stream.readable<f64>, %2 : !stream.writable<f64>):
      memref_stream.generic {bounds = [8, 16], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : !stream.readable<f64>, !stream.readable<f64>) outs(%2 : !stream.writable<f64>) {
      ^1(%in : f64, %in_0 : f64, %out : f64):
        %3 = arith.addf %in, %in_0 : f64
        memref_stream.yield %3 : f64
      }
    }
    func.return %arg2 : memref<8x16xf64>
  }
// CHECK-NEXT:    func.func public @dsum(%{{.*}} : memref<8x16xf64>, %{{.*}} : memref<8x16xf64>, %{{.*}} : memref<8x16xf64>) -> memref<8x16xf64> {
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>, #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>, #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>]} ins(%{{.*}}, %{{.*}} : memref<8x16xf64>, memref<8x16xf64>) outs(%{{.*}} : memref<8x16xf64>) {
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
    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>,
        #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>,
        #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>
    ]
  } ins(%arg0_1 : memref<16x16xf64>) outs(%arg1_1 : memref<16x16xf64>) {
    ^2(%4 : !stream.readable<f64>, %5 : !stream.writable<f64>):
      memref_stream.generic {bounds = [16, 16], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : !stream.readable<f64>) outs(%5 : !stream.writable<f64>) {
      ^3(%in_1 : f64, %out_1 : f64):
        %6 = arith.maximumf %in_1, %cst : f64
        memref_stream.yield %6 : f64
      }
    }
    func.return %arg1_1 : memref<16x16xf64>
  }
// CHECK-NEXT:    func.func public @relu(%{{.*}} : memref<16x16xf64>, %{{.*}} : memref<16x16xf64>) -> memref<16x16xf64> {
// CHECK-NEXT:      %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>, #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>, #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>]} ins(%{{.*}} : memref<16x16xf64>) outs(%{{.*}} : memref<16x16xf64>) {
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

func.func public @fill(%arg0 : memref<16x16xf64>) -> memref<16x16xf64> {
  // Scalar argument
  %zero = arith.constant 0.0 : f64
  memref_stream.streaming_region {
    patterns = [
      #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>
    ]
  } outs(%arg0 : memref<16x16xf64>) {
  ^3(%7 : !stream.writable<f64>):
    memref_stream.generic {
        bounds = [16, 16],
        indexing_maps = [
            affine_map<(d0, d1) -> ()>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%zero : f64) outs(%7 : !stream.writable<f64>) {
    ^4(%in: f64, %out: f64):
        memref_stream.yield %in : f64
    }
  }
  func.return %arg0 : memref<16x16xf64>
}

// CHECK-NEXT:    func.func public @fill(%{{.*}} : memref<16x16xf64>) -> memref<16x16xf64> {
// CHECK-NEXT:      %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>]} outs(%{{.*}} : memref<16x16xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stream.writable<f64>):
// CHECK-NEXT:        %{{.*}} = arith.constant 16 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 16 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:            memref_stream.write %{{.*}} to %{{.*}} : f64
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<16x16xf64>
// CHECK-NEXT:    }

func.func @main(%A : memref<4x2xf64>, %B : memref<2x3xf64>, %C : memref<4x3xf64>) -> memref<4x3xf64> {
    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d0, d2)>,
        #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d2, d1)>
      ]
    } ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) {
    ^0(%0 : !stream.readable<f64>, %1 : !stream.readable<f64>):
      memref_stream.generic {
        bounds = [4, 3, 2],
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%0, %1 : !stream.readable<f64>, !stream.readable<f64>) outs(%C : memref<4x3xf64>) {
      ^1(%a : f64, %b : f64, %acc_old : f64):
        %prod = arith.mulf %a, %b : f64
        %acc_new = arith.addf %acc_old, %prod : f64
        memref_stream.yield %acc_new : f64
      }
    }
    func.return %C : memref<4x3xf64>
}
// CHECK-NEXT:    func.func @main(%{{.*}} : memref<4x2xf64>, %{{.*}} : memref<2x3xf64>, %{{.*}} : memref<4x3xf64>) -> memref<4x3xf64> {
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d0, d2)>, #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d2, d1)>]} ins(%{{.*}}, %{{.*}} : memref<4x2xf64>, memref<2x3xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stream.readable<f64>, %{{.*}} : !stream.readable<f64>):
// CHECK-NEXT:        %{{.*}} = arith.constant 4 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:            %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x3xf64>
// CHECK-NEXT:            %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f64) {
// CHECK-NEXT:              %{{.*}} = memref_stream.read from %{{.*}} : f64
// CHECK-NEXT:              %{{.*}} = memref_stream.read from %{{.*}} : f64
// CHECK-NEXT:              %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:              %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:              scf.yield %{{.*}} : f64
// CHECK-NEXT:            }
// CHECK-NEXT:            memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x3xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<4x3xf64>
// CHECK-NEXT:    }

func.func @elide_affine(%A : memref<6xf64>, %B : memref<f64>) -> memref<f64> {
    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [2, 3], index_map = (d0, d1) -> (d0 * 3 + d1)>
      ]
    } ins(%A : memref<6xf64>) {
    ^0(%0 : !stream.readable<f64>):
      memref_stream.generic {
        bounds = [2, 3],
        indexing_maps = [
          affine_map<(d0, d1) -> (d0 * 3 + d1)>,
          affine_map<(d0, d1) -> ()>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%0 : !stream.readable<f64>) outs(%B : memref<f64>) {
      ^1(%a : f64, %acc_old : f64):
        %acc_new = arith.addf %acc_old, %a : f64
        memref_stream.yield %acc_new : f64
      }
    }
    func.return %B : memref<f64>
}
// CHECK-NEXT:    func.func @elide_affine(%{{.*}} : memref<6xf64>, %{{.*}} : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [2, 3], index_map = (d0, d1) -> (((d0 * 3) + d1))>]} ins(%{{.*}} : memref<6xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stream.readable<f64>):
// CHECK-NEXT:        %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:            %{{.*}} = memref_stream.read from %{{.*}} : f64
// CHECK-NEXT:            %{{.*}} = memref.load %{{.*}}[] : memref<f64>
// CHECK-NEXT:            %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:            memref.store %{{.*}}, %{{.*}}[] : memref<f64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<f64>
// CHECK-NEXT:    }

func.func @nested_imperfect(%A : memref<2x3x4xf64>, %B : memref<f64>) -> memref<f64> {
    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [2, 3, 4], index_map = (d0, d1, d2) -> (d0, d1, d2)>
      ]
    } ins(%A : memref<2x3x4xf64>) {
    ^0(%0 : !stream.readable<f64>):
      memref_stream.generic {
        bounds = [2, 3, 4],
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<() -> ()>
        ],
        iterator_types = ["reduction", "reduction", "reduction"]
      } ins(%0 : !stream.readable<f64>) outs(%B : memref<f64>) {
      ^1(%a : f64, %acc_old : f64):
        %acc_new = arith.addf %acc_old, %a : f64
        memref_stream.yield %acc_new : f64
      }
    }
    func.return %B : memref<f64>
}

// CHECK-NEXT:    func.func @nested_imperfect(%{{.*}} : memref<2x3x4xf64>, %{{.*}} : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [2, 3, 4], index_map = (d0, d1, d2) -> (d0, d1, d2)>]} ins(%{{.*}} : memref<2x3x4xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stream.readable<f64>):
// CHECK-NEXT:        %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 4 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:        %{{.*}} = memref.load %{{.*}}[] : memref<f64>
// CHECK-NEXT:        %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f64) {
// CHECK-NEXT:          %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f64) {
// CHECK-NEXT:            %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f64) {
// CHECK-NEXT:              %{{.*}} = memref_stream.read from %{{.*}} : f64
// CHECK-NEXT:              %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:              scf.yield %{{.*}} : f64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %{{.*}} : f64
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %{{.*}} : f64
// CHECK-NEXT:        }
// CHECK-NEXT:        memref.store %{{.*}}, %{{.*}}[] : memref<f64>
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<f64>
// CHECK-NEXT:    }

func.func @main_inits(%A : memref<4x2xf64>, %B : memref<2x3xf64>, %C : memref<4x3xf64>) -> memref<4x3xf64> {
    %zero_float = arith.constant 0.000000e+00 : f64
    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d0, d2)>,
        #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d2, d1)>
      ]
    } ins(%A, %B : memref<4x2xf64>, memref<2x3xf64>) {
    ^0(%0 : !stream.readable<f64>, %1 : !stream.readable<f64>):
      memref_stream.generic {
        bounds = [4, 3, 2],
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%0, %1 : !stream.readable<f64>, !stream.readable<f64>) outs(%C : memref<4x3xf64>) inits(%zero_float : f64) {
      ^1(%a : f64, %b : f64, %acc_old : f64):
        %prod = arith.mulf %a, %b : f64
        %acc_new = arith.addf %acc_old, %prod : f64
        memref_stream.yield %acc_new : f64
      }
    }
    func.return %C : memref<4x3xf64>
}
// CHECK-NEXT:    func.func @main_inits(%{{.*}} : memref<4x2xf64>, %{{.*}} : memref<2x3xf64>, %{{.*}} : memref<4x3xf64>) -> memref<4x3xf64> {
// CHECK-NEXT:      %zero_float = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {patterns = [#memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d0, d2)>, #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d2, d1)>]} ins(%{{.*}}, %{{.*}} : memref<4x2xf64>, memref<2x3xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !stream.readable<f64>, %{{.*}} : !stream.readable<f64>):
// CHECK-NEXT:        %{{.*}} = arith.constant 4 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 3 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:        %{{.*}} = arith.constant 1 : index
// CHECK-NEXT:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:            %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %zero_float) -> (f64) {
// CHECK-NEXT:              %{{.*}} = memref_stream.read from %{{.*}} : f64
// CHECK-NEXT:              %{{.*}} = memref_stream.read from %{{.*}} : f64
// CHECK-NEXT:              %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:              %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f64
// CHECK-NEXT:              scf.yield %{{.*}} : f64
// CHECK-NEXT:            }
// CHECK-NEXT:            memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x3xf64>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<4x3xf64>
// CHECK-NEXT:    }

// CHECK-NEXT:  }

// -----

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
    ^1(
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

// CHECK: Error while applying pattern: Cannot yet lower interleaved iterators

// -----

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
    ^1(
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

// CHECK: Error while applying pattern: Cannot yet lower interleaved iterators
