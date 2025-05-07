// RUN: xdsl-opt -p convert-memref-stream-to-loops %s | filecheck %s

// CHECK:       builtin.module {

  func.func public @dsum(%arg0 : memref<8x16xf64>, %arg1 : memref<8x16xf64>, %arg2 : memref<8x16xf64>) -> memref<8x16xf64> {
    memref_stream.streaming_region {
      patterns = [
        #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>,
        #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>,
        #memref_stream.stride_pattern<ub = [8, 16], index_map = (d0, d1) -> (d0, d1)>
    ]
  } ins(%arg0, %arg1 : memref<8x16xf64>, memref<8x16xf64>) outs(%arg2 : memref<8x16xf64>) {
    ^0(%0 : !memref_stream.readable<f64>, %1 : !memref_stream.readable<f64>, %2 : !memref_stream.writable<f64>):
      memref_stream.generic {bounds = [8, 16], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %1 : !memref_stream.readable<f64>, !memref_stream.readable<f64>) outs(%2 : !memref_stream.writable<f64>) {
      ^1(%in : f64, %in_0 : f64, %out : f64):
        %3 = arith.addf %in, %in_0 : f64
        memref_stream.yield %3 : f64
      }
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
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.writable<f64>):
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
    ^2(%4 : !memref_stream.readable<f64>, %5 : !memref_stream.writable<f64>):
      memref_stream.generic {bounds = [16, 16], indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%4 : !memref_stream.readable<f64>) outs(%5 : !memref_stream.writable<f64>) {
      ^3(%in_1 : f64, %out_1 : f64):
        %6 = arith.maximumf %in_1, %cst : f64
        memref_stream.yield %6 : f64
      }
    }
    func.return %arg1_1 : memref<16x16xf64>
  }
// CHECK-NEXT:    func.func public @relu(%{{.*}} : memref<16x16xf64>, %{{.*}} : memref<16x16xf64>) -> memref<16x16xf64> {
// CHECK-NEXT:      %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%{{.*}} : memref<16x16xf64>) outs(%{{.*}} : memref<16x16xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.writable<f64>):
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
  ^3(%7 : !memref_stream.writable<f64>):
    memref_stream.generic {
        bounds = [16, 16],
        indexing_maps = [
            affine_map<(d0, d1) -> ()>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%zero : f64) outs(%7 : !memref_stream.writable<f64>) {
    ^4(%in: f64, %out: f64):
        memref_stream.yield %in : f64
    }
  }
  func.return %arg0 : memref<16x16xf64>
}

// CHECK-NEXT:    func.func public @fill(%{{.*}} : memref<16x16xf64>) -> memref<16x16xf64> {
// CHECK-NEXT:      %{{.*}} = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [16, 16], index_map = (d0, d1) -> (d0, d1)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } outs(%{{.*}} : memref<16x16xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !memref_stream.writable<f64>):
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
    ^0(%0 : !memref_stream.readable<f64>, %1 : !memref_stream.readable<f64>):
      memref_stream.generic {
        bounds = [4, 3, 2],
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%0, %1 : !memref_stream.readable<f64>, !memref_stream.readable<f64>) outs(%C : memref<4x3xf64>) {
      ^1(%a : f64, %b : f64, %acc_old : f64):
        %prod = arith.mulf %a, %b : f64
        %acc_new = arith.addf %acc_old, %prod : f64
        memref_stream.yield %acc_new : f64
      }
    }
    func.return %C : memref<4x3xf64>
}
// CHECK-NEXT:    func.func @main(%{{.*}} : memref<4x2xf64>, %{{.*}} : memref<2x3xf64>, %{{.*}} : memref<4x3xf64>) -> memref<4x3xf64> {
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d0, d2)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d2, d1)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%{{.*}}, %{{.*}} : memref<4x2xf64>, memref<2x3xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.readable<f64>):
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
    ^0(%0 : !memref_stream.readable<f64>):
      memref_stream.generic {
        bounds = [2, 3],
        indexing_maps = [
          affine_map<(d0, d1) -> (d0 * 3 + d1)>,
          affine_map<(d0, d1) -> ()>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%0 : !memref_stream.readable<f64>) outs(%B : memref<f64>) {
      ^1(%a : f64, %acc_old : f64):
        %acc_new = arith.addf %acc_old, %a : f64
        memref_stream.yield %acc_new : f64
      }
    }
    func.return %B : memref<f64>
}
// CHECK-NEXT:    func.func @elide_affine(%{{.*}} : memref<6xf64>, %{{.*}} : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [2, 3], index_map = (d0, d1) -> (((d0 * 3) + d1))>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%{{.*}} : memref<6xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !memref_stream.readable<f64>):
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
    ^0(%0 : !memref_stream.readable<f64>):
      memref_stream.generic {
        bounds = [2, 3, 4],
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<() -> ()>
        ],
        iterator_types = ["reduction", "reduction", "reduction"]
      } ins(%0 : !memref_stream.readable<f64>) outs(%B : memref<f64>) {
      ^1(%a : f64, %acc_old : f64):
        %acc_new = arith.addf %acc_old, %a : f64
        memref_stream.yield %acc_new : f64
      }
    }
    func.return %B : memref<f64>
}

// CHECK-NEXT:    func.func @nested_imperfect(%{{.*}} : memref<2x3x4xf64>, %{{.*}} : memref<f64>) -> memref<f64> {
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [2, 3, 4], index_map = (d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%{{.*}} : memref<2x3x4xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !memref_stream.readable<f64>):
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
    ^0(%0 : !memref_stream.readable<f64>, %1 : !memref_stream.readable<f64>):
      memref_stream.generic {
        bounds = [4, 3, 2],
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%0, %1 : !memref_stream.readable<f64>, !memref_stream.readable<f64>) outs(%C : memref<4x3xf64>) inits(%zero_float : f64) {
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
// CHECK-NEXT:      memref_stream.streaming_region {
// CHECK-NEXT:        patterns = [
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d0, d2)>,
// CHECK-NEXT:          #memref_stream.stride_pattern<ub = [4, 3, 2], index_map = (d0, d1, d2) -> (d2, d1)>
// CHECK-NEXT:        ]
// CHECK-NEXT:      } ins(%{{.*}}, %{{.*}} : memref<4x2xf64>, memref<2x3xf64>) {
// CHECK-NEXT:      ^{{.*}}(%{{.*}} : !memref_stream.readable<f64>, %{{.*}} : !memref_stream.readable<f64>):
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

// CHECK-NEXT:    func.func @interleaved_no_init(%A : memref<3x5xf64>, %B : memref<5x8xf64>, %C : memref<3x8xf64>) -> memref<3x8xf64> {
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 1 : index
// CHECK-NEXT:      %2 = arith.constant 2 : index
// CHECK-NEXT:      %3 = arith.constant 3 : index
// CHECK-NEXT:      %4 = arith.constant 3 : index
// CHECK-NEXT:      %5 = arith.constant 2 : index
// CHECK-NEXT:      %6 = arith.constant 5 : index
// CHECK-NEXT:      %7 = arith.constant 0 : index
// CHECK-NEXT:      %8 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %9 = %7 to %4 step %8 {
// CHECK-NEXT:        scf.for %10 = %7 to %5 step %8 {
// CHECK-NEXT:          %11 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %0)
// CHECK-NEXT:          %12 = memref.load %C[%9, %11] : memref<3x8xf64>
// CHECK-NEXT:          %13 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %1)
// CHECK-NEXT:          %14 = memref.load %C[%9, %13] : memref<3x8xf64>
// CHECK-NEXT:          %15 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %2)
// CHECK-NEXT:          %16 = memref.load %C[%9, %15] : memref<3x8xf64>
// CHECK-NEXT:          %17 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %3)
// CHECK-NEXT:          %18 = memref.load %C[%9, %17] : memref<3x8xf64>
// CHECK-NEXT:          %19, %20, %21, %22 = scf.for %23 = %7 to %6 step %8 iter_args(%c0 = %12, %c1 = %14, %c2 = %16, %c3 = %18) -> (f64, f64, f64, f64) {
// CHECK-NEXT:            %a0 = memref.load %A[%9, %23] : memref<3x5xf64>
// CHECK-NEXT:            %a1 = memref.load %A[%9, %23] : memref<3x5xf64>
// CHECK-NEXT:            %a2 = memref.load %A[%9, %23] : memref<3x5xf64>
// CHECK-NEXT:            %a3 = memref.load %A[%9, %23] : memref<3x5xf64>
// CHECK-NEXT:            %24 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %0)
// CHECK-NEXT:            %b0 = memref.load %B[%23, %24] : memref<5x8xf64>
// CHECK-NEXT:            %25 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %1)
// CHECK-NEXT:            %b1 = memref.load %B[%23, %25] : memref<5x8xf64>
// CHECK-NEXT:            %26 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %2)
// CHECK-NEXT:            %b2 = memref.load %B[%23, %26] : memref<5x8xf64>
// CHECK-NEXT:            %27 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %3)
// CHECK-NEXT:            %b3 = memref.load %B[%23, %27] : memref<5x8xf64>
// CHECK-NEXT:            %prod0 = arith.mulf %a0, %b0 fastmath<fast> : f64
// CHECK-NEXT:            %prod1 = arith.mulf %a1, %b1 fastmath<fast> : f64
// CHECK-NEXT:            %prod2 = arith.mulf %a2, %b2 fastmath<fast> : f64
// CHECK-NEXT:            %prod3 = arith.mulf %a3, %b3 fastmath<fast> : f64
// CHECK-NEXT:            %res0 = arith.addf %prod0, %c0 fastmath<fast> : f64
// CHECK-NEXT:            %res1 = arith.addf %prod1, %c1 fastmath<fast> : f64
// CHECK-NEXT:            %res2 = arith.addf %prod2, %c2 fastmath<fast> : f64
// CHECK-NEXT:            %res3 = arith.addf %prod3, %c3 fastmath<fast> : f64
// CHECK-NEXT:            scf.yield %res0, %res1, %res2, %res3 : f64, f64, f64, f64
// CHECK-NEXT:          }
// CHECK-NEXT:          %28 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %0)
// CHECK-NEXT:          memref.store %19, %C[%9, %28] : memref<3x8xf64>
// CHECK-NEXT:          %29 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %1)
// CHECK-NEXT:          memref.store %20, %C[%9, %29] : memref<3x8xf64>
// CHECK-NEXT:          %30 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %2)
// CHECK-NEXT:          memref.store %21, %C[%9, %30] : memref<3x8xf64>
// CHECK-NEXT:          %31 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %3)
// CHECK-NEXT:          memref.store %22, %C[%9, %31] : memref<3x8xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %C : memref<3x8xf64>
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

// CHECK-NEXT:    func.func @interleaved_init(%A : memref<3x5xf64>, %B : memref<5x8xf64>, %C : memref<3x8xf64>) -> memref<3x8xf64> {
// CHECK-NEXT:      %zero_float = arith.constant 0.000000e+00 : f64
// CHECK-NEXT:      %0 = arith.constant 0 : index
// CHECK-NEXT:      %1 = arith.constant 1 : index
// CHECK-NEXT:      %2 = arith.constant 2 : index
// CHECK-NEXT:      %3 = arith.constant 3 : index
// CHECK-NEXT:      %4 = arith.constant 3 : index
// CHECK-NEXT:      %5 = arith.constant 2 : index
// CHECK-NEXT:      %6 = arith.constant 5 : index
// CHECK-NEXT:      %7 = arith.constant 0 : index
// CHECK-NEXT:      %8 = arith.constant 1 : index
// CHECK-NEXT:      scf.for %9 = %7 to %4 step %8 {
// CHECK-NEXT:        scf.for %10 = %7 to %5 step %8 {
// CHECK-NEXT:          %11, %12, %13, %14 = scf.for %15 = %7 to %6 step %8 iter_args(%c0 = %zero_float, %c1 = %zero_float, %c2 = %zero_float, %c3 = %zero_float) -> (f64, f64, f64, f64) {
// CHECK-NEXT:            %a0 = memref.load %A[%9, %15] : memref<3x5xf64>
// CHECK-NEXT:            %a1 = memref.load %A[%9, %15] : memref<3x5xf64>
// CHECK-NEXT:            %a2 = memref.load %A[%9, %15] : memref<3x5xf64>
// CHECK-NEXT:            %a3 = memref.load %A[%9, %15] : memref<3x5xf64>
// CHECK-NEXT:            %16 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %0)
// CHECK-NEXT:            %b0 = memref.load %B[%15, %16] : memref<5x8xf64>
// CHECK-NEXT:            %17 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %1)
// CHECK-NEXT:            %b1 = memref.load %B[%15, %17] : memref<5x8xf64>
// CHECK-NEXT:            %18 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %2)
// CHECK-NEXT:            %b2 = memref.load %B[%15, %18] : memref<5x8xf64>
// CHECK-NEXT:            %19 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %3)
// CHECK-NEXT:            %b3 = memref.load %B[%15, %19] : memref<5x8xf64>
// CHECK-NEXT:            %prod0 = arith.mulf %a0, %b0 fastmath<fast> : f64
// CHECK-NEXT:            %prod1 = arith.mulf %a1, %b1 fastmath<fast> : f64
// CHECK-NEXT:            %prod2 = arith.mulf %a2, %b2 fastmath<fast> : f64
// CHECK-NEXT:            %prod3 = arith.mulf %a3, %b3 fastmath<fast> : f64
// CHECK-NEXT:            %res0 = arith.addf %prod0, %c0 fastmath<fast> : f64
// CHECK-NEXT:            %res1 = arith.addf %prod1, %c1 fastmath<fast> : f64
// CHECK-NEXT:            %res2 = arith.addf %prod2, %c2 fastmath<fast> : f64
// CHECK-NEXT:            %res3 = arith.addf %prod3, %c3 fastmath<fast> : f64
// CHECK-NEXT:            scf.yield %res0, %res1, %res2, %res3 : f64, f64, f64, f64
// CHECK-NEXT:          }
// CHECK-NEXT:          %20 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %0)
// CHECK-NEXT:          memref.store %11, %C[%9, %20] : memref<3x8xf64>
// CHECK-NEXT:          %21 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %1)
// CHECK-NEXT:          memref.store %12, %C[%9, %21] : memref<3x8xf64>
// CHECK-NEXT:          %22 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %2)
// CHECK-NEXT:          memref.store %13, %C[%9, %22] : memref<3x8xf64>
// CHECK-NEXT:          %23 = affine.apply affine_map<(d0, d1) -> (((d0 * 4) + d1))> (%10, %3)
// CHECK-NEXT:          memref.store %14, %C[%9, %23] : memref<3x8xf64>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %C : memref<3x8xf64>
// CHECK-NEXT:    }

// CHECK-NEXT:  }
