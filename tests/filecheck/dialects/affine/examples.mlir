// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({

  // Sum elements of a vector

  "func.func"() ({
  ^bb0(%ref: memref<128xi32>):
    %const0 = "arith.constant"() {"value" = 0 : i32} : () -> i32
    %r = affine.for %i = 0 to 256 iter_args(%sum = %const0) -> (i32) {
      %val = "memref.load"(%ref, %i) : (memref<128xi32>, index) -> i32
      %res = "arith.addi"(%sum, %val) : (i32, i32) -> i32
      affine.yield %res : i32
    }
    func.return %r : i32
  }) {"sym_name" = "sum_vec", "function_type" = (memref<128xi32>) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:         func.func private @sum_vec(%{{.*}}: memref<128xi32>) -> i32 {
// CHECK-NEXT:      %{{.*}} = arith.constant 0 : i32
// CHECK-NEXT:      %{{.*}} = affine.for %{{.*}} = 0 to 256 iter_args(%{{.*}} = %{{.*}}) -> (i32) {
// CHECK-NEXT:        %{{.*}} = memref.load %{{.*}}[%{{.*}}] : memref<128xi32>
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT:        affine.yield %{{.*}} : i32
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : i32
// CHECK-NEXT:    }


  // Matrix multiplication

  "func.func"() ({
  ^bb2(%0: memref<256x256xf32>, %1: memref<256x256xf32>, %2: memref<256x256xf32>):
    affine.for %3 = 0 to 256 {
      affine.for %4 = 0 to 256 {
        affine.for %5 = 0 to 256 {
          %6 = "memref.load"(%0, %3, %5) : (memref<256x256xf32>, index, index) -> f32
          %7 = "memref.load"(%1, %5, %4) : (memref<256x256xf32>, index, index) -> f32
          %8 = "memref.load"(%2, %3, %4) : (memref<256x256xf32>, index, index) -> f32
          %9 = "arith.mulf"(%6, %7) : (f32, f32) -> f32
          %10 = "arith.addf"(%8, %9) : (f32, f32) -> f32
          "memref.store"(%10, %2, %3, %4) : (f32, memref<256x256xf32>, index, index) -> ()
        }
      }
    }
    "func.return"(%2) : (memref<256x256xf32>) -> ()
  }) {"sym_name" = "affine_mm", "function_type" = (memref<256x256xf32>, memref<256x256xf32>, memref<256x256xf32>) -> memref<256x256xf32>, "sym_visibility" = "private"} : () -> ()

// CHECK:         func.func private @affine_mm(%{{.*}}: memref<256x256xf32>, %{{.*}}: memref<256x256xf32>, %{{.*}}: memref<256x256xf32>) -> memref<256x256xf32> {
// CHECK-NEXT:      affine.for %{{.*}} = 0 to 256 {
// CHECK-NEXT:        affine.for %{{.*}} = 0 to 256 {
// CHECK-NEXT:          affine.for %{{.*}} = 0 to 256 {
// CHECK-NEXT:            %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<256x256xf32>
// CHECK-NEXT:            %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<256x256xf32>
// CHECK-NEXT:            %{{.*}} = memref.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<256x256xf32>
// CHECK-NEXT:            %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:            %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT:            memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<256x256xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      func.return %{{.*}} : memref<256x256xf32>
// CHECK-NEXT:    }

}) : () -> ()
