// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({

  // Sum elements of a vector

  "func.func"() ({
  ^0(%ref : memref<128xi32>):
    %const0 = "arith.constant"() {"value" = 0 : i32} : () -> i32
    %r = "affine.for"(%const0) ({
    ^1(%i : index, %sum : i32):
      %val = "memref.load"(%ref, %i) : (memref<128xi32>, index) -> i32
      %res = "arith.addi"(%sum, %val) : (i32, i32) -> i32
      "affine.yield"(%res) : (i32) -> ()
    }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : (i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "sum_vec", "function_type" = (memref<128xi32>) -> i32, "sym_visibility" = "private"} : () -> ()

  // CHECK:      "func.func"() ({
  // CHECK-NEXT: ^{{.*}}(%{{.*}} : memref<128xi32>):
  // CHECK-NEXT:   %{{.*}} = "arith.constant"() {"value" = 0 : i32} : () -> i32
  // CHECK-NEXT:   %{{.*}} = "affine.for"(%{{.*}}) ({
  // CHECK-NEXT:   ^{{.*}}(%{{.*}} : index, %{{.*}} : i32):
  // CHECK-NEXT:     %{{.*}} = "memref.load"(%{{.*}}, %{{.*}}) : (memref<128xi32>, index) -> i32
  // CHECK-NEXT:     %{{.*}} = "arith.addi"(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
  // CHECK-NEXT:     "affine.yield"(%{{.*}}) : (i32) -> ()
  // CHECK-NEXT:   }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : (i32) -> i32
  // CHECK-NEXT:   "func.return"(%{{.*}}) : (i32) -> ()
  // CHECK-NEXT: }) {"sym_name" = "sum_vec", "function_type" = (memref<128xi32>) -> i32, "sym_visibility" = "private"} : () -> ()


  // Matrix multiplication

  "func.func"() ({
  ^2(%0 : memref<256x256xf32>, %1 : memref<256x256xf32>, %2 : memref<256x256xf32>):
    "affine.for"() ({
    ^3(%3 : index):
      "affine.for"() ({
      ^4(%4 : index):
        "affine.for"() ({
        ^5(%5 : index):
          %6 = "memref.load"(%0, %3, %5) : (memref<256x256xf32>, index, index) -> f32
          %7 = "memref.load"(%1, %5, %4) : (memref<256x256xf32>, index, index) -> f32
          %8 = "memref.load"(%2, %3, %4) : (memref<256x256xf32>, index, index) -> f32
          %9 = "arith.mulf"(%6, %7) : (f32, f32) -> f32
          %10 = "arith.addf"(%8, %9) : (f32, f32) -> f32
          "memref.store"(%10, %2, %3, %4) : (f32, memref<256x256xf32>, index, index) -> ()
        }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : () -> ()
      }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : () -> ()
    }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : () -> ()
    "func.return"(%2) : (memref<256x256xf32>) -> ()
  }) {"sym_name" = "affine_mm", "function_type" = (memref<256x256xf32>, memref<256x256xf32>, memref<256x256xf32>) -> memref<256x256xf32>, "sym_visibility" = "private"} : () -> ()

  //CHECK:      "func.func"() ({
  //CHECK-NEXT: ^2(%{{.*}} : memref<256x256xf32>, %{{.*}} : memref<256x256xf32>, %{{.*}} : memref<256x256xf32>):
  //CHECK-NEXT:   "affine.for"() ({
  //CHECK-NEXT:   ^3(%{{.*}} : index):
  //CHECK-NEXT:     "affine.for"() ({
  //CHECK-NEXT:     ^4(%{{.*}} : index):
  //CHECK-NEXT:       "affine.for"() ({
  //CHECK-NEXT:       ^5(%{{.*}} : index):
  //CHECK-NEXT:         %{{.*}} = "memref.load"(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<256x256xf32>, index, index) -> f32
  //CHECK-NEXT:         %{{.*}} = "memref.load"(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<256x256xf32>, index, index) -> f32
  //CHECK-NEXT:         %{{.*}} = "memref.load"(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<256x256xf32>, index, index) -> f32
  //CHECK-NEXT:         %{{.*}} = "arith.mulf"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  //CHECK-NEXT:         %{{.*}} = "arith.addf"(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  //CHECK-NEXT:         "memref.store"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (f32, memref<256x256xf32>, index, index) -> ()
  //CHECK-NEXT:       }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : () -> ()
  //CHECK-NEXT:     }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : () -> ()
  //CHECK-NEXT:   }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : () -> ()
  //CHECK-NEXT:   "func.return"(%{{.*}}) : (memref<256x256xf32>) -> ()
  //CHECK-NEXT: }) {"sym_name" = "affine_mm", "function_type" = (memref<256x256xf32>, memref<256x256xf32>, memref<256x256xf32>) -> memref<256x256xf32>, "sym_visibility" = "private"} : () -> ()

}) : () -> ()
