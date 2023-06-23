// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({

    // For without value being passed during iterations

    "affine.for"() ({
    ^0(%i : index):
      "affine.yield"() : () -> ()
    }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : () -> ()

    // CHECK:      "affine.for"() ({
    // CHECK-NEXT: ^0(%{{.*}} : index):
    // CHECK-NEXT:   "affine.yield"() : () -> ()
    // CHECK-NEXT: }) {"lower_bound" = 0 : index, "upper_bound" = 256 : index, "step" = 1 : index} : () -> ()


    // For with values being passed during iterations

    %init_value = "test.op"() : () -> !test.type<"int">
    %res = "affine.for"(%init_value) ({
    ^1(%i : index, %step_value : !test.type<"int">):
      %next_value = "test.op"() : () -> !test.type<"int">
      "affine.yield"(%next_value) : (!test.type<"int">) -> ()
    }) {"lower_bound" = -10 : index, "upper_bound" = 10 : index, "step" = 1 : index} : (!test.type<"int">) -> (!test.type<"int">)

    // CHECK:      %res = "affine.for"(%{{.*}}) ({
    // CHECK-NEXT: ^1(%{{.*}} : index, %{{.*}} : !test.type<"int">):
    // CHECK-NEXT:   %{{.*}} = "test.op"() : () -> !test.type<"int">
    // CHECK-NEXT:   "affine.yield"(%{{.*}}) : (!test.type<"int">) -> ()
    // CHECK-NEXT: }) {"lower_bound" = -10 : index, "upper_bound" = 10 : index, "step" = 1 : index} : (!test.type<"int">) -> !test.type<"int">

    %memref = "test.op"() : () -> memref<2x3xf64>
    %value = "test.op"() : () -> f64
    "affine.store"(%value, %memref) {"map" = #affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()

    // CHECK:      %memref = "test.op"() : () -> memref<2x3xf64>
    // CHECK-NEXT: %value = "test.op"() : () -> f64
    // CHECK-NEXT: "affine.store"(%value, %memref) {"map" = #affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()

    %zero = "test.op"() : () -> index
    %same_value = "affine.load"(%memref, %zero, %zero) {"map" = #affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64

    // CHECK:      %zero = "test.op"() : () -> index
    // CHECK-NEXT: %same_value = "affine.load"(%memref, %zero, %zero) {"map" = #affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64


}) : () -> ()
