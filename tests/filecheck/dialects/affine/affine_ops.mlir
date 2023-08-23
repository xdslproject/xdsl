// RUN: XDSL_AUTO_ROUNDTRIP

builtin.module {

    // For without value being passed during iterations

    "affine.for"() ({
    ^0(%i : index):
      "affine.yield"() : () -> ()
    }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (256)>, "step" = 1 : index} : () -> ()

    // For with values being passed during iterations

    %init_value = "test.op"() : () -> !test.type<"int">
    %res = "affine.for"(%init_value) ({
    ^1(%i : index, %step_value : !test.type<"int">):
      %next_value = "test.op"() : () -> !test.type<"int">
      "affine.yield"(%next_value) : (!test.type<"int">) -> ()
    }) {"lower_bound" = affine_map<() -> (-10)>, "upper_bound" = affine_map<() -> (10)>, "step" = 1 : index} : (!test.type<"int">) -> !test.type<"int">


    %memref = "test.op"() : () -> memref<2x3xf64>
    %value = "test.op"() : () -> f64
    "affine.store"(%value, %memref) {"map" = affine_map<() -> (0, 0)>} : (f64, memref<2x3xf64>) -> ()

    %zero = "test.op"() : () -> index
    %same_value = "affine.load"(%memref, %zero, %zero) {"map" = affine_map<(d0, d1) -> (d0, d1)>} : (memref<2x3xf64>, index, index) -> f64


}
