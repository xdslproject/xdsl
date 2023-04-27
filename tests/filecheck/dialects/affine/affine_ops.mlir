// RUN: xdsl-opt %s -t mlir | xdsl-opt -f mlir -t mlir | filecheck %s

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

}) : () -> ()
