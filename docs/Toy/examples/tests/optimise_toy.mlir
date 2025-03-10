// RUN: python -m toy %s --emit=toy-opt --ir | filecheck %s

"builtin.module"() ({
// CHECK:       builtin.module {

  "toy.func"() ({
    %10 = "toy.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %11 = "toy.transpose"(%10) : (tensor<2x3xf64>) -> tensor<3x2xf64>
    %12 = "toy.transpose"(%11) : (tensor<3x2xf64>) -> tensor<2x3xf64>
    "toy.return"(%12) : (tensor<2x3xf64>) -> ()
  }) {sym_name = "redundant_transpose", function_type = () -> tensor<2x3xf64>} : () -> ()

// CHECK-NEXT:  "toy.func"() ({
// CHECK-NEXT:    %{{.*}} = "toy.constant"() {value =
// CHECK-SAME{LITERAL}: dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : ()
// CHECK-NEXT:    "toy.return"(%{{.*}}) : (tensor<2x3xf64>) -> ()
// CHECK-NEXT:  }) {sym_name = "redundant_transpose", function_type = () -> tensor<2x3xf64>} : () -> ()



  "toy.func"() ({
    %30 = "toy.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %31 = "toy.reshape"(%30) : (tensor<2x3xf64>) -> tensor<6x1xf64>
    %32 = "toy.reshape"(%31) : (tensor<6x1xf64>) -> tensor<1x6xf64>
    %33 = "toy.reshape"(%32) : (tensor<1x6xf64>) -> tensor<2x3xf64>
    "toy.return"(%33) : (tensor<2x3xf64>) -> ()
  }) {sym_name = "redundant_reshape", function_type = () -> tensor<2x3xf64>} : () -> ()

// CHECK-NEXT:  "toy.func"() ({
// CHECK-NEXT:    %{{.*}} = "toy.constant"() {value =
// CHECK-SAME{LITERAL}: dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
// CHECK-NEXT:    "toy.return"(%{{.*}}) : (tensor<2x3xf64>) -> ()
// CHECK-NEXT:  }) {sym_name = "redundant_reshape", function_type = () -> tensor<2x3xf64>} : () -> ()


  "toy.func"() ({
    %30 = "toy.constant"() {value = dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf64>} : () -> tensor<6xf64>
    %31 = "toy.reshape"(%30) : (tensor<6xf64>) -> tensor<2x3xf64>
    "toy.return"(%31) : (tensor<2x3xf64>) -> ()
  }) {sym_name = "constant_reshape", function_type = () -> tensor<2x3xf64>} : () -> ()

// CHECK-NEXT:  "toy.func"() ({
// CHECK-NEXT:    %{{.*}} = "toy.constant"() {value =
// CHECK-SAME{LITERAL}: dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
// CHECK-NEXT:    "toy.return"(%{{.*}}) : (tensor<2x3xf64>) -> ()
// CHECK-NEXT:  }) {sym_name = "constant_reshape", function_type = () -> tensor<2x3xf64>} : () -> ()


}) : () -> ()
