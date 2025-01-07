// RUN: xdsl-opt --print-op-generic %s | mlir-opt --mlir-print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"func.func"() <{function_type = (tensor<8x8xf64>, tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8x8xf64>), res_attrs = [{llvm.noalias}, {llvm.noalias}], sym_name = "arg_attrs", sym_visibility = "public"}> ({
^bb0(%arg0: tensor<8x8xf64>, %arg1: tensor<8x8xf64>):
    "func.return"(%arg0, %arg1) : (tensor<8x8xf64>, tensor<8x8xf64>) -> ()
}) : () -> ()


// CHECK: "func.func"() <{function_type = (tensor<8x8xf64>, tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8x8xf64>), res_attrs = [{llvm.noalias}, {llvm.noalias}], sym_name = "arg_attrs", sym_visibility = "public"}> ({
// CHECK-NEXT: ^0(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>):
// CHECK-NEXT: "func.return"(%arg0, %arg1) : (tensor<8x8xf64>, tensor<8x8xf64>) -> ()
// CHECK-NEXT: }) : () -> ()
