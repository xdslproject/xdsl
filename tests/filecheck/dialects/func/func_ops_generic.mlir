// RUN: xdsl-opt %s --print-op-generic | xdsl-opt --print-op-generic | filecheck %s

"func.func"() <{function_type = (tensor<8x8xf64>, tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8x8xf64>), res_attrs = [{llvm.noalias}, {llvm.noalias}], sym_name = "arg_attrs", sym_visibility = "public"}> ({
^bb0(%arg0: tensor<8x8xf64>, %arg1: tensor<8x8xf64>):
    "func.return"(%arg0, %arg1) : (tensor<8x8xf64>, tensor<8x8xf64>) -> ()
}) : () -> ()


// CHECK: "func.func"() <{function_type = (tensor<8x8xf64>, tensor<8x8xf64>) -> (tensor<8x8xf64>, tensor<8x8xf64>), res_attrs = [{llvm.noalias}, {llvm.noalias}], sym_name = "arg_attrs", sym_visibility = "public"}> ({
// CHECK-NEXT: ^0(%arg0 : tensor<8x8xf64>, %arg1 : tensor<8x8xf64>):
// CHECK-NEXT: "func.return"(%arg0, %arg1) : (tensor<8x8xf64>, tensor<8x8xf64>) -> ()
// CHECK-NEXT: }) : () -> ()

func.func @output_attributes() -> (f32 {dialect.a = 0 : i32}, f32 {dialect.b = 0 : i32, dialect.c = 1 : i64}) {
    %r1, %r2 = "test.op"() : () -> (f32, f32)
    return %r1, %r2 : f32, f32
}

// CHECK:       "func.func"() <{sym_name = "output_attributes", function_type = () -> (f32, f32), res_attrs = [{dialect.a = 0 : i32}, {dialect.b = 0 : i32, dialect.c = 1 : i64}]}> ({
// CHECK-NEXT:      %r1, %r2 = "test.op"() : () -> (f32, f32)
// CHECK-NEXT:      "func.return"(%r1, %r2) : (f32, f32) -> ()
// CHECK-NEXT:  }) : () -> ()
