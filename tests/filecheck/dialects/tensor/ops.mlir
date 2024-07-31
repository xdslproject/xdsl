// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%index, %tensor = "test.op"() : () -> (index, tensor<?x?xf32>)
%dim = "tensor.dim"(%tensor, %index) : (tensor<?x?xf32>, index) -> index

// CHECK: builtin.module {
// CHECK-NEXT:   %index, %tensor = "test.op"() : () -> (index, tensor<?x?xf32>)
// CHECK-NEXT:   %dim = tensor.dim %tensor, %index : index
// CHECK-NEXT: }

// CHECK-GENERIC: "builtin.module"() ({
// CHECK-GENERIC-NEXT:   %index, %tensor = "test.op"() : () -> (index, tensor<?x?xf32>)
// CHECK-GENERIC-NEXT:   %dim = "tensor.dim"(%tensor, %index) : (tensor<?x?xf32>, index) -> index
// CHECK-GENERIC-NEXT: }) : () -> ()


