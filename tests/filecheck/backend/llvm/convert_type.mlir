// RUN: xdsl-opt -t llvm %s | filecheck %s

module {
  // Integers
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "int1", global_type = i1, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "int32", global_type = i32, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "int64", global_type = i64, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "index", global_type = index, addr_space = 0 : i32}> ({}) : () -> ()

  // Floats
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "f16", global_type = f16, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "f32", global_type = f32, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "f64", global_type = f64, addr_space = 0 : i32}> ({}) : () -> ()

  // Pointers
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "ptr_opaque", global_type = !llvm.ptr, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "ptr_addrspace", global_type = !llvm.ptr<1>, addr_space = 0 : i32}> ({}) : () -> ()

  // Vectors
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "vec_i32", global_type = vector<4xi32>, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "vec_f32", global_type = vector<2xf32>, addr_space = 0 : i32}> ({}) : () -> ()

  // Arrays
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "array_i32", global_type = !llvm.array<4 x i32>, addr_space = 0 : i32}> ({}) : () -> ()

  // Aggregates
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "struct_literal", global_type = !llvm.struct<(i32, f64)>, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "complex_f32", global_type = complex<f32>, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "tuple_mixed", global_type = tuple<i32, f64>, addr_space = 0 : i32}> ({}) : () -> ()
  
  // Nested Aggregates
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "nested_struct", global_type = !llvm.struct<(i32, !llvm.struct<(f32)>)>, addr_space = 0 : i32}> ({}) : () -> ()
}

// Integers
// CHECK: @"int1" = external global i1
// CHECK-NEXT: @"int32" = external global i32
// CHECK-NEXT: @"int64" = external global i64
// CHECK-NEXT: @"index" = external global i64

// Floats
// CHECK-NEXT: @"f16" = external global half
// CHECK-NEXT: @"f32" = external global float
// CHECK-NEXT: @"f64" = external global double

// Pointers
// CHECK-NEXT: @"ptr_opaque" = external global ptr
// CHECK-NEXT: @"ptr_addrspace" = external global 1*

// Vectors
// CHECK-NEXT: @"vec_i32" = external global <4 x i32>
// CHECK-NEXT: @"vec_f32" = external global <2 x float>

// Arrays
// CHECK-NEXT: @"array_i32" = external global [4 x i32]

// Aggregates
// CHECK-NEXT: @"struct_literal" = external global {i32, double}
// CHECK-NEXT: @"complex_f32" = external global {float, float}
// CHECK-NEXT: @"tuple_mixed" = external global {i32, double}

// Nested
// CHECK-NEXT: @"nested_struct" = external global {i32, {float}}
