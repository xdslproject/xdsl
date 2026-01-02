// RUN: xdsl-opt -t llvm %s | filecheck %s

module {
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "a", global_type = i32, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "b", global_type = f64, addr_space = 0 : i32}> ({}) : () -> ()
  "llvm.mlir.global"() <{linkage = #llvm.linkage<"external">, sym_name = "c", global_type = i1, addr_space = 0 : i32}> ({}) : () -> ()
}

// CHECK: @"a" = external global i32
// CHECK-NEXT: @"b" = external global double
// CHECK-NEXT: @"c" = external global i1
