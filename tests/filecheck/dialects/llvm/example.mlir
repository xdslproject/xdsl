// RUN: xdsl-opt %s | xdsl-opt --print-op-generic | filecheck %s

"builtin.module"() ({

  // Type tests
  "func.func"() ({
  ^0(%0 : !llvm.struct<(i32)>):
    "func.return"(%0) : (!llvm.struct<(i32)>) -> ()
  }) {"sym_name" = "struct_to_struct", "function_type" = (!llvm.struct<(i32)>) -> !llvm.struct<(i32)>, "sym_visibility" = "private"} : () -> ()

// CHECK:       "func.func"() ({
// CHECK-NEXT:  ^0(%0 : !llvm.struct<(i32)>):
// CHECK-NEXT:    "func.return"(%0) : (!llvm.struct<(i32)>) -> ()
// CHECK-NEXT:  }) {"sym_name" = "struct_to_struct", "function_type" = (!llvm.struct<(i32)>) -> !llvm.struct<(i32)>, "sym_visibility" = "private"} : () -> ()


  "func.func"() ({
  ^1(%1 : !llvm.struct<(i32, i32)>):
    "func.return"(%1) : (!llvm.struct<(i32, i32)>) -> ()
  }) {"sym_name" = "struct_to_struct2", "function_type" = (!llvm.struct<(i32, i32)>) -> !llvm.struct<(i32, i32)>, "sym_visibility" = "private"} : () -> ()

// CHECK:       "func.func"() ({
// CHECK-NEXT:  ^1(%1 : !llvm.struct<(i32, i32)>):
// CHECK-NEXT:    "func.return"(%1) : (!llvm.struct<(i32, i32)>) -> ()
// CHECK-NEXT:  }) {"sym_name" = "struct_to_struct2", "function_type" = (!llvm.struct<(i32, i32)>) -> !llvm.struct<(i32, i32)>, "sym_visibility" = "private"} : () -> ()

  "func.func"() ({
  ^2(%2 : !llvm.struct<(!llvm.struct<(i32)>)>):
    "func.return"(%2) : (!llvm.struct<(!llvm.struct<(i32)>)>) -> ()
  }) {"sym_name" = "nested_struct_to_struct", "function_type" = (!llvm.struct<(!llvm.struct<(i32)>)>) -> !llvm.struct<(!llvm.struct<(i32)>)>, "sym_visibility" = "private"} : () -> ()

// CHECK:       "func.func"() ({
// CHECK-NEXT:  ^2(%2 : !llvm.struct<(!llvm.struct<(i32)>)>):
// CHECK-NEXT:    "func.return"(%2) : (!llvm.struct<(!llvm.struct<(i32)>)>) -> ()
// CHECK-NEXT:  }) {"sym_name" = "nested_struct_to_struct", "function_type" = (!llvm.struct<(!llvm.struct<(i32)>)>) -> !llvm.struct<(!llvm.struct<(i32)>)>, "sym_visibility" = "private"} : () -> ()

  "func.func"() ({
  ^3(%3 : !llvm.array<2 x i64>):
    %4 = "llvm.mlir.undef"() : () -> !llvm.array<1 x i32>
    "func.return"(%4) : (!llvm.array<1 x i32>) -> ()
  }) {"sym_name" = "array", "function_type" = (!llvm.array<2 x i64>) -> !llvm.array<1 x i32>, "sym_visibility" = "private"} : () -> ()

// CHECK:       "func.func"() ({
// CHECK-NEXT:  ^3(%3 : !llvm.array<2 x i64>):
// CHECK-NEXT:    %4 = "llvm.mlir.undef"() : () -> !llvm.array<1 x i32>
// CHECK-NEXT:    "func.return"(%4) : (!llvm.array<1 x i32>) -> ()
// CHECK-NEXT:  }) {"sym_name" = "array", "function_type" = (!llvm.array<2 x i64>) -> !llvm.array<1 x i32>, "sym_visibility" = "private"} : () -> ()

  // Op tests
  "func.func"() ({
    %5 = "arith.constant"() {"value" = 1 : i32} : () -> i32
    %6 = "llvm.mlir.undef"() : () -> !llvm.struct<(i32)>
    %7 = "llvm.insertvalue"(%6, %5) {"position" = array<i64: 0>} : (!llvm.struct<(i32)>, i32) -> !llvm.struct<(i32)>
    %8 = "llvm.extractvalue"(%7) {"position" = array<i64: 0>} : (!llvm.struct<(i32)>) -> i32
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "public"} : () -> ()

// CHECK:       "func.func"() ({
// CHECK-NEXT:    %5 = "arith.constant"() {"value" = 1 : i32} : () -> i32
// CHECK-NEXT:    %6 = "llvm.mlir.undef"() : () -> !llvm.struct<(i32)>
// CHECK-NEXT:    %7 = "llvm.insertvalue"(%6, %5) {"position" = array<i64: 0>} : (!llvm.struct<(i32)>, i32) -> !llvm.struct<(i32)>
// CHECK-NEXT:    %8 = "llvm.extractvalue"(%7) {"position" = array<i64: 0>} : (!llvm.struct<(i32)>) -> i32
// CHECK-NEXT:    "func.return"() : () -> ()
// CHECK-NEXT:  }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "public"} : () -> ()

}) : () -> ()
