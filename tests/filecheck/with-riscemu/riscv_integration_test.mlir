// RUN: xdsl-run %s | filecheck %s

// Integration tests for builtin -> allocated riscv assembly lowering.
// TODO:
//   - builtin -> riscv
//   - register allocation
// in the future: xdsl-opt (lower to riscv) %s | riscemu | filecheck %s
// (in addition to the xdsl-run above, the test is that they should print the same values)


builtin.module {
  "memref.global"() {"sym_name" = "a", "type" = memref<2x2xf64>, "initial_value" = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>, "sym_visibility" = "public"} : () -> ()
  "memref.global"() {"sym_name" = "b", "type" = memref<2x2xf64>, "initial_value" = dense<[[5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<2x2xf64>, "sym_visibility" = "public"} : () -> ()
  func.func @main() {
    %0 = "memref.get_global"() {"name" = @a} : () -> memref<2x2xf64>
    %1 = "memref.get_global"() {"name" = @b} : () -> memref<2x2xf64>

    // Print inputs
    printf.print_format "{}", %0 : memref<2x2xf64>
    printf.print_format "{}", %1 : memref<2x2xf64>

    // Add

    %2 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x2xf64>
    %3 = arith.constant 0 : index
    %4 = arith.constant 1 : index
    %5 = arith.constant 2 : index

    "scf.for"(%3, %5, %4) ({
    ^0(%arg0 : index):
      "scf.for"(%3, %5, %4) ({
      ^1(%arg1 : index):
        %6 = "memref.load"(%0, %arg0, %arg1) {"nontemporal" = false} : (memref<2x2xf64>, index, index) -> f64
        %7 = "memref.load"(%1, %arg0, %arg1) {"nontemporal" = false} : (memref<2x2xf64>, index, index) -> f64
        %8 = arith.addf %6, %7 : f64
        "memref.store"(%8, %2, %arg0, %arg1) {"nontemporal" = false} : (f64, memref<2x2xf64>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()

    printf.print_format "{}", %2 : memref<2x2xf64>
    "memref.dealloc"(%2) : (memref<2x2xf64>) -> ()

    // Multiply

    %9 = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<2x2xf64>

    "scf.for"(%3, %5, %4) ({
    ^0(%arg0 : index):
      "scf.for"(%3, %5, %4) ({
      ^1(%arg1 : index):
        %10 = arith.constant 0.0 : f64
        %11 = "scf.for"(%3, %5, %4, %10) ({
        ^2(%arg2 : index, %acc : f64):
          %12 = "memref.load"(%0, %arg0, %arg2) {"nontemporal" = false} : (memref<2x2xf64>, index, index) -> f64
          %13 = "memref.load"(%1, %arg2, %arg1) {"nontemporal" = false} : (memref<2x2xf64>, index, index) -> f64
          %14 = arith.mulf %12, %13 : f64
          %15 = arith.addf %acc, %14 : f64
          "scf.yield"(%15) : (f64) -> ()
        }) : (index, index, index, f64) -> (f64)
        "memref.store"(%11, %9, %arg0, %arg1) {"nontemporal" = false} : (f64, memref<2x2xf64>, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()

    printf.print_format "{}", %9 : memref<2x2xf64>
    "memref.dealloc"(%9) : (memref<2x2xf64>) -> ()

    func.return
  }
}

// CHECK:      [[1.0, 2.0], [3.0, 4.0]]
// CHECK-NEXT: [[5.0, 6.0], [7.0, 8.0]]
// CHECK-NEXT: [[6.0, 8.0], [10.0, 12.0]]
// CHECK-NEXT: [[19.0, 22.0], [43.0, 50.0]]
