// RUN: xdsl-opt %s -p accfg-dedup | filecheck %s

func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
  %0 = arith.constant 0 : index
  %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
  %2 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
  %3 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
  %4 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index

  %5 = "accfg.setup"(%1, %2, %3, %4) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index) -> !accfg.state<"snax_hwpe_mult">
  %6 = "accfg.launch"(%5) <{"accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
  "accfg.await"(%6) : (!accfg.token<"snax_hwpe_mult">) -> ()

  %7 = "test.op"() : () -> i1

  %8, %9 = "scf.if"(%7) ({
    %10 = "accfg.setup"(%1, %3, %3, %4, %5) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
    %11 = "accfg.launch"(%10) <{"accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
    "accfg.await"(%11) : (!accfg.token<"snax_hwpe_mult">) -> ()

    %12 = "test.op"() : () -> i32
    scf.yield %12, %10 : i32, !accfg.state<"snax_hwpe_mult">
  }, {
    %13 = "test.op"() : () -> i32

    %14 = "accfg.setup"(%1, %2, %3, %4, %5) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
    %15 = "accfg.launch"(%14) <{"accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
    "accfg.await"(%15) : (!accfg.token<"snax_hwpe_mult">) -> ()

    scf.yield %13, %14 : i32, !accfg.state<"snax_hwpe_mult">
  }) : (i1) -> (i32, !accfg.state<"snax_hwpe_mult">)

  "test.op"(%8) : (i32) -> ()

  %16 = "accfg.setup"(%1, %3, %2, %4, %9) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 1>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
  %17 = "accfg.launch"(%16) <{"accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
  "accfg.await"(%17) : (!accfg.token<"snax_hwpe_mult">) -> ()

  func.return
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
// CHECK-NEXT:     %0 = arith.constant 0 : index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:     %3 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
// CHECK-NEXT:     %4 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index

                   // Full setup:

// CHECK-NEXT:     %5 = "accfg.setup"(%1, %2, %3, %4) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index) -> !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     %6 = "accfg.launch"(%5) <{"accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:     "accfg.await"(%6) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     %7 = "test.op"() : () -> i1
// CHECK-NEXT:     %8, %9 = "scf.if"(%7) ({

                     // Elide A and O setups:

// CHECK-NEXT:       %10 = "accfg.setup"(%3, %5) <{"param_names" = ["B"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 1, 1>}> : (index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:       %11 = "accfg.launch"(%10) <{"accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:       "accfg.await"(%11) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:       %12 = "test.op"() : () -> i32

                     // Hoisted into if, elided A, B

// CHECK-NEXT:       %13 = "accfg.setup"(%2, %10) <{"param_names" = ["O"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 1, 1>}> : (index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:       scf.yield %12, %13 : i32, !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     }, {
// CHECK-NEXT:       %14 = "test.op"() : () -> i32

                     // Elide full setup op

// CHECK-NEXT:       %15 = "accfg.launch"(%5) <{"accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:       "accfg.await"(%15) : (!accfg.token<"snax_hwpe_mult">) -> ()

                     // Hoisted into if, elided A

// CHECK-NEXT:       %16 = "accfg.setup"(%3, %2, %5) <{"param_names" = ["B", "O"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 2, 1>}> : (index, index, !accfg.state<"snax_hwpe_mult">) -> !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:       scf.yield %14, %16 : i32, !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:     }) : (i1) -> (i32, !accfg.state<"snax_hwpe_mult">)
// CHECK-NEXT:     "test.op"(%8) : (i32) -> ()

                   // fully elided setup op

// CHECK-NEXT:     %17 = "accfg.launch"(%9) <{"accelerator" = "snax_hwpe_mult"}> : (!accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:     "accfg.await"(%17) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
