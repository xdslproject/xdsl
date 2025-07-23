// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic | filecheck %s

// CHECK: a = #dlti.dl_entry<"str", i32>
// CHECK: b = #dlti.dl_entry<i32, i32>
"test.op"() {
    a = #dlti.dl_entry<"str", i32>,
    b = #dlti.dl_entry<i32, i32>
} : () -> ()
