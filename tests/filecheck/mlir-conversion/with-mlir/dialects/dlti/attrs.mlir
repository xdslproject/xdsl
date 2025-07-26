// RUN: xdsl-opt %s | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | filecheck %s

// CHECK: entry1 = #dlti.dl_entry<"str", i32>
// CHECK: entry2 = #dlti.dl_entry<i32, i32>
"test.op"() {
    entry1 = #dlti.dl_entry<"str", i32>,
    entry2 = #dlti.dl_entry<i32, i32>
} : () -> ()

// CHECK: spec1 = #dlti.dl_spec<>
// CHECK: spec2 = #dlti.dl_spec<"str" = i32, i32 = i32>
"test.op"() {
    spec1 = #dlti.dl_spec<>,
    spec2 = #dlti.dl_spec<"str" = i32, i32 = i32>
} : () -> ()
