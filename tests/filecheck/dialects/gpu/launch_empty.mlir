// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %n = "arith.constant"() {"value" = 13 : index} : () -> index
        %one = "arith.constant"() {"value" = 1 : index} : () -> index

        "gpu.launch"(%one, %one, %n, %one, %one, %one) ({})
        {"operand_segment_sizes" = array<i32: 0, 1, 1, 1, 1, 1, 1, 0>} : (index, index, index, index, index, index) -> () 
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: gpu.launch requires a non-empty body.
