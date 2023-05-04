// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %n = "arith.constant"() {"value" = 13 : index} : () -> index
        %one = "arith.constant"() {"value" = 1 : index} : () -> index

        "gpu.launch"(%one, %one, %n, %one, %one, %one) ({
        ^bb0(%bx : index, %by : index, %bz : index,
            %tx : index, %ty : index, %tz : index):
            %sum = "gpu.all_reduce"(%tx) ({
            }) {"op" = #gpu<all_reduce_op add>} : (index) -> index
            %final = "arith.muli"(%sum, %one) : (index, index) -> index
            "gpu.terminator"() : () -> ()
        }) {"operand_segment_sizes" = array<i32: 0, 1, 1, 1, 1, 1, 1, 0>} : (index, index, index, index, index, index) -> ()
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: Expected [12 x index], got ['index', 'index', 'index', 'index', 'index', 'index']. gpu.launch's body arguments are 12 index arguments, with 3 block indices, 3 block sizes, 3 thread indices, and 3 thread counts
