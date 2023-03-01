// RUN: xdsl-opt -t mlir %s --verify-diagnostics | filecheck %s

"builtin.module"()({
    "gpu.module"()({
        %n = "arith.constant"() {"value" = 13 : index} : () -> index
        %one = "arith.constant"() {"value" = 1 : index} : () -> index

        "gpu.launch"(%n) ({
        ^bb0(%bx : index, %by : index, %bz : index,
            %tx : index, %ty : index, %tz : index,
            %num_bx : index, %num_by : index, %num_bz : index,
            %num_tx : index, %num_ty : index, %num_tz : index):
            %sum = "gpu.all_reduce"(%tx) ({}) {"op" = #gpu<all_reduce_op add>} : (index) -> index
            %final = "arith.muli"(%sum, %one) : (index, index) -> index    
            "gpu.terminator"() : () -> ()
        }) {"operand_segment_sizes" = array<i32: 0, 0, 0, 0, 1, 0, 0, 0>} : (index, index, index, index, index, index) -> () 
        "gpu.module_end"() : () -> ()
    }) {"sym_name" = "gpu"} : () -> ()
}) {} : () -> ()

// CHECK: gpu.launch requires 3 gridSize and blockSize arguments. Please explicitely set the unused ones to 1
