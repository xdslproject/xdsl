// RUN: xdsl-opt -p scf-for-loop-range-folding --split-input-file %s | filecheck %s

// CHECK:       %shift_idx_lb = arith.addi %lb, %shift : index
// CHECK-NEXT:  %shift_idx_ub = arith.addi %ub, %shift : index
// CHECK-NEXT:  %mult_idx_lb = arith.muli %shift_idx_lb, %mul_shift : index
// CHECK-NEXT:  %mult_idx_ub = arith.muli %shift_idx_ub, %mul_shift : index
// CHECK-NEXT:  %mult_idx_step = arith.muli %step, %mul_shift : index
// CHECK-NEXT:  scf.for %i = %mult_idx_lb to %mult_idx_ub step %mult_idx_step {
// CHECK-NEXT:    "test.op"(%i) : (index) -> ()
// CHECK-NEXT:  }

%shift, %mul_shift, %lb, %ub, %step = "test.op"() : () -> (index, index, index, index, index)
scf.for %i = %lb to %ub step %step {
    %shift_idx = arith.addi %shift, %i : index
    %mult_idx = arith.muli %shift_idx, %mul_shift : index
    "test.op"(%mult_idx) : (index) -> ()
}

// CHECK:       %shift_idx_lb_1 = arith.addi %lb, %shift : index
// CHECK-NEXT:  %shift_idx_ub_1 = arith.addi %ub, %shift : index
// CHECK-NEXT:  scf.for %i_1 = %shift_idx_lb_1 to %shift_idx_ub_1 step %step {
// CHECK-NEXT:    %new_shift = arith.addi %mul_shift, %mul_shift : index
// CHECK-NEXT:    %0 = arith.muli %i_1, %new_shift : index
// CHECK-NEXT:    "test.op"(%0) : (index) -> ()
// CHECK-NEXT:  }

// In this example muli can't be taken out of the loop. We need to loop invariant code motion transform to deal with it.
scf.for %i = %lb to %ub step %step {
    %shift_idx = arith.addi %shift, %i : index
    %new_shift = arith.addi %mul_shift, %mul_shift : index
    %2 = arith.muli %shift_idx, %new_shift : index
    "test.op"(%2) : (index) -> ()
}

// CHECK:       %new_lb_lb = arith.addi %lb, %shift : index
// CHECK-NEXT:  %new_lb_ub = arith.addi %ub, %shift : index
// CHECK-NEXT:  %new_lb_lb_1 = arith.muli %new_lb_lb, %mul_shift : index
// CHECK-NEXT:  %new_lb_ub_1 = arith.muli %new_lb_ub, %mul_shift : index
// CHECK-NEXT:  %new_lb_step = arith.muli %step, %mul_shift : index
// CHECK-NEXT:  scf.for %j = %new_lb_lb_1 to %new_lb_ub_1 step %new_lb_step {
// CHECK-NEXT:    %new_ub = arith.addi %ub2, %shift : index
// CHECK-NEXT:    %new_ub_1 = arith.muli %new_ub, %mul_shift : index
// CHECK-NEXT:    %new_step = arith.muli %step2, %mul_shift : index
// CHECK-NEXT:    scf.for %i_2 = %j to %new_ub_1 step %new_step {
// CHECK-NEXT:      "test.op"(%i_2) : (index) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }

%ub2, %step2 = "test.op"() : () -> (index, index)
scf.for %j = %lb to %ub step %step {
    scf.for %i = %j to %ub2 step %step2 {
        %0 = arith.addi %shift, %i : index
        %1 = arith.muli %0, %mul_shift : index
        "test.op"(%1) : (index) -> ()
    }
}
