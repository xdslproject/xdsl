// RUN: xdsl-opt -p scf-for-loop-range-folding --split-input-file %s | filecheck %s

// CHECK:       %new_lb = arith.addi %lb, %shift : index
// CHECK-NEXT:  %new_ub = arith.addi %ub, %shift : index
// CHECK-NEXT:  %new_lb_1 = arith.muli %new_lb, %mul_shift : index
// CHECK-NEXT:  %new_ub_1 = arith.muli %new_ub, %mul_shift : index
// CHECK-NEXT:  %new_step = arith.muli %step, %mul_shift : index
// CHECK-NEXT:  scf.for %i = %new_lb_1 to %new_ub_1 step %new_step {
// CHECK-NEXT:    "test.op"(%i) : (index) -> ()
// CHECK-NEXT:  }

%shift, %mul_shift, %lb, %ub, %step = "test.op"() : () -> (index, index, index, index, index)
scf.for %i = %lb to %ub step %step {
    %0 = arith.addi %shift, %i : index
    %1 = arith.muli %0, %mul_shift : index
    "test.op"(%1) : (index) -> ()
}

// CHECK:       %new_lb_2 = arith.addi %lb, %shift : index
// CHECK-NEXT:  %new_ub_2 = arith.addi %ub, %shift : index
// CHECK-NEXT:  scf.for %i_1 = %new_lb_2 to %new_ub_2 step %step {
// CHECK-NEXT:    %new_shift = arith.addi %mul_shift, %mul_shift : index
// CHECK-NEXT:    %0 = arith.muli %i_1, %new_shift : index
// CHECK-NEXT:    "test.op"(%0) : (index) -> ()
// CHECK-NEXT:  }

// In this example muli can't be taken out of the loop. We need to loop invariant code motion transform to deal with it.
scf.for %i = %lb to %ub step %step {
    %0 = arith.addi %shift, %i : index
    %new_shift = arith.addi %mul_shift, %mul_shift : index
    %2 = arith.muli %0, %new_shift : index
    "test.op"(%2) : (index) -> ()
}

// CHECK:       %new_lb_3 = arith.addi %lb, %shift : index
// CHECK-NEXT:  %new_ub_3 = arith.addi %ub, %shift : index
// CHECK-NEXT:  %new_lb_4 = arith.muli %new_lb_3, %mul_shift : index
// CHECK-NEXT:  %new_ub_4 = arith.muli %new_ub_3, %mul_shift : index
// CHECK-NEXT:  %new_step_1 = arith.muli %step, %mul_shift : index
// CHECK-NEXT:  scf.for %j = %new_lb_4 to %new_ub_4 step %new_step_1 {
// CHECK-NEXT:    %new_ub_5 = arith.addi %ub2, %shift : index
// CHECK-NEXT:    %new_ub_6 = arith.muli %new_ub_5, %mul_shift : index
// CHECK-NEXT:    %new_step_2 = arith.muli %step2, %mul_shift : index
// CHECK-NEXT:    scf.for %i_2 = %j to %new_ub_6 step %new_step_2 {
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
