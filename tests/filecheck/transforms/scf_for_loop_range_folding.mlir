// RUN: xdsl-opt -p scf-for-loop-range-folding --split-input-file %s | filecheck %s

// CHECK:       %0 = arith.addi %lb, %shift : index
// CHECK-NEXT:  %1 = arith.addi %ub, %shift : index
// CHECK-NEXT:  %2 = arith.muli %0, %mul_shift : index
// CHECK-NEXT:  %3 = arith.muli %1, %mul_shift : index
// CHECK-NEXT:  %4 = arith.muli %step, %mul_shift : index
// CHECK-NEXT:  scf.for %i = %2 to %3 step %4 {
// CHECK-NEXT:    "test.op"(%i) : (index) -> ()
// CHECK-NEXT:  }

%shift, %mul_shift, %lb, %ub, %step = "test.op"() : () -> (index, index, index, index, index)
scf.for %i = %lb to %ub step %step {
    %shift_idx = arith.addi %shift, %i : index
    %mult_idx = arith.muli %shift_idx, %mul_shift : index
    "test.op"(%mult_idx) : (index) -> ()
}

// CHECK:      %5 = arith.addi %lb, %shift : index
// CHECK-NEXT  %6 = arith.addi %ub, %shift : index
// CHECK-NEXT  scf.for %i_1 = %5 to %6 step %step {
// CHECK-NEXT    %new_shift = arith.addi %mul_shift, %mul_shift : index
// CHECK-NEXT    %mult_idx = arith.muli %i_1, %new_shift : index
// CHECK-NEXT    "test.op"(%mult_idx) : (index) -> ()
// CHECK-NEXT  }

// In this example muli can't be taken out of the loop. We need to loop invariant code motion transform to deal with it.
scf.for %i = %lb to %ub step %step {
    %shift_idx = arith.addi %shift, %i : index
    %new_shift = arith.addi %mul_shift, %mul_shift : index
    %mult_idx = arith.muli %shift_idx, %new_shift : index
    "test.op"(%mult_idx) : (index) -> ()
}

// CHECK:       %7 = arith.addi %lb, %shift : index
// CHECK-NEXT:  %8 = arith.addi %ub, %shift : index
// CHECK-NEXT:  %9 = arith.muli %7, %mul_shift : index
// CHECK-NEXT:  %10 = arith.muli %8, %mul_shift : index
// CHECK-NEXT:  %11 = arith.muli %step, %mul_shift : index
// CHECK-NEXT:  scf.for %j = %9 to %10 step %11 {
// CHECK-NEXT:    %12 = arith.addi %ub2, %shift : index
// CHECK-NEXT:    %13 = arith.muli %12, %mul_shift : index
// CHECK-NEXT:    %14 = arith.muli %step2, %mul_shift : index
// CHECK-NEXT:    scf.for %i_2 = %j to %13 step %14 {
// CHECK-NEXT:      "test.op"(%i_2) : (index) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:  }

%ub2, %step2 = "test.op"() : () -> (index, index)
scf.for %j = %lb to %ub step %step {
    scf.for %i = %j to %ub2 step %step2 {
        %shift_idx = arith.addi %shift, %i : index
        %mult_idx = arith.muli %shift_idx, %mul_shift : index
        "test.op"(%mult_idx) : (index) -> ()
    }
}

// CHECK-NEXT:   %15 = arith.muli %lb, %mul_shift : index
// CHECK-NEXT:   %16 = arith.muli %ub, %mul_shift : index
// CHECK-NEXT:   %17 = arith.muli %step, %mul_shift : index
// CHECK-NEXT:   scf.for %i_3 = %15 to %16 step %17 {
// CHECK-NEXT:     scf.for %j_1 = %lb to %ub step %step {
// CHECK-NEXT:       %b = arith.addi %i_3, %j_1 : index
// CHECK-NEXT:       %c = arith.muli %b, %mul_shift : index
// CHECK-NEXT:       "test.op"(%c) : (index) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }

scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
        %a = arith.muli %i, %mul_shift : index
        %b = arith.addi %a, %j : index
        %c = arith.muli %b, %mul_shift : index
        "test.op"(%c) : (index) -> ()
    }
}
