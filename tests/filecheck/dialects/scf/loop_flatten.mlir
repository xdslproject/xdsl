// RUN: xdsl-opt -p scf-for-loop-flatten --split-input-file %s | filecheck %s

// CHECK:       builtin.module {

// Success cases
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c3 = arith.constant 3 : index
%c5 = arith.constant 5 : index
%c8 = arith.constant 8 : index
%c64 = arith.constant 64 : index

// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c5 = arith.constant 5 : index
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    %c64 = arith.constant 64 : index

%non_const = "test.op"() : () -> index
// CHECK-NEXT:    %non_const = "test.op"() : () -> index

%int0, %int1, %float0 = "test.op"() : () -> (index, index, f32)
// CHECK-NEXT:    %int0, %int1, %float0 = "test.op"() : () -> (index, index, f32)

scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %0 = %c0 to %c64 step %c1 {
// CHECK-NEXT:      %1 = arith.constant 8 : index
// CHECK-NEXT:      "test.op"(%0) : (index) -> ()
// CHECK-NEXT:    }

// Unknown outer bound prevents flattening.
scf.for %16 = %c0 to %non_const step %c8 {
    scf.for %17 = %c0 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %c0 to %non_const step %c8 {
// CHECK-NEXT:      scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Neither range tiles exactly: 64 is not divisible by 5, nor 8 by 3.
// Keep both loops because flattening would change the number of iterations.
scf.for %i = %c0 to %c64 step %c5 {
    scf.for %j = %c0 to %c8 step %c3 {
        %k = arith.constant 8 : index
        "test.op"(%k) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %c0 to %c64 step %c5 {
// CHECK-NEXT:      scf.for %{{.*}} = %c0 to %c8 step %c3 {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }

%e0, %e1, %e2 = scf.for %16 = %c0 to %c64 step %c8 iter_args(%a0 = %int1, %a1 = %int1, %a2 = %float0) -> (index, index, f32) {
    %d0, %d1, %d2 = scf.for %17 = %c0 to %c8 step %c1 iter_args(%b0 = %a0, %b1 = %a1, %b2 = %a2) -> (index, index, f32) {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
        scf.yield %b0, %b1, %b2 : index, index, f32
    }
    scf.yield %d0, %d1, %d2 : index, index, f32
}

// CHECK-NEXT:    %e0, %e1, %e2 = scf.for %{{.*}} = %c0 to %c64 step %c1 iter_args(%b0 = %int1, %b1 = %int1, %b2 = %float0) -> (index, index, f32) {
// CHECK-NEXT:      %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:      "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      scf.yield %b0, %b1, %b2 : index, index, f32
// CHECK-NEXT:    }

%g0, %g1, %g2 = scf.for %16 = %c0 to %c64 step %c8 iter_args(%a0 = %int1, %a1 = %int1, %a2 = %float0) -> (index, index, f32) {
    %d0, %d1, %d2 = scf.for %17 = %c0 to %c8 step %c1 iter_args(%b0 = %a0, %b1 = %a1, %b2 = %a2) -> (index, index, f32) {
        %k = arith.constant 8 : index
        "test.op"(%k) : (index) -> ()
        scf.yield %b0, %b1, %b2 : index, index, f32
    }
    scf.yield %d0, %d1, %d2 : index, index, f32
}

// CHECK-NEXT:    %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:    %{{.*}} = arith.muli %c64, %{{.*}} : index
// CHECK-NEXT:    %g0, %g1, %g2 = scf.for %{{.*}} = %c0 to %{{.*}} step %c8 iter_args(%b0_1 = %int1, %b1_1 = %int1, %b2_1 = %float0) -> (index, index, f32) {
// CHECK-NEXT:      %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:      "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      scf.yield %b0_1, %b1_1, %b2_1 : index, index, f32
// CHECK-NEXT:    }

// Inner yield does not forward the iteration arguments
%g3, %g4, %g5 = scf.for %16 = %c0 to %c64 step %c8 iter_args(%a0 = %int1, %a1 = %int1, %a2 = %float0) -> (index, index, f32) {
    %d0, %d1, %d2 = scf.for %17 = %c0 to %c8 step %c1 iter_args(%b0 = %a0, %b1 = %a1, %b2 = %a2) -> (index, index, f32) {
        %k = arith.constant 8 : index
        %j = "test.op"(%k) : (index) -> index
        scf.yield %j, %b1, %b2 : index, index, f32
    }
    scf.yield %d0, %d1, %d2 : index, index, f32
}

// CHECK-NEXT:    %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:    %{{.*}} = arith.muli %c64, %{{.*}} : index
// CHECK-NEXT:    %g3, %g4, %g5 = scf.for %{{.*}} = %c0 to %{{.*}} step %c8 iter_args(%b0_2 = %int1, %b1_2 = %int1, %b2_2 = %float0) -> (index, index, f32) {
// CHECK-NEXT:      %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:      %j_1 = "test.op"(%{{.*}}) : (index) -> index
// CHECK-NEXT:      scf.yield %j_1, %b1_2, %b2_2 : index, index, f32
// CHECK-NEXT:    }

// Failures add induction variables:

// Cannot fuse outer loop with iteration arguments
%res0 = scf.for %16 = %c0 to %c64 step %c8 iter_args(%arg0 = %c0) -> (index) {
    scf.for %17 = %c0 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
    scf.yield %arg0 : index
}

// CHECK-NEXT:    %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %{{.*}} : index
// CHECK-NEXT:    }

// Inner loop must be the only operation in the outer loop, aside from yield
scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
    %20 = arith.constant 42 : index
}

// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      %{{.*}} = arith.constant 42 : index
// CHECK-NEXT:    }

// Indices must be used by the same operation
scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %c1 {
        "test.op"(%16) : (index) -> ()
        "test.op"(%17) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Cannot fuse inner loop with iteration arguments
scf.for %16 = %c0 to %c64 step %c8 {
    %res1 = scf.for %17 = %c0 to %c8 step %c1 iter_args(%arg1 = %c0) -> (index) {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
        scf.yield %arg1 : index
    }
}
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        scf.yield %{{.*}} : index
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Cannot fuse inner loop with non-zero lb
scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c8 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }


// Each iter arg must only be used once, in an add

scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19, %16) : (index, index) -> ()
    }
}
scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19, %17) : (index, index) -> ()
    }
}
scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.muli %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}, %{{.*}}) : (index, index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.muli %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Inner loop step must be constant
scf.for %16 = %non_const to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %non_const {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %non_const to %c64 step %c8 {
// CHECK-NEXT:        scf.for %{{.*}} = %c0 to %c8 step %non_const {
// CHECK-NEXT:            %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:            %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:            "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:    }

// Inner loop lb must be constant
scf.for %16 = %non_const to %c64 step %c8 {
    scf.for %17 = %non_const to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %non_const to %c64 step %c8 {
// CHECK-NEXT:        scf.for %{{.*}} = %non_const to %c8 step %c1 {
// CHECK-NEXT:            %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:            %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:            "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:    }

// Inner loop ub must be constant
scf.for %16 = %non_const to %c64 step %c8 {
    scf.for %17 = %c0 to %non_const step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %non_const to %c64 step %c8 {
// CHECK-NEXT:        scf.for %{{.*}} = %c0 to %non_const step %c1 {
// CHECK-NEXT:            %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:            %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:            "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:    }

// Outer loop step must be constant
scf.for %16 = %non_const to %c64 step %non_const {
    scf.for %17 = %c0 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %non_const to %c64 step %non_const {
// CHECK-NEXT:        scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK-NEXT:            %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:            %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:            "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:    }

// Inner loop step must evenly divide outer loop step
scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %c3 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Inner loop step must evenly divide outer loop step
scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %c3 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Inner loop ub must equal divide outer loop step
scf.for %16 = %c0 to %c64 step %c8 {
    scf.for %17 = %c0 to %c5 step %c3 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }


// Failures no induction variables:

scf.for %i = %c1 to %c64 step %c5 {
    scf.for %j = %c0 to %c8 step %c3 {
        %k = arith.constant 8 : index
        "test.op"(%k) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %c1 to %c64 step %c5 {
// CHECK-NEXT:        scf.for %{{.*}} = %c0 to %c8 step %c3 {
// CHECK-NEXT:            %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:            "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:    }

scf.for %i = %non_const to %c64 step %c5 {
    scf.for %j = %c0 to %c8 step %c3 {
        %k = arith.constant 8 : index
        "test.op"(%k) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %{{.*}} = %non_const to %c64 step %c5 {
// CHECK-NEXT:        scf.for %{{.*}} = %c0 to %c8 step %c3 {
// CHECK-NEXT:            %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:            "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:    }

// Iter args failures:

// Different order of induction arguments
%h0, %h1, %h2 = scf.for %16 = %c0 to %c64 step %c8 iter_args(%a0 = %int1, %a1 = %int1, %a2 = %float0) -> (index, index, f32) {
    %d0, %d1, %d2 = scf.for %17 = %c0 to %c8 step %c1 iter_args(%b0 = %a1, %b1 = %a0, %b2 = %a2) -> (index, index, f32) {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
        scf.yield %b0, %b1, %b2 : index, index, f32
    }
    scf.yield %d0, %d1, %d2 : index, index, f32
}

// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (index, index, f32) {
// CHECK-NEXT:      %{{.*}}, %{{.*}}, %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (index, index, f32) {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        scf.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, f32
// CHECK-NEXT:    }


%x0, %x1, %x2 = scf.for %16 = %c0 to %c64 step %c8 iter_args(%a0 = %int1, %a1 = %int1, %a2 = %float0) -> (index, index, f32) {
    %d0, %d1 = scf.for %17 = %c0 to %c8 step %c1 iter_args(%b0 = %a0, %b1 = %a1) -> (index, index) {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
        scf.yield %b0, %b1 : index, index
    }
    scf.yield %d0, %d1, %a2 : index, index, f32
}

// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (index, index, f32) {
// CHECK-NEXT:      %{{.*}}, %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (index, index) {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        scf.yield %{{.*}}, %{{.*}} : index, index
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, f32
// CHECK-NEXT:    }

// Different order of yielded values
%k0, %k1, %k2 = scf.for %16 = %c0 to %c64 step %c8 iter_args(%a0 = %int1, %a1 = %int1, %a2 = %float0) -> (index, index, f32) {
    %d0, %d1, %d2 = scf.for %17 = %c0 to %c8 step %c1 iter_args(%b0 = %a0, %b1 = %a1, %b2 = %a2) -> (index, index, f32) {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
        scf.yield %b0, %b1, %b2 : index, index, f32
    }
    scf.yield %d1, %d0, %d2 : index, index, f32
}

// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (index, index, f32) {
// CHECK-NEXT:      %{{.*}}, %{{.*}}, %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (index, index, f32) {
// CHECK-NEXT:        %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:        %{{.*}} = arith.addi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:        "test.op"(%{{.*}}) : (index) -> ()
// CHECK-NEXT:        scf.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, f32
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, f32
// CHECK-NEXT:    }

// CHECK-NEXT:  }

// -----

// The inner range has a partial tile, even though the outer range is exact.
// With unused IVs, rounding the inner trip count down would lose iterations.
%c0 = arith.constant 0 : index
%c3 = arith.constant 3 : index
%c4 = arith.constant 4 : index
%c8 = arith.constant 8 : index
scf.for %outer = %c0 to %c8 step %c4 {
    scf.for %inner = %c0 to %c4 step %c3 {
        "test.op"() : () -> ()
    }
}
// CHECK:      scf.for %outer = %c0 to %c8 step %c4 {
// CHECK-NEXT:   scf.for %inner = %c0 to %c4 step %c3 {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// The outer range has a partial tile; the inner range tiles exactly.
// This independently checks the outer-span guard when the IVs are unused.
%c0 = arith.constant 0 : index
%c2 = arith.constant 2 : index
%c4 = arith.constant 4 : index
%c10 = arith.constant 10 : index
scf.for %outer = %c0 to %c10 step %c4 {
    scf.for %inner = %c0 to %c4 step %c2 {
        "test.op"() : () -> ()
    }
}
// CHECK:      scf.for %outer = %c0 to %c10 step %c4 {
// CHECK-NEXT:   scf.for %inner = %c0 to %c4 step %c2 {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// An outer remainder also blocks the IV-addition rewrite. Its final inner
// iterations can visit sums beyond the outer upper bound.
%c0 = arith.constant 0 : index
%c2 = arith.constant 2 : index
%c4 = arith.constant 4 : index
%c10 = arith.constant 10 : index
scf.for %outer = %c0 to %c10 step %c4 {
    scf.for %inner = %c0 to %c4 step %c2 {
        %sum = arith.addi %outer, %inner : index
        "test.op"(%sum) : (index) -> ()
    }
}
// CHECK:      scf.for %outer = %c0 to %c10 step %c4 {
// CHECK-NEXT:   scf.for %inner = %c0 to %c4 step %c2 {
// CHECK-NEXT:     %sum = arith.addi %outer, %inner : index
// CHECK-NEXT:     "test.op"(%sum) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// An unknown outer upper bound cannot establish exact tiling, even when
// neither induction variable is used.
%c0 = arith.constant 0 : index
%c2 = arith.constant 2 : index
%c4 = arith.constant 4 : index
%upper = "test.op"() : () -> index
scf.for %outer = %c0 to %upper step %c4 {
    scf.for %inner = %c0 to %c4 step %c2 {
        "test.op"() : () -> ()
    }
}
// CHECK:      scf.for %outer = %c0 to %upper step %c4 {
// CHECK-NEXT:   scf.for %inner = %c0 to %c4 step %c2 {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// The unused-IV rewrite requires a known zero outer lower bound.
// Keep an unknown lower bound, with all other tiling conditions satisfied.
%c0 = arith.constant 0 : index
%c2 = arith.constant 2 : index
%c4 = arith.constant 4 : index
%c8 = arith.constant 8 : index
%lower = "test.op"() : () -> index
scf.for %outer = %lower to %c8 step %c4 {
    scf.for %inner = %c0 to %c4 step %c2 {
        "test.op"() : () -> ()
    }
}
// CHECK:      scf.for %outer = %lower to %c8 step %c4 {
// CHECK-NEXT:   scf.for %inner = %c0 to %c4 step %c2 {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// An unknown lower bound also prevents proving the outer span is an exact
// tile in the IV-addition rewrite, even though nonzero constant bounds can fold.
%c0 = arith.constant 0 : index
%c2 = arith.constant 2 : index
%c4 = arith.constant 4 : index
%c8 = arith.constant 8 : index
%lower = "test.op"() : () -> index
scf.for %outer = %lower to %c8 step %c4 {
    scf.for %inner = %c0 to %c4 step %c2 {
        %sum = arith.addi %outer, %inner : index
        "test.op"(%sum) : (index) -> ()
    }
}
// CHECK:      scf.for %outer = %lower to %c8 step %c4 {
// CHECK-NEXT:   scf.for %inner = %c0 to %c4 step %c2 {
// CHECK-NEXT:     %sum = arith.addi %outer, %inner : index
// CHECK-NEXT:     "test.op"(%sum) : (index) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

// A zero-trip inner range is an exact empty tile. The unused-IV rewrite
// produces a zero upper bound and preserves the initial carried value.
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c4 = arith.constant 4 : index
%c8 = arith.constant 8 : index
%result = scf.for %outer = %c0 to %c8 step %c4 iter_args(%outer_acc = %c0) -> (index) {
    %inner_result = scf.for %inner = %c4 to %c4 step %c2 iter_args(%acc = %outer_acc) -> (index) {
        %next = arith.addi %acc, %c1 : index
        scf.yield %next : index
    }
    scf.yield %inner_result : index
}
// CHECK:      %c8 = arith.constant 8 : index
// CHECK-NEXT: %[[FACTOR:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[UPPER:.*]] = arith.muli %c8, %[[FACTOR]] : index
// CHECK-NEXT: %result = scf.for %inner = %c0 to %[[UPPER]] step %c4 iter_args(%acc = %c0) -> (index) {
// CHECK-NEXT:   %next = arith.addi %acc, %c1 : index
// CHECK-NEXT:   scf.yield %next : index
// CHECK-NEXT: }

// -----

// A nonzero constant outer lower bound still folds in the IV-addition case
// when the span tiles exactly. Keep the lower bound and carried-value mapping.
%c0 = arith.constant 0 : index
%c2 = arith.constant 2 : index
%c4 = arith.constant 4 : index
%c10 = arith.constant 10 : index
%result = scf.for %outer = %c2 to %c10 step %c4 iter_args(%outer_acc = %c0) -> (index) {
    %inner_result = scf.for %inner = %c0 to %c4 step %c2 iter_args(%acc = %outer_acc) -> (index) {
        %sum = arith.addi %outer, %inner : index
        %next = arith.addi %acc, %sum : index
        scf.yield %next : index
    }
    scf.yield %inner_result : index
}
// CHECK:      %result = scf.for %inner = %c2 to %c10 step %c2 iter_args(%acc = %c0) -> (index) {
// CHECK-NEXT:   %next = arith.addi %acc, %inner : index
// CHECK-NEXT:   scf.yield %next : index
// CHECK-NEXT: }

// -----

// Nonpositive steps violate the SCF contract. These are defensive rewrite
// checks only: the pass must leave them alone, not divide by zero or flatten.
// No execution semantics are asserted for these invalid-step inputs.
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%cm1 = arith.constant -1 : index
%c4 = arith.constant 4 : index
%c8 = arith.constant 8 : index
scf.for %outer_zero = %c0 to %c8 step %c0 {
    scf.for %inner = %c0 to %c4 step %c1 {
        "test.op"() : () -> ()
    }
}
scf.for %outer_negative = %c0 to %c8 step %cm1 {
    scf.for %inner = %c0 to %c4 step %c1 {
        "test.op"() : () -> ()
    }
}
scf.for %outer = %c0 to %c8 step %c1 {
    scf.for %inner_zero = %c0 to %c4 step %c0 {
        "test.op"() : () -> ()
    }
}
scf.for %outer = %c0 to %c8 step %c1 {
    scf.for %inner_negative = %c0 to %c4 step %cm1 {
        "test.op"() : () -> ()
    }
}
// CHECK:      scf.for %outer_zero = %c0 to %c8 step %c0 {
// CHECK-NEXT:   scf.for %{{.*}} = %c0 to %c4 step %c1 {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: scf.for %outer_negative = %c0 to %c8 step %cm1 {
// CHECK-NEXT:   scf.for %{{.*}} = %c0 to %c4 step %c1 {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK-NEXT:   scf.for %inner_zero = %c0 to %c4 step %c0 {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: scf.for %{{.*}} = %c0 to %c8 step %c1 {
// CHECK-NEXT:   scf.for %inner_negative = %c0 to %c4 step %cm1 {
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:   }
// CHECK-NEXT: }
