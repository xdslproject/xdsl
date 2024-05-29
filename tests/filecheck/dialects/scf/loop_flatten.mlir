// RUN: xdsl-opt -p scf-for-loop-flatten %s | filecheck %s

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

scf.for %16 = %non_const to %c64 step %c8 {
    scf.for %17 = %c0 to %c8 step %c1 {
        %18 = arith.constant 8 : index
        %19 = arith.addi %16, %17 : index
        "test.op"(%19) : (index) -> ()
    }
}

// CHECK-NEXT:    scf.for %0 = %non_const to %c64 step %c1 {
// CHECK-NEXT:      %1 = arith.constant 8 : index
// CHECK-NEXT:      "test.op"(%0) : (index) -> ()
// CHECK-NEXT:    }

scf.for %i = %c0 to %c64 step %c5 {
    scf.for %j = %c0 to %c8 step %c3 {
        %k = arith.constant 8 : index
        "test.op"(%k) : (index) -> ()
    }
}

// CHECK-NEXT:    %{{.*}} = arith.constant 2 : index
// CHECK-NEXT:    %{{.*}} = arith.muli %c64, %{{.*}} : index
// CHECK-NEXT:    scf.for %{{.*}} = %c0 to %{{.*}} step %c5 {
// CHECK-NEXT:      %{{.*}} = arith.constant 8 : index
// CHECK-NEXT:      "test.op"(%{{.*}}) : (index) -> ()
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

// CHECK-NEXT:  }
