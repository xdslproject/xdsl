// RUN: xdsl-opt -p riscv-scf-loop-flatten %s | filecheck %s

// CHECK:       builtin.module {

// Success cases
%c0 = riscv.li 0 : () -> !riscv.reg<>
%c1 = riscv.li 1 : () -> !riscv.reg<>
%c3 = riscv.li 3 : () -> !riscv.reg<>
%c5 = riscv.li 5 : () -> !riscv.reg<>
%c8 = riscv.li 8 : () -> !riscv.reg<>
%c64 = riscv.li 64 : () -> !riscv.reg<>

// CHECK-NEXT:    %c0 = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:    %c1 = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:    %c3 = riscv.li 3 : () -> !riscv.reg<>
// CHECK-NEXT:    %c5 = riscv.li 5 : () -> !riscv.reg<>
// CHECK-NEXT:    %c8 = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:    %c64 = riscv.li 64 : () -> !riscv.reg<>

%non_const = "test.op"() : () -> !riscv.reg<>
// CHECK-NEXT:    %non_const = "test.op"() : () -> !riscv.reg<>

riscv_scf.for %16 : !riscv.reg<> = %non_const to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c1 {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19) : (!riscv.reg<>) -> ()
    }
}

// CHECK-NEXT:    riscv_scf.for %0 : !riscv.reg<> = %non_const to %c64 step %c1 {
// CHECK-NEXT:      %1 = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:      "test.op"(%0) : (!riscv.reg<>) -> ()
// CHECK-NEXT:    }

riscv_scf.for %i : !riscv.reg<> = %c0 to %c64 step %c5 {
    riscv_scf.for %j : !riscv.reg<> = %c0 to %c8 step %c3 {
        %k = riscv.li 8 : () -> !riscv.reg<>
        "test.op"(%k) : (!riscv.reg<>) -> ()
    }
}

// CHECK-NEXT:    %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:    %{{.*}} = riscv.mul %c64, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %c0 to %{{.*}} step %c5 {
// CHECK-NEXT:      %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:      "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:    }

// Failures add induction variables:

// Cannot fuse outer loop with iteration arguments
%res0 = riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 iter_args(%arg0 = %c0) -> (!riscv.reg<>) {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c1 {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19) : (!riscv.reg<>) -> ()
    }
    riscv_scf.yield %arg0 : !riscv.reg<>
}

// CHECK-NEXT:    %{{.*}} = riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!riscv.reg<>) {
// CHECK-NEXT:      riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      riscv_scf.yield %{{.*}} : !riscv.reg<>
// CHECK-NEXT:    }

// Inner loop must be the only operation in the outer loop, aside from yield
riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c1 {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19) : (!riscv.reg<>) -> ()
    }
    %20 = riscv.li 42 : () -> !riscv.reg<>
}

// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:      %{{.*}} = riscv.li 42 : () -> !riscv.reg<>
// CHECK-NEXT:    }

// Cannot fuse inner loop with iteration arguments
riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 {
    %res1 = riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c1 iter_args(%arg1 = %c0) -> (!riscv.reg<>) {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19) : (!riscv.reg<>) -> ()
        riscv_scf.yield %arg1 : !riscv.reg<>
    }
}
// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      %{{.*}} = riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!riscv.reg<>) {
// CHECK-NEXT:        %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:        riscv_scf.yield %{{.*}} : !riscv.reg<>
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Cannot fuse inner loop with non-zero lb
riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c8 to %c8 step %c1 {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19) : (!riscv.reg<>) -> ()
    }
}

// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }


// Each iter arg must only be used once, in an add

riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c1 {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19, %16) : (!riscv.reg<>, !riscv.reg<>) -> ()
    }
}
riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c1 {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19, %17) : (!riscv.reg<>, !riscv.reg<>) -> ()
    }
}
riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c1 {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.mul %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19) : (!riscv.reg<>) -> ()
    }
}

// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        "test.op"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        "test.op"(%{{.*}}, %{{.*}}) : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Inner loop step must be constant
riscv_scf.for %16 : !riscv.reg<> = %non_const to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %non_const {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19) : (!riscv.reg<>) -> ()
    }
}

// CHECK-NEXT:    riscv_scf.for %16 : !riscv.reg<> = %non_const to %c64 step %c8 {
// CHECK-NEXT:        riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %non_const {
// CHECK-NEXT:            %18 = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:            %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:            "test.op"(%19) : (!riscv.reg<>) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:    }

// Inner loop step must evenly divide outer loop step
riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c3 {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19) : (!riscv.reg<>) -> ()
    }
}

// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }

// Inner loop step must evenly divide outer loop step
riscv_scf.for %16 : !riscv.reg<> = %c0 to %c64 step %c8 {
    riscv_scf.for %17 : !riscv.reg<> = %c0 to %c8 step %c3 {
        %18 = riscv.li 8 : () -> !riscv.reg<>
        %19 = riscv.add %16, %17 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%19) : (!riscv.reg<>) -> ()
    }
}

// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }


// Failures no induction variables:

riscv_scf.for %i : !riscv.reg<> = %c1 to %c64 step %c5 {
    riscv_scf.for %j : !riscv.reg<> = %c0 to %c8 step %c3 {
        %k = riscv.li 8 : () -> !riscv.reg<>
        "test.op"(%k) : (!riscv.reg<>) -> ()
    }
}

// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %c1 to %c64 step %c5 {
// CHECK-NEXT:        riscv_scf.for %{{.*}} : !riscv.reg<> = %c0 to %c8 step %c3 {
// CHECK-NEXT:            %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:            "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:    }

riscv_scf.for %i : !riscv.reg<> = %non_const to %c64 step %c5 {
    riscv_scf.for %j : !riscv.reg<> = %c0 to %c8 step %c3 {
        %k = riscv.li 8 : () -> !riscv.reg<>
        "test.op"(%k) : (!riscv.reg<>) -> ()
    }
}

// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg<> = %non_const to %c64 step %c5 {
// CHECK-NEXT:        riscv_scf.for %{{.*}} : !riscv.reg<> = %c0 to %c8 step %c3 {
// CHECK-NEXT:            %{{.*}} = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:            "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:        }
// CHECK-NEXT:    }

// CHECK-NEXT:  }
