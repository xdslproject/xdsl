// RUN: xdsl-opt -p canonicalize %s | filecheck %s

%0, %1, %2, %3 = "test.op"() : () -> (!riscv.freg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
%c0 = riscv.li 0 : () -> !riscv.reg<>
%c64 = riscv.li 64 : () -> !riscv.reg<>
%c1 = riscv.li 1 : () -> !riscv.reg<>
"riscv_scf.for"(%c0, %c64, %c1) ({
^0(%arg4 : !riscv.reg<>):
    %4 = riscv.li 4 : () -> !riscv.reg<>
    %5 = riscv.mul %arg4, %4 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %6 = riscv.add %1, %5 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %7 = riscv.flw %6, 0 {"comment" = "load value from memref of shape (64,)"} : (!riscv.reg<>) -> !riscv.freg<>
    %8 = riscv.li 4 : () -> !riscv.reg<>
    %9 = riscv.mul %arg4, %8 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %10 = riscv.add %2, %9 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %11 = riscv.flw %10, 0 {"comment" = "load value from memref of shape (64,)"} : (!riscv.reg<>) -> !riscv.freg<>
    %12 = riscv.fmul.s %7, %0 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %13 = riscv.fadd.s %12, %11 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %14 = riscv.li 4 : () -> !riscv.reg<>
    %15 = riscv.mul %arg4, %14 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %16 = riscv.add %3, %15 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    riscv.fsw %16, %13, 0 {"comment" = "store float value to memref of shape (64,)"} : (!riscv.reg<>, !riscv.freg<>) -> ()
    "riscv_scf.yield"() : () -> ()
}) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()

// Don't hoist multiplication by different constants
"riscv_scf.for"(%c0, %c64, %c1) ({
^0(%arg4 : !riscv.reg<>):
    %4 = riscv.li 4 : () -> !riscv.reg<>
    %5 = riscv.mul %arg4, %4 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %6 = riscv.li 5 : () -> !riscv.reg<>
    %7 = riscv.mul %arg4, %6 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    "riscv_scf.yield"() : () -> ()
}) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> ()


// CHECK:           builtin.module {
// CHECK-NEXT:        %0, %1, %2, %3 = "test.op"() : () -> (!riscv.freg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:        %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:        %c64 = riscv.li 64 : () -> !riscv.reg<>
// CHECK-NEXT:        %c1 = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.li 256 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:        riscv_scf.for %{{.*}} : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          %{{.*}} = riscv.add %{{.*}}, %arg4 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          %{{.*}} = riscv.flw %{{.*}}, 0 {"comment" = "load value from memref of shape (64,)"} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:          %{{.*}} = riscv.add %{{.*}}, %arg4 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          %{{.*}} = riscv.flw %{{.*}}, 0 {"comment" = "load value from memref of shape (64,)"} : (!riscv.reg<>) -> !riscv.freg<>
// CHECK-NEXT:          %{{.*}} = riscv.fmul.s %{{.*}}, %0 : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:          %{{.*}} = riscv.fadd.s %{{.*}}, %{{.*}} : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
// CHECK-NEXT:          %{{.*}} = riscv.add %{{.*}}, %arg4 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          riscv.fsw %{{.*}}, %{{.*}}, 0 {"comment" = "store float value to memref of shape (64,)"} : (!riscv.reg<>, !riscv.freg<>) -> ()
// CHECK-NEXT:          riscv_scf.yield
// CHECK-NEXT:        }
// CHECK-NEXT:        riscv_scf.for %{{.*}} : !riscv.reg<> = %c0 to %c64 step %c1 {
// CHECK-NEXT:            %{{.*}} = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:            %{{.*}} = riscv.mul %{{.*}}, %{{.*}} {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:            %{{.*}} = riscv.li 5 : () -> !riscv.reg<>
// CHECK-NEXT:            %{{.*}} = riscv.mul %{{.*}}, %{{.*}} {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:            riscv_scf.yield
// CHECK-NEXT:        }
// CHECK-NEXT:      }
