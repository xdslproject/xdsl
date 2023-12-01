// RUN: xdsl-opt -p riscv-scf-loop-range-folding %s | filecheck %s

%0, %1, %2, %3 = "test.op"() : () -> (!riscv.freg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
%c0 = riscv.li 0 : () -> !riscv.reg<>
%c64 = riscv.li 64 : () -> !riscv.reg<>
%c1 = riscv.li 1 : () -> !riscv.reg<>

riscv_scf.for %arg4 : !riscv.reg<> = %c0 to %c64 step %c1 {
    %4 = riscv.li 4 : () -> !riscv.reg<>
    %5 = riscv.mul %arg4, %4 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    "test.op"(%5) : (!riscv.reg<>) -> ()
    %6 = riscv.li 4 : () -> !riscv.reg<>
    %7 = riscv.mul %arg4, %6 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    "test.op"(%7) : (!riscv.reg<>) -> ()
    riscv_scf.yield
}

// Don't hoist multiplication by different constants
riscv_scf.for %arg4 : !riscv.reg<> = %c0 to %c64 step %c1 {
    %4 = riscv.li 4 : () -> !riscv.reg<>
    %5 = riscv.mul %arg4, %4 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %6 = riscv.li 5 : () -> !riscv.reg<>
    %7 = riscv.mul %arg4, %6 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    riscv_scf.yield
}


// CHECK:           builtin.module {
// CHECK-NEXT:        %0, %1, %2, %3 = "test.op"() : () -> (!riscv.freg<>, !riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:        %c0 = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:        %c64 = riscv.li 64 : () -> !riscv.reg<>
// CHECK-NEXT:        %c1 = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.mul %c0, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.mul %c64, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.mul %c1, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        riscv_scf.for %arg4 : !riscv.reg<> = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:            %{{.*}} = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:            "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:            %{{.*}} = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:            "test.op"(%{{.*}}) : (!riscv.reg<>) -> ()
// CHECK-NEXT:            riscv_scf.yield
// CHECK-NEXT:        }
// CHECK-NEXT:        riscv_scf.for %{{.*}} : !riscv.reg<> = %c0 to %c64 step %c1 {
// CHECK-NEXT:            %{{.*}} = riscv.li 4 : () -> !riscv.reg<>
// CHECK-NEXT:            %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:            %{{.*}} = riscv.li 5 : () -> !riscv.reg<>
// CHECK-NEXT:            %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:            riscv_scf.yield
// CHECK-NEXT:        }
// CHECK-NEXT:      }
