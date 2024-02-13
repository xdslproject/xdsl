// RUN: xdsl-opt -p riscv-cse %s | filecheck %s

%0, %1, %2 = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)

%c0 = riscv.get_register : () -> !riscv.reg<zero>
%c8 = riscv.li 8 : () -> !riscv.reg<>
%c8_1 = riscv.li 8 : () -> !riscv.reg<>
%c1 = riscv.li 1 : () -> !riscv.reg<>
riscv_scf.for %arg3 : !riscv.reg<> = %c0 to %c8 step %c1 {
    riscv_scf.for %arg4 : !riscv.reg<> = %c0 to %c8 step %c1 {
        %3 = riscv.li 8 : () -> !riscv.reg<>
        %4 = riscv.mul %3, %arg3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %5 = riscv.add %4, %arg5 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        riscv_scf.for %arg5 : !riscv.reg<> = %c0 to %c8 step %c1 {
            %6 = riscv.li 8 : () -> !riscv.reg<>
            %7 = riscv.mul %3, %arg3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
            %8 = riscv.add %4, %arg5 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
            %9 = riscv.li 8 : () -> !riscv.reg<>
            %10 = riscv.mul %3, %arg3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
            %11 = riscv.add %4, %arg5 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
            %12 = riscv.mul %5, %6 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

            %13 = riscv.li 8 : () -> !riscv.reg<>
            %14 = riscv.mul %3, %arg3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
            %15 = riscv.add %4, %arg5 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

        }
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2 = "test.op"() : () -> (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>)
// CHECK-NEXT:    %c0 = riscv.get_register : () -> !riscv.reg<zero>
// CHECK-NEXT:    %c8 = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:    %c1 = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:    riscv_scf.for %arg3 : !riscv.reg<> = %c0 to %c8 step %c1 {
// CHECK-NEXT:      riscv_scf.for %arg4 : !riscv.reg<> = %c0 to %c8 step %c1 {
// CHECK-NEXT:        %3 = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:        %4 = riscv.mul %3, %arg3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        %5 = riscv.add %4, %arg5 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        riscv_scf.for %arg5 : !riscv.reg<> = %c0 to %c8 step %c1 {
// CHECK-NEXT:          %6 = riscv.li 8 : () -> !riscv.reg<>
// CHECK-NEXT:          %7 = riscv.mul %3, %arg3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          %8 = riscv.add %4, %arg5 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          %9 = riscv.mul %3, %arg3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          %10 = riscv.mul %5, %6 {"comment" = "multiply by element size"} : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:          %11 = riscv.mul %3, %arg3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
