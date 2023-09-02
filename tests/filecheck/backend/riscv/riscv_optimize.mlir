// RUN: xdsl-opt -p canonicalize,riscv-optimize,canonicalize %s | filecheck %s

builtin.module {
    // Check that multiplication by a constant gets hoisted out
    %0 = riscv.li 0 : () -> !riscv.reg<>
    %1 = riscv.li 3 : () -> !riscv.reg<>
    %2 = riscv.li 1 : () -> !riscv.reg<>
    "riscv_scf.for"(%0, %1, %2) ({
    ^0(%i : !riscv.reg<>):
        %3 = riscv.li 2 : () -> !riscv.reg<>
        %4 = riscv.mul %i, %3 : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %5 = riscv.li 2 : () -> !riscv.reg<>
        %6 = riscv.mul %3, %i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%4, %6) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "riscv_scf.yield"() : () -> ()
    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

    // Check that addition with same value gets hoisted out
    %v = "test.op"() : () -> !riscv.reg<>
    "riscv_scf.for"(%0, %1, %2) ({
    ^0(%i : !riscv.reg<>):
        %a0 = riscv.add %i, %v : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        %a1 = riscv.add %v, %i : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
        "test.op"(%a0, %a1) : (!riscv.reg<>, !riscv.reg<>) -> ()
        "riscv_scf.yield"() : () -> ()
    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}} = riscv.li 1 : () -> !riscv.reg<>
// CHECK-NEXT:    %{{.*}} = riscv.li 0 : () -> !riscv.reg<>
// CHECK-NEXT:    %{{.*}} = riscv.li 6 : () -> !riscv.reg<>
// CHECK-NEXT:    %{{.*}} = riscv.li 2 : () -> !riscv.reg<>
// CHECK-NEXT:    %{{.*}} = "riscv_scf.for"(%{{.*}}, %{{.*}}, %{{.*}}) ({
// CHECK-NEXT:    ^0(%i : !riscv.reg<>):
// CHECK-NEXT:      %{{.*}} = "test.op"(%i, %i) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:      "riscv_scf.yield"() : () -> ()
// CHECK-NEXT:    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:    %v = "test.op"() : () -> !riscv.reg<>
// CHECK-NEXT:    %{{.*}} = riscv.addi %v, 3 : (!riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:    %{{.*}} = "riscv_scf.for"(%v, %{{.*}}, %{{.*}}) ({
// CHECK-NEXT:    ^1(%i_1 : !riscv.reg<>):
// CHECK-NEXT:      "test.op"(%i_1, %i_1) : (!riscv.reg<>, !riscv.reg<>) -> ()
// CHECK-NEXT:      "riscv_scf.yield"() : () -> ()
// CHECK-NEXT:    }) : (!riscv.reg<>, !riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
// CHECK-NEXT:  }
