// RUN: xdsl-opt -p cse --split-input-file %s | filecheck %s

%a8 = riscv.li 8 : !riscv.reg
%b8 = riscv.li 8 : !riscv.reg
%c8 = riscv.li 8 : !riscv.reg

%a7 = riscv.li 7 : !riscv.reg
%b7 = riscv.li 7 : !riscv.reg

riscv.assembly_section ".text" {
    %d8 = riscv.li 8 : !riscv.reg
    %e8 = riscv.li 8 : !riscv.reg

    "test.op"(%d8, %e8) : (!riscv.reg, !riscv.reg) -> ()
}

%f8 = riscv.li 8 : !riscv.reg

"test.op"(%a8, %b8, %c8, %a7, %a7, %b7, %f8) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %a8 = riscv.li 8 : !riscv.reg
// CHECK-NEXT:    %a7 = riscv.li 7 : !riscv.reg
// CHECK-NEXT:    riscv.assembly_section ".text" {
// CHECK-NEXT:      %d8 = riscv.li 8 : !riscv.reg
// CHECK-NEXT:      "test.op"(%d8, %d8) : (!riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    "test.op"(%a8, %a8, %a8, %a7, %a7, %a7, %a8) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()
// CHECK-NEXT:  }

// -----

%0, %1, %2 = "test.op"() : () -> (!stream.readable<!riscv.freg>, !stream.readable<!riscv.freg>, !riscv.reg)

%8 = riscv.li 8 : !riscv.reg
%9 = riscv.li 8 : !riscv.reg
%10 = riscv.li 8 : !riscv.reg
%11 = riscv.li 0 : !riscv.reg
%12 = riscv.li 1 : !riscv.reg
riscv_scf.for %13 : !riscv.reg = %11 to %8 step %12 {
    riscv_scf.for %14 : !riscv.reg = %11 to %9 step %12 {
        %15 = riscv.li 8 : !riscv.reg
        %16 = riscv.mul %15, %13 : (!riscv.reg, !riscv.reg) -> !riscv.reg
        %17 = riscv.add %16, %14 : (!riscv.reg, !riscv.reg) -> !riscv.reg
        %18 = riscv.li 8 : !riscv.reg
        %19 = riscv.mul %17, %18 {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
        %20 = riscv.add %2, %19 : (!riscv.reg, !riscv.reg) -> !riscv.reg
        %21 = riscv.fld %20, 0 {"comment" = "load double from memref of shape (8, 8)"} : (!riscv.reg) -> !riscv.freg
        %22 = riscv.fmv.d %21 : (!riscv.freg) -> !riscv.freg
        %23 = riscv_scf.for %24 : !riscv.reg = %11 to %10 step %12 iter_args(%25 = %22) -> (!riscv.freg) {
            %26 = riscv_snitch.read from %0 : !riscv.freg
            %27 = riscv_snitch.read from %1 : !riscv.freg
            %28 = riscv.fmul.d %26, %27 : (!riscv.freg, !riscv.freg) -> !riscv.freg
            %29 = riscv.fadd.d %21, %28 : (!riscv.freg, !riscv.freg) -> !riscv.freg
            riscv_scf.yield %29 : !riscv.freg
        }
        %30 = riscv.li 8 : !riscv.reg
        %31 = riscv.mul %30, %13 : (!riscv.reg, !riscv.reg) -> !riscv.reg
        %32 = riscv.add %31, %14 : (!riscv.reg, !riscv.reg) -> !riscv.reg
        %33 = riscv.li 8 : !riscv.reg
        %34 = riscv.mul %32, %33 {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
        %35 = riscv.add %2, %34 : (!riscv.reg, !riscv.reg) -> !riscv.reg
        riscv.fsd %35, %23, 0 {"comment" = "store double value to memref of shape (8, 8)"} : (!riscv.reg, !riscv.freg) -> ()
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %{{.*}}, %{{.*}}, %{{.*}} = "test.op"() : () -> (!stream.readable<!riscv.freg>, !stream.readable<!riscv.freg>, !riscv.reg)
// CHECK-NEXT:    %{{.*}} = riscv.li 8 : !riscv.reg
// CHECK-NEXT:    %{{.*}} = riscv.li 0 : !riscv.reg
// CHECK-NEXT:    %{{.*}} = riscv.li 1 : !riscv.reg
// CHECK-NEXT:    riscv_scf.for %{{.*}} : !riscv.reg = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      riscv_scf.for %{{.*}} : !riscv.reg = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %{{.*}} = riscv.mul %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:        %{{.*}} = riscv.mul %{{.*}}, %{{.*}} {"comment" = "multiply by element size"} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:        %{{.*}} = riscv.add %{{.*}}, %{{.*}} : (!riscv.reg, !riscv.reg) -> !riscv.reg
// CHECK-NEXT:        %{{.*}} = riscv.fld %{{.*}}, 0 {"comment" = "load double from memref of shape (8, 8)"} : (!riscv.reg) -> !riscv.freg
// CHECK-NEXT:        %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg) -> !riscv.freg
// CHECK-NEXT:        %{{.*}} = riscv_scf.for %{{.*}} : !riscv.reg = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!riscv.freg) {
// CHECK-NEXT:          %{{.*}} = riscv_snitch.read from %{{.*}} : !riscv.freg
// CHECK-NEXT:          %{{.*}} = riscv_snitch.read from %{{.*}} : !riscv.freg
// CHECK-NEXT:          %{{.*}} = riscv.fmul.d %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:          %{{.*}} = riscv.fadd.d %{{.*}}, %{{.*}} : (!riscv.freg, !riscv.freg) -> !riscv.freg
// CHECK-NEXT:          riscv_scf.yield %{{.*}} : !riscv.freg
// CHECK-NEXT:        }
// CHECK-NEXT:        riscv.fsd %{{.*}}, %{{.*}}, 0 {"comment" = "store double value to memref of shape (8, 8)"} : (!riscv.reg, !riscv.freg) -> ()
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
