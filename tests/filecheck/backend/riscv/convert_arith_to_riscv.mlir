// RUN: xdsl-opt -p convert-arith-to-riscv,reconcile-unrealized-casts %s | filecheck %s
builtin.module {
    // Shared operands
    %lhsi32 = arith.constant 1 : i32
    // CHECK: %{{.*}} = rv32.li 1 : !riscv.reg
    %rhsi32 = arith.constant 2 : i32
    // CHECK-NEXT: %{{.*}} = rv32.li 2 : !riscv.reg
    %lhsindex = arith.constant 1 : index
    // CHECK-NEXT: %{{.*}} = rv32.li 1 : !riscv.reg
    %rhsindex = arith.constant 2 : index
    // CHECK-NEXT: %{{.*}} = rv32.li 2 : !riscv.reg
    %lhsf32 = arith.constant 1.000000e+00 : f32
    // CHECK-NEXT: %{{.*}} = rv32.li 1065353216 : !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fmv.w.x %lhsf32 : (!riscv.reg) -> !riscv.freg
    %rhsf32 = arith.constant 2.000000e+00 : f32
    // CHECK-NEXT: %{{.*}} = rv32.li 1073741824 : !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fmv.w.x %rhsf32 : (!riscv.reg) -> !riscv.freg

    // f64 constants lowered via fcvt / stack (sink each so lowering stays observable)
    %constf64zero = arith.constant 0.0 : f64
    %constf64zero_1 = builtin.unrealized_conversion_cast %constf64zero : f64 to !riscv.freg
    "test.op"(%constf64zero_1) : (!riscv.freg) -> ()
    // CHECK-NEXT: %{{.*}} = rv32.li 0 : !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.d.w %{{.*}} : (!riscv.reg) -> !riscv.freg
    // CHECK-NEXT: "test.op"(%constf64zero_1) : (!riscv.freg) -> ()

    %lhsf64_reg, %rhsf64_reg = "test.op"() : () -> (!riscv.freg, !riscv.freg)
    %lhsf64 = builtin.unrealized_conversion_cast %lhsf64_reg : !riscv.freg to f64
    %rhsf64 = builtin.unrealized_conversion_cast %rhsf64_reg : !riscv.freg to f64

    // CHECK-NEXT: %lhsf64_reg, %rhsf64_reg = "test.op"() : () -> (!riscv.freg, !riscv.freg)

    %f64 = arith.constant 1234.5678 : f64
    // CHECK-NEXT: %{{.*}} = rv32.get_register : !riscv.reg<sp>
    // CHECK-NEXT: %{{.*}} = rv32.li 1083394629 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -4 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = rv32.li 1834810029 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -8 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = riscv.fld %{{.*}}, -8 : (!riscv.reg<sp>) -> !riscv.freg
    %f64_1 = builtin.unrealized_conversion_cast %f64 : f64 to !riscv.freg
    "test.op"(%f64_1) : (!riscv.freg) -> ()
    // CHECK-NEXT: "test.op"(%f64_3) : (!riscv.freg) -> ()

    // Dense vector constants (bits loaded via stack)
    %dense_1xf64 = arith.constant dense<[3.140000e+00]> : vector<1xf64>
    %dense_1xf64_1 = builtin.unrealized_conversion_cast %dense_1xf64 : vector<1xf64> to !riscv.freg
    // CHECK-NEXT: %{{.*}} = rv32.get_register : !riscv.reg<sp>
    // CHECK-NEXT: %{{.*}} = rv32.li 1074339512 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -4 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = rv32.li 1374389535 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -8 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = riscv.fld %{{.*}}, -8 : (!riscv.reg<sp>) -> !riscv.freg
    %dense_2xf32 = arith.constant dense<[1.000000e+00, 2.000000e+00]> : vector<2xf32>
    %dense_2xf32_1 = builtin.unrealized_conversion_cast %dense_2xf32 : vector<2xf32> to !riscv.freg
    // CHECK-NEXT: %{{.*}} = rv32.get_register : !riscv.reg<sp>
    // CHECK-NEXT: %{{.*}} = rv32.li 1073741824 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -4 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = rv32.li 1065353216 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -8 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = riscv.fld %{{.*}}, -8 : (!riscv.reg<sp>) -> !riscv.freg
    %dense_4xf16 = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : vector<4xf16>
    %dense_4xf16_1 = builtin.unrealized_conversion_cast %dense_4xf16 : vector<4xf16> to !riscv.freg
    // CHECK-NEXT: %{{.*}} = rv32.get_register : !riscv.reg<sp>
    // CHECK-NEXT: %{{.*}} = rv32.li 1140867584 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -4 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = rv32.li 1073757184 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -8 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = riscv.fld %{{.*}}, -8 : (!riscv.reg<sp>) -> !riscv.freg
    "test.op"(%dense_1xf64_1, %dense_2xf32_1, %dense_4xf16_1) : (!riscv.freg, !riscv.freg, !riscv.freg) -> ()
    // CHECK-NEXT: "test.op"(%dense_1xf64_3, %dense_2xf32_3, %dense_4xf16_3) : (!riscv.freg, !riscv.freg, !riscv.freg) -> ()

    // Integer add / sub
    %addi32 = arith.addi %lhsi32, %rhsi32 : i32
    %addi32_1 = builtin.unrealized_conversion_cast %addi32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.add %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %addindex = arith.addi %lhsindex, %rhsindex : index
    %addindex_1 = builtin.unrealized_conversion_cast %addindex : index to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.add %lhsindex, %rhsindex : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %subi32 = arith.subi %lhsi32, %rhsi32 : i32
    %subi32_1 = builtin.unrealized_conversion_cast %subi32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sub %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %subindex = arith.subi %lhsindex, %rhsindex : index
    %subindex_1 = builtin.unrealized_conversion_cast %subindex : index to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sub %lhsindex, %rhsindex : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "test.op"(%addi32_1, %addindex_1, %subi32_1, %subindex_1) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: "test.op"(%addi32, %addindex, %subi32, %subindex) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()

    // Integer mul
    %muli32 = arith.muli %lhsi32, %rhsi32 : i32
    %muli32_1 = builtin.unrealized_conversion_cast %muli32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.mul %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %mulindex = arith.muli %lhsindex, %rhsindex : index
    %mulindex_1 = builtin.unrealized_conversion_cast %mulindex : index to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.mul %lhsindex, %rhsindex : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "test.op"(%muli32_1, %mulindex_1) : (!riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: "test.op"(%muli32, %mulindex) : (!riscv.reg, !riscv.reg) -> ()

    // Integer div
    %divui32 = arith.divui %lhsi32, %rhsi32 : i32
    %divui32_1 = builtin.unrealized_conversion_cast %divui32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.divu %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %divsi32 = arith.divsi %lhsi32, %rhsi32 : i32
    %divsi32_1 = builtin.unrealized_conversion_cast %divsi32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.div %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %remui = arith.remui %lhsi32, %rhsi32 : i32
    %remui_1 = builtin.unrealized_conversion_cast %remui : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.remu %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %remsi = arith.remsi %lhsi32, %rhsi32 : i32
    %remsi_1 = builtin.unrealized_conversion_cast %remsi : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.rem %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "test.op"(%divui32_1, %divsi32_1, %remui_1, %remsi_1) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: "test.op"(%divui32, %divsi32, %remui, %remsi) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()

    // Bitwise
    %andi32 = arith.andi %lhsi32, %rhsi32 : i32
    %andi32_1 = builtin.unrealized_conversion_cast %andi32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %ori32 = arith.ori %lhsi32, %rhsi32 : i32
    %ori32_1 = builtin.unrealized_conversion_cast %ori32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %xori32 = arith.xori %lhsi32, %rhsi32 : i32
    %xori32_1 = builtin.unrealized_conversion_cast %xori32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xor %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "test.op"(%andi32_1, %ori32_1, %xori32_1) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: "test.op"(%andi32, %ori32, %xori32) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()

    // Shifts
    %shli32 = arith.shli %lhsi32, %rhsi32 : i32
    %shli32_1 = builtin.unrealized_conversion_cast %shli32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sll %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %shrui32 = arith.shrui %lhsi32, %rhsi32 : i32
    %shrui32_1 = builtin.unrealized_conversion_cast %shrui32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.srl %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %shrsi32 = arith.shrsi %lhsi32, %rhsi32 : i32
    %shrsi32_1 = builtin.unrealized_conversion_cast %shrsi32 : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sra %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    "test.op"(%shli32_1, %shrui32_1, %shrsi32_1) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: "test.op"(%shli32, %shrui32, %shrsi32) : (!riscv.reg, !riscv.reg, !riscv.reg) -> ()

    // arith.cmpi
    %cmpi0 = arith.cmpi eq, %lhsi32, %rhsi32 : i32
    %cmpi0_2 = builtin.unrealized_conversion_cast %cmpi0 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xor %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltiu %cmpi0, 1 : (!riscv.reg) -> !riscv.reg
    %cmpi1 = arith.cmpi ne, %lhsi32, %rhsi32 : i32
    %cmpi1_3 = builtin.unrealized_conversion_cast %cmpi1 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = rv32.get_register : !riscv.reg<zero>
    // CHECK-NEXT: %{{.*}} = riscv.xor %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltu %cmpi1, %cmpi1_1 : (!riscv.reg<zero>, !riscv.reg) -> !riscv.reg
    %cmpi2 = arith.cmpi slt, %lhsi32, %rhsi32 : i32
    %cmpi2_1 = builtin.unrealized_conversion_cast %cmpi2 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.slt %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpi3 = arith.cmpi sle, %lhsi32, %rhsi32 : i32
    %cmpi3_2 = builtin.unrealized_conversion_cast %cmpi3 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.slt %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpi3, 1 : (!riscv.reg) -> !riscv.reg
    %cmpi4 = arith.cmpi sgt, %lhsi32, %rhsi32 : i32
    %cmpi4_1 = builtin.unrealized_conversion_cast %cmpi4 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltu %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpi5 = arith.cmpi sge, %lhsi32, %rhsi32 : i32
    %cmpi5_2 = builtin.unrealized_conversion_cast %cmpi5 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltu %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpi5, 1 : (!riscv.reg) -> !riscv.reg
    %cmpi6 = arith.cmpi ult, %lhsi32, %rhsi32 : i32
    %cmpi6_1 = builtin.unrealized_conversion_cast %cmpi6 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltu %rhsi32, %lhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpi7 = arith.cmpi ule, %lhsi32, %rhsi32 : i32
    %cmpi7_2 = builtin.unrealized_conversion_cast %cmpi7 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltu %rhsi32, %lhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpi7, 1 : (!riscv.reg) -> !riscv.reg
    "test.op"(%cmpi0_2, %cmpi1_3, %cmpi2_1, %cmpi3_2, %cmpi4_1, %cmpi5_2, %cmpi6_1, %cmpi7_2) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: "test.op"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()

    // f32 arithmetic
    %addf32 = arith.addf %lhsf32, %rhsf32 : f32
    %addf32_1 = builtin.unrealized_conversion_cast %addf32 : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fadd.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf32 = arith.subf %lhsf32, %rhsf32 : f32
    %subf32_1 = builtin.unrealized_conversion_cast %subf32 : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsub.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf32 = arith.mulf %lhsf32, %rhsf32 : f32
    %mulf32_1 = builtin.unrealized_conversion_cast %mulf32 : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmul.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf32 = arith.divf %lhsf32, %rhsf32 : f32
    %divf32_1 = builtin.unrealized_conversion_cast %divf32 : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %negf32 = arith.negf %rhsf32 : f32
    %negf32_1 = builtin.unrealized_conversion_cast %negf32 : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsgnjn.s %rhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf32 = arith.minimumf %lhsf32, %rhsf32 : f32
    %minf32_1 = builtin.unrealized_conversion_cast %minf32 : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmin.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf32 = arith.maximumf %lhsf32, %rhsf32 : f32
    %maxf32_1 = builtin.unrealized_conversion_cast %maxf32 : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmax.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    "test.op"(%addf32_1, %subf32_1, %mulf32_1, %divf32_1, %negf32_1, %minf32_1, %maxf32_1) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()
    // CHECK-NEXT: "test.op"(%addf32, %subf32, %mulf32, %divf32, %negf32, %minf32, %maxf32) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()

    // f32 fastmath<fast>
    %addf32_fm = arith.addf %lhsf32, %rhsf32 fastmath<fast> : f32
    %addf32_fm_1 = builtin.unrealized_conversion_cast %addf32_fm : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fadd.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf32_fm = arith.subf %lhsf32, %rhsf32 fastmath<fast> : f32
    %subf32_fm_1 = builtin.unrealized_conversion_cast %subf32_fm : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsub.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf32_fm = arith.mulf %lhsf32, %rhsf32 fastmath<fast> : f32
    %mulf32_fm_1 = builtin.unrealized_conversion_cast %mulf32_fm : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmul.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf32_fm = arith.divf %lhsf32, %rhsf32 fastmath<fast> : f32
    %divf32_fm_1 = builtin.unrealized_conversion_cast %divf32_fm : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf32_fm = arith.minimumf %lhsf32, %rhsf32 fastmath<fast> : f32
    %minf32_fm_1 = builtin.unrealized_conversion_cast %minf32_fm : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmin.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf32_fm = arith.maximumf %lhsf32, %rhsf32 fastmath<fast> : f32
    %maxf32_fm_1 = builtin.unrealized_conversion_cast %maxf32_fm : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmax.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    "test.op"(%addf32_fm_1, %subf32_fm_1, %mulf32_fm_1, %divf32_fm_1, %minf32_fm_1, %maxf32_fm_1) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()
    // CHECK-NEXT: "test.op"(%addf32_fm, %subf32_fm, %mulf32_fm, %divf32_fm, %minf32_fm, %maxf32_fm) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()

    // f64 arithmetic
    %addf64 = arith.addf %lhsf64, %rhsf64 : f64
    %addf64_1 = builtin.unrealized_conversion_cast %addf64 : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fadd.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf64 = arith.subf %lhsf64, %rhsf64 : f64
    %subf64_1 = builtin.unrealized_conversion_cast %subf64 : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsub.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf64 = arith.mulf %lhsf64, %rhsf64 : f64
    %mulf64_1 = builtin.unrealized_conversion_cast %mulf64 : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmul.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf64 = arith.divf %lhsf64, %rhsf64 : f64
    %divf64_1 = builtin.unrealized_conversion_cast %divf64 : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf64 = arith.minimumf %lhsf64, %rhsf64 : f64
    %minf64_1 = builtin.unrealized_conversion_cast %minf64 : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmin.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf64 = arith.maximumf %lhsf64, %rhsf64 : f64
    %maxf64_1 = builtin.unrealized_conversion_cast %maxf64 : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmax.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    "test.op"(%addf64_1, %subf64_1, %mulf64_1, %divf64_1, %minf64_1, %maxf64_1) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()
    // CHECK-NEXT: "test.op"(%addf64, %subf64, %mulf64, %divf64, %minf64, %maxf64) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()

    // f64 fastmath<fast>
    %addf64_fm = arith.addf %lhsf64, %rhsf64 fastmath<fast> : f64
    %addf64_fm_1 = builtin.unrealized_conversion_cast %addf64_fm : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fadd.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf64_fm = arith.subf %lhsf64, %rhsf64 fastmath<fast> : f64
    %subf64_fm_1 = builtin.unrealized_conversion_cast %subf64_fm : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsub.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf64_fm = arith.mulf %lhsf64, %rhsf64 fastmath<fast> : f64
    %mulf64_fm_1 = builtin.unrealized_conversion_cast %mulf64_fm : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmul.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf64_fm = arith.divf %lhsf64, %rhsf64 fastmath<fast> : f64
    %divf64_fm_1 = builtin.unrealized_conversion_cast %divf64_fm : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf64_fm = arith.minimumf %lhsf64, %rhsf64 fastmath<fast> : f64
    %minf64_fm_1 = builtin.unrealized_conversion_cast %minf64_fm : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmin.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf64_fm = arith.maximumf %lhsf64, %rhsf64 fastmath<fast> : f64
    %maxf64_fm_1 = builtin.unrealized_conversion_cast %maxf64_fm : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmax.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    "test.op"(%addf64_fm_1, %subf64_fm_1, %mulf64_fm_1, %divf64_fm_1, %minf64_fm_1, %maxf64_fm_1) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()
    // CHECK-NEXT: "test.op"(%addf64_fm, %subf64_fm, %mulf64_fm, %divf64_fm, %minf64_fm, %maxf64_fm) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()

    // f64 fastmath<contract>
    %addf64_fm_contract = arith.addf %lhsf64, %rhsf64 fastmath<contract> : f64
    %addf64_fm_contract_1 = builtin.unrealized_conversion_cast %addf64_fm_contract : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fadd.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf64_fm_contract = arith.subf %lhsf64, %rhsf64 fastmath<contract> : f64
    %subf64_fm_contract_1 = builtin.unrealized_conversion_cast %subf64_fm_contract : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fsub.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf64_fm_contract = arith.mulf %lhsf64, %rhsf64 fastmath<contract> : f64
    %mulf64_fm_contract_1 = builtin.unrealized_conversion_cast %mulf64_fm_contract : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmul.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf64_fm_contract = arith.divf %lhsf64, %rhsf64 fastmath<contract> : f64
    %divf64_fm_contract_1 = builtin.unrealized_conversion_cast %divf64_fm_contract : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf64_fm_contract = arith.minimumf %lhsf64, %rhsf64 fastmath<contract> : f64
    %minf64_fm_contract_1 = builtin.unrealized_conversion_cast %minf64_fm_contract : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmin.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf64_fm_contract = arith.maximumf %lhsf64, %rhsf64 fastmath<contract> : f64
    %maxf64_fm_contract_1 = builtin.unrealized_conversion_cast %maxf64_fm_contract : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fmax.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    "test.op"(%addf64_fm_contract_1, %subf64_fm_contract_1, %mulf64_fm_contract_1, %divf64_fm_contract_1, %minf64_fm_contract_1, %maxf64_fm_contract_1) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()
    // CHECK-NEXT: "test.op"(%addf64_fm_contract, %subf64_fm_contract, %mulf64_fm_contract, %divf64_fm_contract, %minf64_fm_contract, %maxf64_fm_contract) : (!riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg, !riscv.freg) -> ()

    // Conversions
    %sitofp32 = arith.sitofp %lhsi32 : i32 to f32
    %sitofp32_1 = builtin.unrealized_conversion_cast %sitofp32 : f32 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.s.w %lhsi32 : (!riscv.reg) -> !riscv.freg
    %fp32tosi = arith.fptosi %lhsf32 : f32 to i32
    %fp32tosi_1 = builtin.unrealized_conversion_cast %fp32tosi : i32 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.w.s %lhsf32_1 : (!riscv.freg) -> !riscv.reg
    %sitofp64 = arith.sitofp %lhsi32 : i32 to f64
    %sitofp64_1 = builtin.unrealized_conversion_cast %sitofp64 : f64 to !riscv.freg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.d.w %lhsi32 : (!riscv.reg) -> !riscv.freg
    "test.op"(%sitofp32_1, %fp32tosi_1, %sitofp64_1) : (!riscv.freg, !riscv.reg, !riscv.freg) -> ()
    // CHECK-NEXT: "test.op"(%sitofp32, %fp32tosi, %sitofp64) : (!riscv.freg, !riscv.reg, !riscv.freg) -> ()

    // arith.cmpf
    %cmpf0 = arith.cmpf false, %lhsf32, %rhsf32 : f32
    %cmpf0_1 = builtin.unrealized_conversion_cast %cmpf0 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = rv32.li 0 : !riscv.reg
    %cmpf1 = arith.cmpf oeq, %lhsf32, %rhsf32 : f32
    %cmpf1_1 = builtin.unrealized_conversion_cast %cmpf1 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf2 = arith.cmpf ogt, %lhsf32, %rhsf32 : f32
    %cmpf2_1 = builtin.unrealized_conversion_cast %cmpf2 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf3 = arith.cmpf oge, %lhsf32, %rhsf32 : f32
    %cmpf3_1 = builtin.unrealized_conversion_cast %cmpf3 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf4 = arith.cmpf olt, %lhsf32, %rhsf32 : f32
    %cmpf4_1 = builtin.unrealized_conversion_cast %cmpf4 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf5 = arith.cmpf ole, %lhsf32, %rhsf32 : f32
    %cmpf5_1 = builtin.unrealized_conversion_cast %cmpf5 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf6 = arith.cmpf one, %lhsf32, %rhsf32 : f32
    %cmpf6_3 = builtin.unrealized_conversion_cast %cmpf6 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %cmpf6_1, %cmpf6 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpf7 = arith.cmpf ord, %lhsf32, %rhsf32 : f32
    %cmpf7_3 = builtin.unrealized_conversion_cast %cmpf7 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %rhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %cmpf7_1, %cmpf7 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpf8 = arith.cmpf ueq, %lhsf32, %rhsf32 : f32
    %cmpf8_4 = builtin.unrealized_conversion_cast %cmpf8 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %cmpf8_1, %cmpf8 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf8_2, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf9 = arith.cmpf ugt, %lhsf32, %rhsf32 : f32
    %cmpf9_2 = builtin.unrealized_conversion_cast %cmpf9 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf9, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf10 = arith.cmpf uge, %lhsf32, %rhsf32 : f32
    %cmpf10_2 = builtin.unrealized_conversion_cast %cmpf10 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf10, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf11 = arith.cmpf ult, %lhsf32, %rhsf32 : f32
    %cmpf11_2 = builtin.unrealized_conversion_cast %cmpf11 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf11, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf12 = arith.cmpf ule, %lhsf32, %rhsf32 : f32
    %cmpf12_2 = builtin.unrealized_conversion_cast %cmpf12 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf12, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf13 = arith.cmpf une, %lhsf32, %rhsf32 : f32
    %cmpf13_2 = builtin.unrealized_conversion_cast %cmpf13 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf13, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf14 = arith.cmpf uno, %lhsf32, %rhsf32 : f32
    %cmpf14_4 = builtin.unrealized_conversion_cast %cmpf14 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %rhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %cmpf14_1, %cmpf14 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf14_2, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf15 = arith.cmpf true, %lhsf32, %rhsf32 : f32
    %cmpf15_1 = builtin.unrealized_conversion_cast %cmpf15 : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = rv32.li 1 : !riscv.reg
    "test.op"(%cmpf0_1, %cmpf1_1, %cmpf2_1, %cmpf3_1, %cmpf4_1, %cmpf5_1, %cmpf6_3, %cmpf7_3, %cmpf8_4, %cmpf9_2, %cmpf10_2, %cmpf11_2, %cmpf12_2, %cmpf13_2, %cmpf14_4, %cmpf15_1) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: "test.op"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()

    // arith.cmpf fastmath<fast>
    %cmpf1_fm = arith.cmpf oeq, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf1_fm_1 = builtin.unrealized_conversion_cast %cmpf1_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf2_fm = arith.cmpf ogt, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf2_fm_1 = builtin.unrealized_conversion_cast %cmpf2_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf3_fm = arith.cmpf oge, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf3_fm_1 = builtin.unrealized_conversion_cast %cmpf3_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf4_fm = arith.cmpf olt, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf4_fm_1 = builtin.unrealized_conversion_cast %cmpf4_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf5_fm = arith.cmpf ole, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf5_fm_1 = builtin.unrealized_conversion_cast %cmpf5_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf6_fm = arith.cmpf one, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf6_fm_3 = builtin.unrealized_conversion_cast %cmpf6_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %cmpf6_fm_1, %cmpf6_fm : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpf7_fm = arith.cmpf ord, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf7_fm_3 = builtin.unrealized_conversion_cast %cmpf7_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %rhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %cmpf7_fm_1, %cmpf7_fm : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpf8_fm = arith.cmpf ueq, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf8_fm_4 = builtin.unrealized_conversion_cast %cmpf8_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %cmpf8_fm_1, %cmpf8_fm : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf8_fm_2, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf9_fm = arith.cmpf ugt, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf9_fm_2 = builtin.unrealized_conversion_cast %cmpf9_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf9_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf10_fm = arith.cmpf uge, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf10_fm_2 = builtin.unrealized_conversion_cast %cmpf10_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf10_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf11_fm = arith.cmpf ult, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf11_fm_2 = builtin.unrealized_conversion_cast %cmpf11_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf11_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf12_fm = arith.cmpf ule, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf12_fm_2 = builtin.unrealized_conversion_cast %cmpf12_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf12_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf13_fm = arith.cmpf une, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf13_fm_2 = builtin.unrealized_conversion_cast %cmpf13_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf13_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf14_fm = arith.cmpf uno, %lhsf32, %rhsf32 fastmath<fast> : f32
    %cmpf14_fm_4 = builtin.unrealized_conversion_cast %cmpf14_fm : i1 to !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %rhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %cmpf14_fm_1, %cmpf14_fm : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf14_fm_2, 1 : (!riscv.reg) -> !riscv.reg
    "test.op"(%cmpf1_fm_1, %cmpf2_fm_1, %cmpf3_fm_1, %cmpf4_fm_1, %cmpf5_fm_1, %cmpf6_fm_3, %cmpf7_fm_3, %cmpf8_fm_4, %cmpf9_fm_2, %cmpf10_fm_2, %cmpf11_fm_2, %cmpf12_fm_2, %cmpf13_fm_2, %cmpf14_fm_4) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()
    // CHECK-NEXT: "test.op"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg, !riscv.reg) -> ()

    // No-op, gets deleted
    %index_cast = arith.index_cast %lhsindex : index to i32
    // CHECK-NEXT: }
}
