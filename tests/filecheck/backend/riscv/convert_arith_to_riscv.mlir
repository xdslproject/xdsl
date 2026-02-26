// RUN: xdsl-opt -p convert-arith-to-riscv,reconcile-unrealized-casts %s | filecheck %s
builtin.module {
    %lhsi32 = "arith.constant"() {value = 1 : i32} : () -> i32
    // CHECK: %{{.*}} = rv32.li 1 : !riscv.reg
    %rhsi32 = "arith.constant"() {value = 2 : i32} : () -> i32
    // CHECK-NEXT: %{{.*}} = rv32.li 2 : !riscv.reg
    %lhsindex = "arith.constant"() {value = 1 : index} : () -> index
    // CHECK-NEXT: %{{.*}} = rv32.li 1 : !riscv.reg
    %rhsindex = "arith.constant"() {value = 2 : index} : () -> index
    // CHECK-NEXT: %{{.*}} = rv32.li 2 : !riscv.reg
    %lhsf32 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    // CHECK-NEXT: %{{.*}} = rv32.li 1065353216 : !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fmv.w.x %lhsf32 : (!riscv.reg) -> !riscv.freg
    %rhsf32 = "arith.constant"() {value = 2.000000e+00 : f32} : () -> f32
    // CHECK-NEXT: %{{.*}} = rv32.li 1073741824 : !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fmv.w.x %rhsf32 : (!riscv.reg) -> !riscv.freg

    %constf64zero = arith.constant 0.0 : f64
    // CHECK-NEXT: %{{.*}} = rv32.li 0 : !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.d.w %{{.*}} : (!riscv.reg) -> !riscv.freg

    %lhsf64_reg, %rhsf64_reg = "test.op"() : () -> (!riscv.freg, !riscv.freg)
    %lhsf64 = builtin.unrealized_conversion_cast %lhsf64_reg : !riscv.freg to f64
    %rhsf64 = builtin.unrealized_conversion_cast %rhsf64_reg : !riscv.freg to f64

    // CHECK-NEXT: %lhsf64_reg, %rhsf64_reg = "test.op"() : () -> (!riscv.freg, !riscv.freg)

    %f64 = "arith.constant"() {value = 1234.5678 : f64} : () -> f64
    // CHECK-NEXT: %{{.*}} = rv32.get_register : !riscv.reg<sp>
    // CHECK-NEXT: %{{.*}} = rv32.li 1083394629 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -4 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = rv32.li 1834810029 : !riscv.reg
    // CHECK-NEXT: riscv.sw %{{.*}}, %{{.*}}, -8 : (!riscv.reg<sp>, !riscv.reg) -> ()
    // CHECK-NEXT: %{{.*}} = riscv.fld %{{.*}}, -8 : (!riscv.reg<sp>) -> !riscv.freg

    %addi32 = "arith.addi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.add %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %addindex = "arith.addi"(%lhsindex, %rhsindex) : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = riscv.add %lhsindex, %rhsindex : (!riscv.reg, !riscv.reg) -> !riscv.reg

    %subi32 = "arith.subi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.sub %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %subindex = "arith.subi"(%lhsindex, %rhsindex) : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = riscv.sub %lhsindex, %rhsindex : (!riscv.reg, !riscv.reg) -> !riscv.reg

    %muli32 = "arith.muli"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.mul %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %mulindex = "arith.muli"(%lhsindex, %rhsindex) : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = riscv.mul %lhsindex, %rhsindex : (!riscv.reg, !riscv.reg) -> !riscv.reg

    %divui32 = "arith.divui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.divu %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %divsi32 = "arith.divsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.div %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg

    %remui = "arith.remui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.remu %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %remsi = "arith.remsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.rem %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg

    %andi32 = "arith.andi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.and %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %ori32 = "arith.ori"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.or %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %xori32 = "arith.xori"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.xor %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg

    %shli32 = "arith.shli"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.sll %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %shrui32 = "arith.shrui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.srl %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %shrsi32 = "arith.shrsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = riscv.sra %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg

    %cmpi0 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 0 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.xor %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltiu %cmpi0, 1 : (!riscv.reg) -> !riscv.reg
    %cmpi1 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 1 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = rv32.get_register : !riscv.reg<zero>
    // CHECK-NEXT: %{{.*}}= riscv.xor %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.sltu %cmpi1, %cmpi1_1 : (!riscv.reg<zero>, !riscv.reg) -> !riscv.reg
    %cmpi2 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 2 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.slt %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpi3 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 3 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.slt %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpi3, 1 : (!riscv.reg) -> !riscv.reg
    %cmpi4 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 4 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.sltu %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpi5 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 5 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.sltu %lhsi32, %rhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpi5, 1 : (!riscv.reg) -> !riscv.reg
    %cmpi6 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 6 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.sltu %rhsi32, %lhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpi7 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 7 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.sltu %rhsi32, %lhsi32 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpi7, 1 : (!riscv.reg) -> !riscv.reg

    %addf32 = "arith.addf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fadd.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf32 = "arith.subf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fsub.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf32 = "arith.mulf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fmul.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf32 = "arith.divf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %negf32 = "arith.negf"(%rhsf32) : (f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fsgnjn.s %rhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf32 = "arith.minimumf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fmin.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf32 = "arith.maximumf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fmax.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.freg

    // tests with fastmath flags when set to "fast"
    %addf32_fm = "arith.addf"(%lhsf32, %rhsf32) {"fastmath" = #arith.fastmath<fast>} : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fadd.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf32_fm = "arith.subf"(%lhsf32, %rhsf32) {"fastmath" = #arith.fastmath<fast>} : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fsub.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf32_fm = "arith.mulf"(%lhsf32, %rhsf32) {"fastmath" = #arith.fastmath<fast>} : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fmul.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf32_fm = "arith.divf"(%lhsf32, %rhsf32) {"fastmath" = #arith.fastmath<fast>} : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf32_fm = "arith.minimumf"(%lhsf32, %rhsf32) {"fastmath" = #arith.fastmath<fast>} : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fmin.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf32_fm = "arith.maximumf"(%lhsf32, %rhsf32) {"fastmath" = #arith.fastmath<fast>} : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = riscv.fmax.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %addf64 = "arith.addf"(%lhsf64, %rhsf64) : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fadd.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf64 = "arith.subf"(%lhsf64, %rhsf64) : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fsub.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf64 = "arith.mulf"(%lhsf64, %rhsf64) : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fmul.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf64 = "arith.divf"(%lhsf64, %rhsf64) : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf64 = "arith.minimumf"(%lhsf64, %rhsf64) : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fmin.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf64 = "arith.maximumf"(%lhsf64, %rhsf64) : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fmax.d %lhsf64_reg, %rhsf64_reg : (!riscv.freg, !riscv.freg) -> !riscv.freg

    // tests with fastmath flags when set to "fast"
    %addf64_fm = "arith.addf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<fast>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fadd.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf64_fm = "arith.subf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<fast>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fsub.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf64_fm = "arith.mulf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<fast>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fmul.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf64_fm = "arith.divf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<fast>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf64_fm = "arith.minimumf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<fast>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fmin.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf64_fm = "arith.maximumf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<fast>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fmax.d %lhsf64_reg, %rhsf64_reg fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.freg

    // tests with fastmath flags when set to "contract"
    %addf64_fm_contract = "arith.addf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<contract>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fadd.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %subf64_fm_contract = "arith.subf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<contract>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fsub.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %mulf64_fm_contract = "arith.mulf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<contract>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fmul.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %divf64_fm_contract = "arith.divf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<contract>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fdiv.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %minf64_fm_contract = "arith.minimumf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<contract>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fmin.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg
    %maxf64_fm_contract = "arith.maximumf"(%lhsf64, %rhsf64) {"fastmath" = #arith.fastmath<contract>} : (f64, f64) -> f64
    // CHECK-NEXT: %{{.*}} = riscv.fmax.d %lhsf64_reg, %rhsf64_reg fastmath<contract> : (!riscv.freg, !riscv.freg) -> !riscv.freg

    %sitofp32 = arith.sitofp %lhsi32 : i32 to f32
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.s.w %lhsi32 : (!riscv.reg) -> !riscv.freg
    %fp32tosi = arith.fptosi %lhsf32 : f32 to i32
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.w.s %lhsf32_1 : (!riscv.freg) -> !riscv.reg
    %sitofp64 = arith.sitofp %lhsi32 : i32 to f64
    // CHECK-NEXT: %{{.*}} = riscv.fcvt.d.w %lhsi32 : (!riscv.reg) -> !riscv.freg

    %cmpf0 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 0 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = rv32.li 0 : !riscv.reg
    %cmpf1 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 1 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf2 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 2 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf3 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 3 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf4 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 4 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf5 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 5 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf6 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 6 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %cmpf6_1, %cmpf6 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpf7 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 7 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %rhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %cmpf7_1, %cmpf7 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpf8 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 8 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %cmpf8_1, %cmpf8 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf8_2, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf9 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 9 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf9, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf10 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 10 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf10, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf11 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 11 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf11, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf12 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 12 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf12, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf13 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 13 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf13, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf14 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 14 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %lhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %rhsf32_1, %rhsf32_1 : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %cmpf14_1, %cmpf14 : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf14_2, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf15 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 15 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = rv32.li 1 : !riscv.reg

    // tests with fastmath flags when set to "fast"
    %cmpf1_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 1 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf2_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 2 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf3_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 3 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf4_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 4 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf5_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 5 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    %cmpf6_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 6 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %cmpf6_fm_1, %cmpf6_fm : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpf7_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 7 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %rhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %cmpf7_fm_1, %cmpf7_fm : (!riscv.reg, !riscv.reg) -> !riscv.reg
    %cmpf8_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 8 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.or %cmpf8_fm_1, %cmpf8_fm : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf8_fm_2, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf9_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 9 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf9_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf10_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 10 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf10_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf11_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 11 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.fle.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf11_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf12_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 12 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.flt.s %rhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf12_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf13_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 13 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf13_fm, 1 : (!riscv.reg) -> !riscv.reg
    %cmpf14_fm = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 14 : i32, "fastmath" = #arith.fastmath<fast>} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %lhsf32_1, %lhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.feq.s %rhsf32_1, %rhsf32_1 fastmath<fast> : (!riscv.freg, !riscv.freg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.and %cmpf14_fm_1, %cmpf14_fm : (!riscv.reg, !riscv.reg) -> !riscv.reg
    // CHECK-NEXT: %{{.*}} = riscv.xori %cmpf14_fm_2, 1 : (!riscv.reg) -> !riscv.reg
    %index_cast = "arith.index_cast"(%lhsindex) : (index) -> i32
    // CHECK-NEXT: }
}
