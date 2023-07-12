// RUN: xdsl-opt -p lower-arith-to-riscv %s | xdsl-opt --print-op-generic | filecheck %s
"builtin.module"() ({
    // CHECK: "builtin.module"() ({
    %lhsi32 = "arith.constant"() {value = 1 : i32} : () -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
    %rhsi32 = "arith.constant"() {value = 2 : i32} : () -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.reg<x$>
    %lhsindex = "arith.constant"() {value = 1 : index} : () -> index
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
    %rhsindex = "arith.constant"() {value = 2 : index} : () -> index
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 2 : i32} : () -> !riscv.reg<x$>
    %lhsf32 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1065353216 : i32} : () -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.w"(%lhsf32) : (!riscv.reg<x$>) -> !riscv.freg<f$>
    %rhsf32 = "arith.constant"() {value = 2.000000e+00 : f32} : () -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1073741824 : i32} : () -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.w"(%rhsf32) : (!riscv.reg<x$>) -> !riscv.freg<f$>

    %addi32 = "arith.addi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.add"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %addindex = "arith.addi"(%lhsindex, %rhsindex) : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = "riscv.add"(%lhsindex, %rhsindex) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>

    %subi32 = "arith.subi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.sub"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %subindex = "arith.subi"(%lhsindex, %rhsindex) : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = "riscv.sub"(%lhsindex, %rhsindex) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>

    %muli32 = "arith.muli"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.mul"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %mulindex = "arith.muli"(%lhsindex, %rhsindex) : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = "riscv.mul"(%lhsindex, %rhsindex) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>

    %divui32 = "arith.divui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.divu"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %divsi32 = "arith.divsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.div"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>

    %remui = "arith.remui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.remu"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %remsi = "arith.remsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.rem"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>

    %andi32 = "arith.andi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.and"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %ori32 = "arith.ori"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.or"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %xori32 = "arith.xori"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.xor"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>

    %shli32 = "arith.shli"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.sll"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %shrui32 = "arith.shrui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.srl"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %shrsi32 = "arith.shrsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.sra"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>

    %cmpi0 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 0 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.xor"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.sltiu"(%cmpi0) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpi1 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 1 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<zero>
    // CHECK-NEXT: %{{.*}}= "riscv.xor"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%cmpi1, %cmpi1_1) : (!riscv.reg<zero>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpi2 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 2 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.slt"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpi3 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 3 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.slt"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpi3) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpi4 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 4 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpi5 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 5 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%lhsi32, %rhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpi5) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpi6 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 6 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%rhsi32, %lhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpi7 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 7 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%rhsi32, %lhsi32) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpi7) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>

    %addf32 = "arith.addf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fadd.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
    %subf32 = "arith.subf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fsub.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
    %mulf32 = "arith.mulf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fmul.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
    %divf32 = "arith.divf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fdiv.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>
    %negf32 = "arith.negf"(%rhsf32) : (f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fsgnjn.s"(%rhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.freg<f$>

    %sitofp = "arith.sitofp"(%lhsi32) : (i32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.w"(%lhsi32) : (!riscv.reg<x$>) -> !riscv.freg<f$>
    %fptosi = "arith.fptosi"(%lhsf32) : (f32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.fcvt.w.s"(%lhsf32_1) : (!riscv.freg<f$>) -> !riscv.reg<x$>

    %cmpf0 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 0 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 0 : i32} : () -> !riscv.reg<x$>
    %cmpf1 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 1 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    %cmpf2 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 2 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    %cmpf3 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 3 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    %cmpf4 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 4 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    %cmpf5 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 5 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    %cmpf6 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 6 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.or"(%cmpf6_1, %cmpf6) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpf7 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 7 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%lhsf32_1, %lhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%rhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.and"(%cmpf7_1, %cmpf7) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpf8 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 8 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.or"(%cmpf8_1, %cmpf8) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf8_2) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpf9 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 9 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf9) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpf10 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 10 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf10) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpf11 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 11 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf11) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpf12 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 12 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf12) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpf13 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 13 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf13) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpf14 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 14 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%lhsf32_1, %lhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%rhsf32_1, %rhsf32_1) : (!riscv.freg<f$>, !riscv.freg<f$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.and"(%cmpf14_1, %cmpf14) : (!riscv.reg<x$>, !riscv.reg<x$>) -> !riscv.reg<x$>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf14_2) {"immediate" = 1 : i32} : (!riscv.reg<x$>) -> !riscv.reg<x$>
    %cmpf15 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 15 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : i32} : () -> !riscv.reg<x$>
    %index_cast = "arith.index_cast"(%lhsindex) : (index) -> i32
    // CHECK-NEXT: }) : () -> ()
}) : () -> ()
