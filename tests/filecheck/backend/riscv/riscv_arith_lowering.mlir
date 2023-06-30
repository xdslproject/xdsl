// RUN: xdsl-opt -p lower-arith-to-riscv %s | xdsl-opt --print-op-generic | filecheck %s
"builtin.module"() ({
    %lhsi32 = "arith.constant"() {value = 1 : i32} : () -> i32
    // CHECK: %{{.*}} = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<>
    %rhsi32 = "arith.constant"() {value = 2 : i32} : () -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 2 : si32} : () -> !riscv.reg<>
    %lhsindex = "arith.constant"() {value = 1 : index} : () -> index
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<>
    %rhsindex = "arith.constant"() {value = 2 : index} : () -> index
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 2 : si32} : () -> !riscv.reg<>
    %lhsf32 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1065353216 : si32} : () -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.w"(%lhsf32) : (!riscv.reg<>) -> !riscv.freg<>
    %rhsf32 = "arith.constant"() {value = 2.000000e+00 : f32} : () -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1073741824 : si32} : () -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.w"(%rhsf32) : (!riscv.reg<>) -> !riscv.freg<>

    %addi32 = "arith.addi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.add"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %addindex = "arith.addi"(%lhsindex, %rhsindex) : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = "riscv.add"(%lhsindex, %rhsindex) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

    %subi32 = "arith.subi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.sub"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %subindex = "arith.subi"(%lhsindex, %rhsindex) : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = "riscv.sub"(%lhsindex, %rhsindex) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

    %muli32 = "arith.muli"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.mul"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %mulindex = "arith.muli"(%lhsindex, %rhsindex) : (index, index) -> index
    // CHECK-NEXT: %{{.*}} = "riscv.mul"(%lhsindex, %rhsindex) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

    %divui32 = "arith.divui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.divu"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %divsi32 = "arith.divsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.div"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

    %remui = "arith.remui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.remu"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %remsi = "arith.remsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.rem"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

    %andi32 = "arith.andi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.and"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %ori32 = "arith.ori"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.or"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %xori32 = "arith.xori"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.xor"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

    %shli32 = "arith.shli"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.sll"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %shrui32 = "arith.shrui"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.srl"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %shrsi32 = "arith.shrsi"(%lhsi32, %rhsi32) : (i32, i32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.sra"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>

    %cmpi0 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 0 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.xor"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.sltiu"(%cmpi0) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpi1 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 1 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.get_register"() : () -> !riscv.reg<zero>
    // CHECK-NEXT: %{{.*}}= "riscv.xor"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%cmpi1, %cmpi1_1) : (!riscv.reg<zero>, !riscv.reg<>) -> !riscv.reg<>
    %cmpi2 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 2 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.slt"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %cmpi3 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 3 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.slt"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpi3) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpi4 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 4 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %cmpi5 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 5 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%lhsi32, %rhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpi5) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpi6 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 6 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%rhsi32, %lhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %cmpi7 = "arith.cmpi"(%lhsi32, %rhsi32) {"predicate" = 7 : i32} : (i32, i32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.sltu"(%rhsi32, %lhsi32) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpi7) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>

    %addf32 = "arith.addf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fadd.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %subf32 = "arith.subf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fsub.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %mulf32 = "arith.mulf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fmul.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %divf32 = "arith.divf"(%lhsf32, %rhsf32) : (f32, f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fdiv.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>
    %negf32 = "arith.negf"(%rhsf32) : (f32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fsgnjn.s"(%rhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.freg<>

    %sitofp = "arith.sitofp"(%lhsi32) : (i32) -> f32
    // CHECK-NEXT: %{{.*}} = "riscv.fcvt.s.w"(%lhsi32) : (!riscv.reg<>) -> !riscv.freg<>
    %fptosi = "arith.fptosi"(%lhsf32) : (f32) -> i32
    // CHECK-NEXT: %{{.*}} = "riscv.fcvt.w.s"(%lhsf32_1) : (!riscv.freg<>) -> !riscv.reg<>

    %cmpf0 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 0 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 0 : si32} : () -> !riscv.reg<>
    %cmpf1 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 1 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    %cmpf2 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 2 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    %cmpf3 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 3 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    %cmpf4 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 4 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    %cmpf5 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 5 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    %cmpf6 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 6 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.or"(%cmpf6_1, %cmpf6) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %cmpf7 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 7 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%lhsf32_1, %lhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%rhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.and"(%cmpf7_1, %cmpf7) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    %cmpf8 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 8 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.or"(%cmpf8_1, %cmpf8) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf8_2) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpf9 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 9 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf9) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpf10 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 10 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf10) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpf11 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 11 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.fle.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf11) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpf12 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 12 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.flt.s"(%rhsf32_1, %lhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf12) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpf13 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 13 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%lhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf13) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpf14 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 14 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%lhsf32_1, %lhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.feq.s"(%rhsf32_1, %rhsf32_1) : (!riscv.freg<>, !riscv.freg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.and"(%cmpf14_1, %cmpf14) : (!riscv.reg<>, !riscv.reg<>) -> !riscv.reg<>
    // CHECK-NEXT: %{{.*}} = "riscv.xori"(%cmpf14_2) {"immediate" = 1 : si12} : (!riscv.reg<>) -> !riscv.reg<>
    %cmpf15 = "arith.cmpf"(%lhsf32, %rhsf32) {"predicate" = 15 : i32} : (f32, f32) -> i1
    // CHECK-NEXT: %{{.*}} = "riscv.li"() {"immediate" = 1 : si32} : () -> !riscv.reg<>
    %index_cast = "arith.index_cast"(%lhsindex) : (index) -> i32
    // CHECK-NEXT: }) : () -> ()
}) : () -> ()
