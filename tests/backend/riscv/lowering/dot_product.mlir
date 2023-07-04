"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {value = 1.000000e+00 : f32} : () -> f32
    %1 = "arith.constant"() {value = 2.000000e+00 : f32} : () -> f32
    %2 = "arith.constant"() {value = 3.000000e+00 : f32} : () -> f32
    %3 = "arith.constant"() {value = 0.000000e+00 : f32} : () -> f32
    %4 = "memref.alloca"() {operand_segment_sizes = array<i32: 0, 0>} : () -> memref<3xf32>
    %5 = "memref.alloca"() {operand_segment_sizes = array<i32: 0, 0>} : () -> memref<3xf32>
    %6 = "arith.constant"() {value = 0 : index} : () -> index
    "memref.store"(%0, %5, %6) : (f32, memref<3xf32>, index) -> ()
    %7 = "arith.constant"() {value = 1 : index} : () -> index
    "memref.store"(%1, %5, %7) : (f32, memref<3xf32>, index) -> ()
    %8 = "arith.constant"() {value = 2 : index} : () -> index
    "memref.store"(%2, %5, %8) : (f32, memref<3xf32>, index) -> ()
    %9 = "arith.constant"() {value = 0 : index} : () -> index
    "memref.store"(%0, %4, %9) : (f32, memref<3xf32>, index) -> ()
    %10 = "arith.constant"() {value = 1 : index} : () -> index
    "memref.store"(%1, %4, %10) : (f32, memref<3xf32>, index) -> ()
    %11 = "arith.constant"() {value = 2 : index} : () -> index
    "memref.store"(%2, %4, %11) : (f32, memref<3xf32>, index) -> ()
    %12 = "arith.constant"() {value = 0 : index} : () -> index
    %13 = "arith.constant"() {value = 3 : index} : () -> index
    %14 = "arith.constant"() {value = 1 : index} : () -> index
    "cf.br"(%12, %3)[^bb1] : (index, f32) -> ()
  ^bb1(%15: index, %16: f32):  // 2 preds: ^bb0, ^bb2
    %17 = "arith.cmpi"(%15, %13) {predicate = 2 : i64} : (index, index) -> i1
    "cf.cond_br"(%17)[^bb2, ^bb3] {operand_segment_sizes = array<i32: 1, 0, 0>} : (i1) -> ()
  ^bb2:  // pred: ^bb1
    %18 = "memref.load"(%5, %15) : (memref<3xf32>, index) -> f32
    %19 = "memref.load"(%4, %15) : (memref<3xf32>, index) -> f32
    %20 = "arith.mulf"(%18, %19) : (f32, f32) -> f32
    %21 = "arith.addf"(%16, %20) : (f32, f32) -> f32
    %22 = "arith.addi"(%15, %14) : (index, index) -> index
    "cf.br"(%22, %21)[^bb1] : (index, f32) -> ()
  ^bb3:  // pred: ^bb1
    %23 = "arith.fptosi"(%16) : (f32) -> i32
    "func.return"(%23) : (i32) -> ()
  }) {function_type = () -> i32, sym_name = "main"} : () -> ()
}) : () -> ()
