"builtin.module"() ({
  "func.func"() ({
  ^0(%0 : memref<?x?x?xf64>, %1 : memref<?x?x?xf64>, %2 : memref<?x?x?xf64>):
    %3 = "memref.cast"(%0) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
    %4 = "memref.cast"(%1) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
    %5 = "memref.cast"(%2) : (memref<?x?x?xf64>) -> memref<72x72x72xf64>
    %6 = "arith.constant"() {"value" = 0 : index} : () -> index
    %7 = "arith.constant"() {"value" = 1 : index} : () -> index
    %8 = "arith.constant"() {"value" = 64 : index} : () -> index
    %9 = "arith.constant"() {"value" = 64 : index} : () -> index
    %10 = "arith.constant"() {"value" = 64 : index} : () -> index
    "scf.parallel"(%6, %6, %6, %8, %9, %10, %7, %7, %7) ({
    ^1(%11 : index, %12 : index, %13 : index):
      %14 = "arith.constant"() {"value" = 4 : index} : () -> index
      %15 = "arith.constant"() {"value" = 4 : index} : () -> index
      %16 = "arith.constant"() {"value" = 3 : index} : () -> index
      %17 = "arith.addi"(%13, %14) : (index, index) -> index
      %18 = "arith.addi"(%12, %15) : (index, index) -> index
      %19 = "arith.addi"(%11, %16) : (index, index) -> index
      %20 = "memref.load"(%3, %17, %18, %19) : (memref<72x72x72xf64>, index, index, index) -> f64
      %21 = "arith.constant"() {"value" = 4 : index} : () -> index
      %22 = "arith.constant"() {"value" = 4 : index} : () -> index
      %23 = "arith.constant"() {"value" = 5 : index} : () -> index
      %24 = "arith.addi"(%13, %21) : (index, index) -> index
      %25 = "arith.addi"(%12, %22) : (index, index) -> index
      %26 = "arith.addi"(%11, %23) : (index, index) -> index
      %27 = "memref.load"(%3, %24, %25, %26) : (memref<72x72x72xf64>, index, index, index) -> f64
      %28 = "arith.constant"() {"value" = 4 : index} : () -> index
      %29 = "arith.constant"() {"value" = 5 : index} : () -> index
      %30 = "arith.constant"() {"value" = 4 : index} : () -> index
      %31 = "arith.addi"(%13, %28) : (index, index) -> index
      %32 = "arith.addi"(%12, %29) : (index, index) -> index
      %33 = "arith.addi"(%11, %30) : (index, index) -> index
      %34 = "memref.load"(%3, %31, %32, %33) : (memref<72x72x72xf64>, index, index, index) -> f64
      %35 = "arith.constant"() {"value" = 4 : index} : () -> index
      %36 = "arith.constant"() {"value" = 3 : index} : () -> index
      %37 = "arith.constant"() {"value" = 4 : index} : () -> index
      %38 = "arith.addi"(%13, %35) : (index, index) -> index
      %39 = "arith.addi"(%12, %36) : (index, index) -> index
      %40 = "arith.addi"(%11, %37) : (index, index) -> index
      %41 = "memref.load"(%3, %38, %39, %40) : (memref<72x72x72xf64>, index, index, index) -> f64
      %42 = "arith.constant"() {"value" = 4 : index} : () -> index
      %43 = "arith.constant"() {"value" = 4 : index} : () -> index
      %44 = "arith.constant"() {"value" = 4 : index} : () -> index
      %45 = "arith.addi"(%13, %42) : (index, index) -> index
      %46 = "arith.addi"(%12, %43) : (index, index) -> index
      %47 = "arith.addi"(%11, %44) : (index, index) -> index
      %48 = "memref.load"(%3, %45, %46, %47) : (memref<72x72x72xf64>, index, index, index) -> f64
      %49 = "arith.addf"(%20, %27) : (f64, f64) -> f64
      %50 = "arith.addf"(%34, %41) : (f64, f64) -> f64
      %51 = "arith.addf"(%49, %50) : (f64, f64) -> f64
      %cst = "arith.constant"() {"value" = -4.0 : f64} : () -> f64
      %52 = "arith.mulf"(%48, %cst) : (f64, f64) -> f64
      %53 = "arith.addf"(%52, %51) : (f64, f64) -> f64
      %54 = "arith.constant"() {"value" = 4 : index} : () -> index
      %55 = "arith.constant"() {"value" = 4 : index} : () -> index
      %56 = "arith.constant"() {"value" = 4 : index} : () -> index
      %57 = "arith.addi"(%13, %54) : (index, index) -> index
      %58 = "arith.addi"(%12, %55) : (index, index) -> index
      %59 = "arith.addi"(%11, %56) : (index, index) -> index
      "memref.store"(%53, %4, %57, %58, %59) : (f64, memref<72x72x72xf64>, index, index, index) -> ()
      %60 = "arith.constant"() {"value" = 4 : index} : () -> index
      %61 = "arith.constant"() {"value" = 4 : index} : () -> index
      %62 = "arith.constant"() {"value" = 4 : index} : () -> index
      %63 = "arith.addi"(%13, %60) : (index, index) -> index
      %64 = "arith.addi"(%12, %61) : (index, index) -> index
      %65 = "arith.addi"(%11, %62) : (index, index) -> index
      "memref.store"(%53, %5, %63, %64, %65) : (f64, memref<72x72x72xf64>, index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
    "func.return"() : () -> ()
  }) {"function_type" = (memref<?x?x?xf64>, memref<?x?x?xf64>, memref<?x?x?xf64>) -> (), "sym_name" = "stencil_hdiff"} : () -> ()
}) : () -> ()


