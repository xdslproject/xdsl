"builtin.module"() ({
  "func.func"() <{function_type = (memref<?x?xi64>, memref<?x?xi64>) -> memref<?x?xi64>, sym_name = "matmul", sym_visibility = "private"}> ({
  ^bb0(%arg0: memref<?x?xi64>, %arg1: memref<?x?xi64>):
    %0 = "arith.constant"() <{value = 0 : index}> : () -> index
    %1 = "arith.constant"() <{value = 1 : index}> : () -> index
    %2 = "memref.dim"(%arg0, %0) : (memref<?x?xi64>, index) -> index
    %3 = "memref.dim"(%arg0, %1) : (memref<?x?xi64>, index) -> index
    %4 = "memref.dim"(%arg1, %0) : (memref<?x?xi64>, index) -> index
    %5 = "memref.dim"(%arg1, %1) : (memref<?x?xi64>, index) -> index
    %6 = "memref.alloca"(%2, %5) <{alignment = 0 : i64, operandSegmentSizes = array<i32: 2, 0>}> : (index, index) -> memref<?x?xi64>
    %7 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    "scf.for"(%0, %2, %1) ({
    ^bb0(%arg2: index):
      "scf.for"(%0, %4, %1) ({
      ^bb0(%arg3: index):
        "memref.store"(%7, %6, %arg2, %arg3) : (i64, memref<?x?xi64>, index, index) -> ()
        "scf.for"(%0, %3, %1) ({
        ^bb0(%arg4: index):
          %8 = "memref.load"(%arg0, %arg2, %arg4) : (memref<?x?xi64>, index, index) -> i64
          %9 = "memref.load"(%arg1, %arg4, %arg3) : (memref<?x?xi64>, index, index) -> i64
          %10 = "memref.load"(%6, %arg2, %arg3) : (memref<?x?xi64>, index, index) -> i64
          %11 = "arith.muli"(%8, %9) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
          %12 = "arith.addi"(%10, %11) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
          "memref.store"(%12, %6, %arg2, %arg3) : (i64, memref<?x?xi64>, index, index) -> ()
          "scf.yield"() : () -> ()
        }) : (index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) : (index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"(%6) : (memref<?x?xi64>) -> ()
  }) : () -> ()
}) : () -> ()
