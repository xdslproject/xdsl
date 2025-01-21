"builtin.module"() ({
  %0:2 = "test.op"() : () -> (f32, f32)
  %1:2 = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)
  %2 = "arith.addf"(%0#0, %0#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %3 = "arith.addf"(%0#0, %0#1) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
  %4 = "arith.addf"(%1#0, %1#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  %5 = "arith.addf"(%1#0, %1#1) <{fastmath = #arith.fastmath<none>}> : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
  "test.op"(%2, %4) : (f32, vector<4xf32>) -> ()
  "func.func"() <{function_type = () -> (), sym_name = "test_const_const"}> ({
    %51 = "arith.constant"() <{value = 2.997900e+00 : f32}> : () -> f32
    %52 = "arith.constant"() <{value = 3.141500e+00 : f32}> : () -> f32
    %53 = "arith.addf"(%51, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %54 = "arith.subf"(%51, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %55 = "arith.mulf"(%51, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %56 = "arith.divf"(%51, %52) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    "test.op"(%53, %54, %55, %56) : (f32, f32, f32, f32) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "test_const_var_const"}> ({
    %42:2 = "test.op"() : () -> (f32, f32)
    %43 = "arith.constant"() <{value = 2.997900e+00 : f32}> : () -> f32
    %44 = "arith.constant"() <{value = 3.141500e+00 : f32}> : () -> f32
    %45 = "arith.constant"() <{value = 4.141500e+00 : f32}> : () -> f32
    %46 = "arith.constant"() <{value = 5.141500e+00 : f32}> : () -> f32
    %47 = "arith.mulf"(%42#0, %43) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %48 = "arith.mulf"(%47, %44) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
    %49 = "arith.mulf"(%42#0, %45) <{fastmath = #arith.fastmath<reassoc>}> : (f32, f32) -> f32
    %50 = "arith.mulf"(%49, %46) <{fastmath = #arith.fastmath<fast>}> : (f32, f32) -> f32
    "test.op"(%48, %50) : (f32, f32) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  %6:2 = "test.op"() : () -> (f32, f32)
  %7 = "arith.constant"() <{value = true}> : () -> i1
  %8 = "arith.constant"() <{value = false}> : () -> i1
  %9 = "arith.select"(%7, %6#0, %6#1) : (i1, f32, f32) -> f32
  %10 = "arith.select"(%7, %6#0, %6#1) : (i1, f32, f32) -> f32
  "test.op"(%9, %10) : (f32, f32) -> ()
  %11 = "test.op"() : () -> i1
  %12 = "arith.select"(%11, %7, %8) : (i1, i1, i1) -> i1
  %13 = "arith.select"(%11, %8, %7) : (i1, i1, i1) -> i1
  "test.op"(%12, %13) : (i1, i1) -> ()
  %14:2 = "test.op"() : () -> (i1, i64)
  %15 = "arith.select"(%14#0, %14#1, %14#1) : (i1, i64, i64) -> i64
  "test.op"(%15) : (i64) -> ()
  %16 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %17 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  %18 = "test.op"() : () -> i32
  %19 = "arith.muli"(%16, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %20 = "arith.muli"(%18, %16) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "test.op"(%19, %20) {"identity multiplication check"} : (i32, i32) -> ()
  %21 = "arith.muli"(%17, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "test.op"(%21) : (i32) -> ()
  %22 = "arith.muli"(%17, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "test.op"(%22) : (i32) -> ()
  %23 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  %24 = "arith.addi"(%23, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %25 = "arith.addi"(%18, %23) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "test.op"(%24, %25) {"identity addition check"} : (i32, i32) -> ()
  %26 = "arith.addi"(%17, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "test.op"(%26) : (i32) -> ()
  %27 = "arith.addi"(%17, %17) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "test.op"(%27) : (i32) -> ()
  %28 = "test.op"() : () -> i32
  %29 = "arith.cmpi"(%28, %28) <{predicate = 0 : i64}> : (i32, i32) -> i1
  %30 = "arith.cmpi"(%28, %28) <{predicate = 1 : i64}> : (i32, i32) -> i1
  %31 = "arith.cmpi"(%28, %28) <{predicate = 2 : i64}> : (i32, i32) -> i1
  %32 = "arith.cmpi"(%28, %28) <{predicate = 3 : i64}> : (i32, i32) -> i1
  %33 = "arith.cmpi"(%28, %28) <{predicate = 4 : i64}> : (i32, i32) -> i1
  %34 = "arith.cmpi"(%28, %28) <{predicate = 5 : i64}> : (i32, i32) -> i1
  %35 = "arith.cmpi"(%28, %28) <{predicate = 6 : i64}> : (i32, i32) -> i1
  %36 = "arith.cmpi"(%28, %28) <{predicate = 7 : i64}> : (i32, i32) -> i1
  %37 = "arith.cmpi"(%28, %28) <{predicate = 8 : i64}> : (i32, i32) -> i1
  %38 = "arith.cmpi"(%28, %28) <{predicate = 9 : i64}> : (i32, i32) -> i1
  "test.op"(%29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %28) : (i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i32) -> ()
  %39 = "arith.subi"(%17, %18) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "test.op"(%39) : (i32) -> ()
  %40 = "arith.constant"() <{value = true}> : () -> i1
  %41 = "arith.addi"(%40, %40) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1
  "test.op"(%41) : (i1) -> ()
}) : () -> ()
