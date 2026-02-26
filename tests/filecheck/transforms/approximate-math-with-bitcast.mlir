// RUN: xdsl-opt -p approximate-math-with-bitcast %s | filecheck %s


// CHECK-LABEL: @test_log
func.func @test_log (%x: f32) -> f32 {
  %a = math.log %x : f32
//CHECK:      %a = arith.constant 0.693147182 : f32
//CHECK-NEXT: %a_1 = arith.constant 0x4B000000 : f32
//CHECK-NEXT: %a_2 = arith.constant 1.06497574e+09 : f32
//CHECK-NEXT: %a_3 = arith.mulf %a_1, %x : f32
//CHECK-NEXT: %a_4 = arith.addf %a_2, %a_3 : f32
//CHECK-NEXT: %a_5 = arith.fptosi %a_4 : f32 to i32
//CHECK-NEXT: %a_6 = arith.bitcast %a_5 : i32 to f32
//CHECK-NEXT: %a_7 = arith.mulf %a, %a_6 : f32
  return %a : f32
//CHECK-NEXT: return %a_7 : f32
}

// CHECK-LABEL: @test_log1p
func.func @test_log1p (%x: f32) -> f32 {
  %a = math.log1p %x : f32
// CHECK:      %a = arith.constant 1.000000e+00 : f32
// CHECK-NEXT: %a_1 = arith.addf %a, %x : f32
// CHECK-NEXT: %a_2 = arith.constant 0.693147182 : f32
// CHECK-NEXT: %a_3 = arith.constant 0x4B000000 : f32
// CHECK-NEXT: %a_4 = arith.constant 1.06497574e+09 : f32
// CHECK-NEXT: %a_5 = arith.mulf %a_3, %a_1 : f32
// CHECK-NEXT: %a_6 = arith.addf %a_4, %a_5 : f32
// CHECK-NEXT: %a_7 = arith.fptosi %a_6 : f32 to i32
// CHECK-NEXT: %a_8 = arith.bitcast %a_7 : i32 to f32
// CHECK-NEXT: %a_9 = arith.mulf %a_2, %a_8 : f32
  return %a : f32
// CHECK-NEXT: func.return %a_9 : f32
}

//CHECK-LABEL: @test_exp
func.func @test_exp (%x: f32) -> f32 {
  %b = math.exp %x fastmath<fast>: f32
// CHECK:      %b = arith.constant 1.44269502 : f32
// CHECK-NEXT: %b_1 = arith.mulf %b, %x fastmath<fast> : f32
// CHECK-NEXT: %b_2 = arith.constant 1.1920929e-07 : f32
// CHECK-NEXT: %b_3 = arith.constant -1.269550e+02 : f32
// CHECK-NEXT: %b_4 = arith.bitcast %b_1 : f32 to i32
// CHECK-NEXT: %b_5 = arith.sitofp %b_4 : i32 to f32
// CHECK-NEXT: %b_6 = arith.mulf %b_2, %b_5 fastmath<fast> : f32
// CHECK-NEXT: %b_7 = arith.addf %b_3, %b_6 fastmath<fast> : f32
  return %b : f32
// CHECK-NEXT: return %b_7 : f32
}
