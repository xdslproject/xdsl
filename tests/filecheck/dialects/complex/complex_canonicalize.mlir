// RUN: xdsl-opt %s -p canonicalize | filecheck %s

%f16_0 = complex.constant [0.: f16, 0.: f16] : complex<f16>
%f16_1 = complex.constant [1.: f16, 3.: f16] : complex<f16>
%f16_2 = complex.constant [-1.: f16, -3.: f16] : complex<f16>
%f16_3 = complex.constant [2.: f16, 3.: f16] : complex<f16>

%f32_0 = complex.constant [0.: f32, 0.: f32] : complex<f32>
%f32_1 = complex.constant [-1.2: f32, -2.2: f32] : complex<f32>
%f32_2 = complex.constant [1.2: f32, 2.2: f32] : complex<f32>
%f32_3 = complex.constant [4.0: f32, 5.0: f32] : complex<f32>

"test.op"(%f16_0, %f16_1, %f16_2, %f16_3, %f32_0, %f32_1, %f32_2, %f32_3) : (complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f32>, complex<f32>, complex<f32>, complex<f32>) -> ()

// CHECK:       %f16 = complex.constant [0.000000e+00 : f16, 0.000000e+00 : f16] : complex<f16>
// CHECK-NEXT:  %f16_1 = complex.constant [1.000000e+00 : f16, 3.000000e+00 : f16] : complex<f16>
// CHECK-NEXT:  %f16_2 = complex.constant [-1.000000e+00 : f16, -3.000000e+00 : f16] : complex<f16>
// CHECK-NEXT:  %f16_3 = complex.constant [2.000000e+00 : f16, 3.000000e+00 : f16] : complex<f16>
// CHECK-NEXT:  %f32 = complex.constant [0.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
// CHECK-NEXT:  %f32_1 = complex.constant [-1.200000e+00 : f32, -2.200000e+00 : f32] : complex<f32>
// CHECK-NEXT:  %f32_2 = complex.constant [1.200000e+00 : f32, 2.200000e+00 : f32] : complex<f32>
// CHECK-NEXT:  %f32_3 = complex.constant [4.000000e+00 : f32, 5.000000e+00 : f32] : complex<f32>
// CHECK-NEXT:  "test.op"(%f16, %f16_1, %f16_2, %f16_3, %f32, %f32_1, %f32_2, %f32_3) : (complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f32>, complex<f32>, complex<f32>, complex<f32>) -> ()

%addf16 = complex.add %f16_1, %f16_2 : complex<f16>
%addf16_1 = complex.add %addf16, %f16_3 : complex<f16>
// CHECK:       %addf16 = complex.constant [0.000000e+00 : f16, 0.000000e+00 : f16] : complex<f16>
// CHECK-NEXT:  %addf16_1 = complex.constant [2.000000e+00 : f16, 3.000000e+00 : f16] : complex<f16>

%addf32 = complex.add %f32_1, %f32_2 : complex<f32>
%addf32_1 = complex.add %addf32, %f32_3 : complex<f32>
// CHECK:       %addf32 = complex.constant [0.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
// CHECK-NEXT:  %addf32_1 = complex.constant [4.000000e+00 : f32, 5.000000e+00 : f32] : complex<f32>

%subf16 = complex.sub %f16_1, %f16_2 : complex<f16>
%subf16_1 = complex.sub %subf16, %f16_3 : complex<f16>
// CHECK:       %subf16 = complex.constant [2.000000e+00 : f16, 6.000000e+00 : f16] : complex<f16>
// CHECK-NEXT:  %subf16_1 = complex.constant [0.000000e+00 : f16, 3.000000e+00 : f16] : complex<f16>

%subf32 = complex.sub %f32_1, %f32_2 : complex<f32>
%subf32_1 = complex.sub %subf32, %f32_3 : complex<f32>
// CHECK:       %subf32 = complex.constant [-2.400000e+00 : f32, -4.400000e+00 : f32] : complex<f32>
// CHECK-NEXT:  %subf32_1 = complex.constant [-6.400000e+00 : f32, -9.400000e+00 : f32] : complex<f32>

%mulf16 = complex.mul %f16_1, %f16_2 : complex<f16>
%mulf16_1 = complex.mul %mulf16, %f16_3 : complex<f16>
// CHECK:       %mulf16 = complex.constant [8.000000e+00 : f16, -6.000000e+00 : f16] : complex<f16>
// CHECK-NEXT:  %mulf16_1 = complex.constant [3.400000e+01 : f16, 1.200000e+01 : f16] : complex<f16>

%mulf32 = complex.mul %f32_1, %f32_2 : complex<f32>
%mulf32_1 = complex.mul %mulf32, %f32_3 : complex<f32>
// CHECK:       %mulf32 = complex.constant [3.400000e+00 : f32, -5.280000e+00 : f32] : complex<f32>
// CHECK-NEXT:  %mulf32_1 = complex.constant [4.000000e+01 : f32, -4.12000036 : f32] : complex<f32>

%divf16 = complex.div %f16_1, %f16_2 : complex<f16>
%divf16_1 = complex.div %divf16, %f16_3 : complex<f16>
// CHECK:       %divf16 = complex.constant [-1.000000e+00 : f16, 0.000000e+00 : f16] : complex<f16>
// CHECK-NEXT:  %divf16_1 = complex.constant [-1.538090e-01 : f16, 2.307130e-01 : f16] : complex<f16>

%divf32 = complex.div %f32_1, %f32_2 : complex<f32>
%divf32_1 = complex.div %divf32, %f32_3 : complex<f32>
// CHECK:       %divf32 = complex.constant [-1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
// CHECK-NEXT:  %divf32_1 = complex.constant [-0.097560972 : f32, 0.121951222 : f32] : complex<f32>

%div16 = complex.div %f16_0, %f16_0 : complex<f16>
%div16_inf = complex.div %f16_3, %f16_0 : complex<f16>
%div16_minus_inf = complex.div %f16_2, %f16_0 : complex<f16>
// CHECK:       %div16 = complex.constant [0x7e00 : f16, 0x7e00 : f16] : complex<f16>
// CHECK-NEXT:  %div16_inf = complex.constant [0x7c00 : f16, 0x7c00 : f16] : complex<f16>
// CHECK-NEXT:  %div16_minus_inf = complex.constant [0xfc00 : f16, 0xfc00 : f16] : complex<f16>

%div32 = complex.div %f32_0, %f32_0 : complex<f32>
%div32_inf = complex.div %f32_2, %f32_0 : complex<f32>
%div32_minus_inf = complex.div %f32_1, %f32_0 : complex<f32>
// CHECK:       %div32 = complex.constant [0x7fc00000 : f32, 0x7fc00000 : f32] : complex<f32>
// CHECK-NEXT:  %div32_inf = complex.constant [0x7f800000 : f32, 0x7f800000 : f32] : complex<f32>
// CHECK-NEXT:  %div32_minus_inf = complex.constant [0xff800000 : f32, 0xff800000 : f32] : complex<f32>

"test.op"(%addf16, %addf16_1, %subf16, %subf16_1, %mulf16, %mulf16_1, %divf16, %divf16_1) : (complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f16>) -> ()
"test.op"(%addf32, %addf32_1, %subf32, %subf32_1, %mulf32, %mulf32_1, %divf32, %divf32_1) : (complex<f32>, complex<f32>, complex<f32>, complex<f32>, complex<f32>, complex<f32>, complex<f32>, complex<f32>) -> ()
"test.op"(%div16, %div16_inf, %div16_minus_inf) : (complex<f16>, complex<f16>, complex<f16>) -> ()
"test.op"(%div32, %div32_inf, %div32_minus_inf) : (complex<f32>, complex<f32>, complex<f32>) -> ()
// CHECK:       "test.op"(%addf16, %addf16_1, %subf16, %subf16_1, %mulf16, %mulf16_1, %divf16, %divf16_1) : (complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f16>, complex<f16>) -> ()
// CHECK-NEXT:  "test.op"(%addf32, %addf32_1, %subf32, %subf32_1, %mulf32, %mulf32_1, %divf32, %divf32_1) : (complex<f32>, complex<f32>, complex<f32>, complex<f32>, complex<f32>, complex<f32>, complex<f32>, complex<f32>) -> ()
// CHECK-NEXT:  "test.op"(%div16, %div16_inf, %div16_minus_inf) : (complex<f16>, complex<f16>, complex<f16>) -> ()
// CHECK-NEXT:  "test.op"(%div32, %div32_inf, %div32_minus_inf) : (complex<f32>, complex<f32>, complex<f32>) -> ()


// CHECK-LABEL: func @test_create_of_real_and_imag
// CHECK-SAME: (%[[CPLX:.*]] : complex<f32>) -> complex<f32>
func.func @test_create_of_real_and_imag(%cplx: complex<f32>) -> complex<f32> {
  %real = complex.re %cplx : complex<f32>
  %imag = complex.im %cplx : complex<f32>
  %complex = complex.create %real, %imag : complex<f32>
  return %complex : complex<f32>

  // CHECK:  return %[[CPLX]] : complex<f32>
}


// CHECK-LABEL: func @test_create_of_real_and_imag2
func.func @test_create_of_real_and_imag2() -> complex<f32> {
  %cplx = "test.op"() : () -> (complex<f32>)
  %real = complex.re %cplx : complex<f32>
  %imag = complex.im %cplx : complex<f32>
  %complex = complex.create %real, %imag : complex<f32>
  return %complex : complex<f32>

  // CHECK:       %[[CPLX:.*]] = "test.op"() : () -> complex<f32>
  // CHECK-NEXT:  return %[[CPLX]] : complex<f32>
}


// CHECK-LABEL: func @test_create_of_real_and_imag_different_operand
// CHECK-SAME: (%[[CPLX:.*]] : complex<f32>, %[[CPLX2:.*]] : complex<f32>) -> complex<f32>
func.func @test_create_of_real_and_imag_different_operand(
    %cplx: complex<f32>, %cplx2 : complex<f32>) -> complex<f32> {
  %real = complex.re %cplx : complex<f32>
  %imag = complex.im %cplx2 : complex<f32>
  %complex = complex.create %real, %imag : complex<f32>
  return %complex: complex<f32>

  // CHECK:       %[[REAL:.*]] = complex.re %[[CPLX]] : complex<f32>
  // CHECK-NEXT:  %[[IMAG:.*]] = complex.im %[[CPLX2]] : complex<f32>
  // CHECK-NEXT:  %[[COMPLEX:.*]] = complex.create %[[REAL]], %[[IMAG]] : complex<f32>
  // CHECK-NEXT:  return %[[COMPLEX]] : complex<f32>
}


// CHECK-LABEL: func @test_real_of_const
func.func @test_real_of_const() -> f32 {
  %complex = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %real = complex.re %complex : complex<f32>
  return %real : f32

  // CHECK:       %[[REAL:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT:  return %[[REAL]] : f32
}


// CHECK-LABEL: func @test_real_of_create_op
func.func @test_real_of_create_op() -> f32 {
  %real = arith.constant 1.0 : f32
  %imag = arith.constant 0.0 : f32
  %complex = complex.create %real, %imag : complex<f32>
  %1 = complex.re %complex : complex<f32>
  return %1 : f32

  // CHECK:       %[[REAL:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT:  return %[[REAL]] : f32
}


// CHECK-LABEL: func @test_imag_of_const
func.func @test_imag_of_const() -> f32 {
  %complex = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %imag = complex.im %complex : complex<f32>
  return %imag : f32

  // CHECK:       %[[IMAG:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT:  return %[[IMAG]] : f32
}


// CHECK-LABEL: func @test_imag_of_create_op
func.func @test_imag_of_create_op() -> f32 {
  %real = arith.constant 1.0 : f32
  %imag = arith.constant 0.0 : f32
  %complex = complex.create %real, %imag : complex<f32>
  %1 = complex.im %complex : complex<f32>
  return %1 : f32

  // CHECK:       %[[IMAG:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT:  return %[[IMAG]] : f32
}


// CHECK-LABEL: func @test_re_neg
// CHECK-SAME: (%[[ARG0:.*]] : f32, %[[ARG1:.*]] : f32) -> f32
func.func @test_re_neg(%arg0: f32, %arg1: f32) -> f32 {
  %create = complex.create %arg0, %arg1: complex<f32>
  %neg = complex.neg %create : complex<f32>
  %re = complex.re %neg : complex<f32>
  return %re : f32

  // CHECK:       %[[RE:.*]] = arith.negf %[[ARG0]] : f32
  // CHECK-NEXT:  return %[[RE]] : f32
}


// CHECK-LABEL: func @test_im_neg
// CHECK-SAME: (%[[ARG0:.*]] : f32, %[[ARG1:.*]] : f32) -> f32
func.func @test_im_neg(%arg0: f32, %arg1: f32) -> f32 {
  %create = complex.create %arg0, %arg1: complex<f32>
  %neg = complex.neg %create : complex<f32>
  %im = complex.im %neg : complex<f32>
  return %im : f32

  // CHECK:       %[[IM:.*]] = arith.negf %[[ARG1]] : f32
  // CHECK-NEXT:  return %[[IM]] : f32
}


// CHECK-LABEL: func @test_complex_neg_neg
func.func @test_complex_neg_neg() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %neg1 = complex.neg %complex1 : complex<f32>
  %neg2 = complex.neg %neg1 : complex<f32>
  return %neg2 : complex<f32>

  // CHECK:       %[[COMPLEX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT:  return %[[COMPLEX]] : complex<f32>
}


// CHECK-LABEL: func @test_complex_log_exp
func.func @test_complex_log_exp() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %exp = complex.exp %complex1 : complex<f32>
  %log = complex.log %exp : complex<f32>
  return %log : complex<f32>

  // CHECK:       %[[COMPLEX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT:  return %[[COMPLEX]] : complex<f32>  
}


// CHECK-LABEL: func @test_complex_exp_log
func.func @test_complex_exp_log() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %log = complex.log %complex1 : complex<f32>
  %exp = complex.exp %log : complex<f32>
  return %exp : complex<f32>

  // CHECK:       %[[COMPLEX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT:  return %[[COMPLEX]] : complex<f32>   
}


// CHECK-LABEL: func @test_complex_conj_conj
func.func @test_complex_conj_conj() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %conj1 = complex.conj %complex1 : complex<f32>
  %conj2 = complex.conj %conj1 : complex<f32>
  return %conj2 : complex<f32>

  // CHECK:       %[[COMPLEX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT:  return %[[COMPLEX]] : complex<f32>  
}
