// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({
  %lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)
  %lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
  %lhsi64, %rhsi64 = "test.op"() : () -> (i64, i64)
  %lhsindex, %rhsindex = "test.op"() : () -> (index, index)
  %lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
  %lhsf64, %rhsf64 = "test.op"() : () -> (f64, f64)
  %lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)


}) : () -> ()
