// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({
  %lhsi1, %rhsi1 = "test.op"() : () -> (i1, i1)
  %lhsi32, %rhsi32 = "test.op"() : () -> (i32, i32)
  %lhsi64, %rhsi64 = "test.op"() : () -> (i64, i64)
  %lhsindex, %rhsindex = "test.op"() : () -> (index, index)
  %lhsf32, %rhsf32 = "test.op"() : () -> (f32, f32)
  %lhsf64, %rhsf64 = "test.op"() : () -> (f64, f64)
  %lhsvec, %rhsvec = "test.op"() : () -> (vector<4xf32>, vector<4xf32>)

  %add = comb.add %lhsi32, %rhsi32 : i32
  // CHECK: %add = comb.add %lhsi32, %rhsi32 : i32

  %mul =  comb.mul %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %mul =  comb.mul %lhsi32, %rhsi32 : i32

  %divu = comb.divu %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %divu = comb.divu %lhsi32, %rhsi32 : i32

  %divs = comb.divs %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %divs = comb.divs %lhsi32, %rhsi32 : i32

  %modu = comb.modu %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %modu = comb.modu %lhsi32, %rhsi32 : i32

  %mods = comb.mods %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %mods = comb.mods %lhsi32, %rhsi32 : i32

  %shl = comb.shl %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %shl = comb.shl %lhsi32, %rhsi32 : i32

  %shru = comb.shru %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %shru = comb.shru %lhsi32, %rhsi32 : i32

  %shrs = comb.shrs %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %shrs = comb.shrs %lhsi32, %rhsi32 : i32

  %sub = comb.sub %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %sub = comb.sub %lhsi32, %rhsi32 : i32

  %and = comb.and %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %and = comb.and %lhsi32, %rhsi32 : i32

  %or = comb.or %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %or = comb.or %lhsi32, %rhsi32 : i32

  %xor = comb.xor %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %xor = comb.xor %lhsi32, %rhsi32 : i32

  %icmp = "comb.icmp"(%lhsi1, %rhsi1) {"predicate" = 2 : i64, "two_state"} : (i1, i1) -> i1
  // CHECK-NEXT: %icmp = comb.icmp slt, %lhsi1, %rhsi1 : i1

  %parity = comb.parity %lhsi1 : i1
  // CHECK-NEXT: %parity = comb.parity %lhsi1 : i1

  %extract = comb.extract %lhsi32 from 1 : (i32) -> i32
  // CHECK-NEXT: %extract = comb.extract %lhsi32 : i32

  %concat = comb.concat %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %concat = comb.concat %lhsi32, %rhsi32 : i32

  %mux = comb.mux %lhsi1, %lhsi32, %rhsi32 : i32
  // CHECK-NEXT: %mux = comb.mux %lhsi1, %lhsi32, %rhsi32 : i32

  %replicate = comb.replicate %lhsi32 : (i32) -> i32
  // CHECK-NEXT: %replicate = comb.replicate %lhsi32 : i32
  }) : () -> ()
  