// RUN: xdsl-opt -p convert-arith-to-llvm %s | filecheck %s
// RUN: xdsl-opt -p convert-arith-to-llvm --print-op-generic %s | filecheck %s --check-prefix=GENERIC

// CHECK-LABEL: func.func @integer
func.func @integer(%a : i32, %b : i32) {
  // CHECK-NEXT: %addi = llvm.add %a, %b : i32
  %addi = arith.addi %a, %b : i32
  // CHECK-NEXT: %subi = llvm.sub %a, %b : i32
  %subi = arith.subi %a, %b : i32
  // CHECK-NEXT: %muli = llvm.mul %a, %b : i32
  %muli = arith.muli %a, %b : i32
  // CHECK-NEXT: %divsi = llvm.sdiv %a, %b : i32
  %divsi = arith.divsi %a, %b : i32
  // CHECK-NEXT: %divui = llvm.udiv %a, %b : i32
  %divui = arith.divui %a, %b : i32
  // CHECK-NEXT: %remsi = llvm.srem %a, %b : i32
  %remsi = arith.remsi %a, %b : i32
  // CHECK-NEXT: %remui = llvm.urem %a, %b : i32
  %remui = arith.remui %a, %b : i32
  // CHECK-NEXT: %andi = llvm.and %a, %b : i32
  %andi = arith.andi %a, %b : i32
  // CHECK-NEXT: %ori = llvm.or %a, %b : i32
  %ori = arith.ori %a, %b : i32
  // CHECK-NEXT: %xori = llvm.xor %a, %b : i32
  %xori = arith.xori %a, %b : i32
  // CHECK-NEXT: %shli = llvm.shl %a, %b : i32
  %shli = arith.shli %a, %b : i32
  // CHECK-NEXT: %shrsi = llvm.ashr %a, %b : i32
  %shrsi = arith.shrsi %a, %b : i32
  // CHECK-NEXT: %shrui = llvm.lshr %a, %b : i32
  %shrui = arith.shrui %a, %b : i32
  // CHECK-NEXT: %maxsi = llvm.intr.smax(%a, %b) : (i32, i32) -> i32
  %maxsi = arith.maxsi %a, %b : i32
  // CHECK-NEXT: %maxui = llvm.intr.umax(%a, %b) : (i32, i32) -> i32
  %maxui = arith.maxui %a, %b : i32
  // CHECK-NEXT: %minsi = llvm.intr.smin(%a, %b) : (i32, i32) -> i32
  %minsi = arith.minsi %a, %b : i32
  // CHECK-NEXT: %minui = llvm.intr.umin(%a, %b) : (i32, i32) -> i32
  %minui = arith.minui %a, %b : i32
  // CHECK-NEXT: "test.op"(%addi, %subi, %muli, %divsi, %divui, %remsi, %remui, %andi, %ori, %xori, %shli, %shrsi, %shrui, %maxsi, %maxui, %minsi, %minui) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
  "test.op"(%addi, %subi, %muli, %divsi, %divui, %remsi, %remui, %andi, %ori, %xori, %shli, %shrsi, %shrui, %maxsi, %maxui, %minsi, %minui) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
  func.return
}

// CHECK-LABEL: func.func @integer_overflow
func.func @integer_overflow(%a : i64, %b : i64) {
  // CHECK-NEXT: %addi = llvm.add %a, %b overflow<nsw> : i64
  %addi = arith.addi %a, %b overflow<nsw> : i64
  // CHECK-NEXT: %subi = llvm.sub %a, %b overflow<nuw> : i64
  %subi = arith.subi %a, %b overflow<nuw> : i64
  // CHECK-NEXT: %muli = llvm.mul %a, %b overflow<nsw,nuw> : i64
  %muli = arith.muli %a, %b overflow<nsw, nuw> : i64
  // CHECK-NEXT: %shli = llvm.shl %a, %b overflow<nsw> : i64
  %shli = arith.shli %a, %b overflow<nsw> : i64
  // CHECK-NEXT: "test.op"(%addi, %subi, %muli, %shli) : (i64, i64, i64, i64) -> ()
  "test.op"(%addi, %subi, %muli, %shli) : (i64, i64, i64, i64) -> ()
  func.return
}

// CHECK-LABEL: func.func @float
func.func @float(%a : f32, %b : f32) {
  // CHECK-NEXT: %addf = llvm.fadd %a, %b : f32
  %addf = arith.addf %a, %b : f32
  // CHECK-NEXT: %subf = llvm.fsub %a, %b : f32
  %subf = arith.subf %a, %b : f32
  // CHECK-NEXT: %mulf = llvm.fmul %a, %b : f32
  %mulf = arith.mulf %a, %b : f32
  // CHECK-NEXT: %divf = llvm.fdiv %a, %b : f32
  %divf = arith.divf %a, %b : f32
  // CHECK-NEXT: %negf = llvm.fneg %a : f32
  %negf = arith.negf %a : f32
  // CHECK-NEXT: %maximumf = llvm.intr.maximum(%a, %b) : (f32, f32) -> f32
  %maximumf = arith.maximumf %a, %b : f32
  // CHECK-NEXT: %maxnumf = llvm.intr.maxnum(%a, %b) : (f32, f32) -> f32
  %maxnumf = arith.maxnumf %a, %b : f32
  // CHECK-NEXT: %minimumf = llvm.intr.minimum(%a, %b) : (f32, f32) -> f32
  %minimumf = arith.minimumf %a, %b : f32
  // CHECK-NEXT: %minnumf = llvm.intr.minnum(%a, %b) : (f32, f32) -> f32
  %minnumf = arith.minnumf %a, %b : f32
  // CHECK-NEXT: "test.op"(%addf, %subf, %mulf, %divf, %negf, %maximumf, %maxnumf, %minimumf, %minnumf) : (f32, f32, f32, f32, f32, f32, f32, f32, f32) -> ()
  "test.op"(%addf, %subf, %mulf, %divf, %negf, %maximumf, %maxnumf, %minimumf, %minnumf) : (f32, f32, f32, f32, f32, f32, f32, f32, f32) -> ()
  func.return
}

// The fast-math flags of `llvm.fcmp` are not printed in its custom format, so they are
// checked in the generic format.

// CHECK-LABEL: func.func @float_fastmath
func.func @float_fastmath(%a : f64, %b : f64) {
  // CHECK-NEXT: %addf = llvm.fadd %a, %b {fastmathFlags = #llvm.fastmath<fast>} : f64
  %addf = arith.addf %a, %b fastmath<fast> : f64
  // CHECK-NEXT: %negf = llvm.fneg %a {fastmathFlags = #llvm.fastmath<nnan,ninf>} : f64
  %negf = arith.negf %a fastmath<nnan,ninf> : f64
  // CHECK-NEXT: %maximumf = llvm.intr.maximum(%a, %b) {fastmathFlags = #llvm.fastmath<nsz>} : (f64, f64) -> f64
  %maximumf = arith.maximumf %a, %b fastmath<nsz> : f64
  // CHECK-NEXT: %cmpf = llvm.fcmp "olt" %a, %b : f64
  // GENERIC: %cmpf = "llvm.fcmp"(%a, %b) <{fastmathFlags = #llvm.fastmath<nnan>, predicate = 4 : i64}> : (f64, f64) -> i1
  %cmpf = arith.cmpf olt, %a, %b fastmath<nnan> : f64
  // CHECK-NEXT: "test.op"(%addf, %negf, %maximumf, %cmpf) : (f64, f64, f64, i1) -> ()
  "test.op"(%addf, %negf, %maximumf, %cmpf) : (f64, f64, f64, i1) -> ()
  func.return
}

// CHECK-LABEL: func.func @comparison
func.func @comparison(%a : i16, %b : i16, %c : bf16, %d : bf16) {
  // CHECK-NEXT: %eq = llvm.icmp "eq" %a, %b : i16
  %eq = arith.cmpi eq, %a, %b : i16
  // CHECK-NEXT: %slt = llvm.icmp "slt" %a, %b : i16
  %slt = arith.cmpi slt, %a, %b : i16
  // CHECK-NEXT: %uge = llvm.icmp "uge" %a, %b : i16
  %uge = arith.cmpi uge, %a, %b : i16
  // CHECK-NEXT: %false = llvm.fcmp "_false" %c, %d : bf16
  %false = arith.cmpf false, %c, %d : bf16
  // CHECK-NEXT: %ueq = llvm.fcmp "ueq" %c, %d : bf16
  %ueq = arith.cmpf ueq, %c, %d : bf16
  // CHECK-NEXT: %true = llvm.fcmp "_true" %c, %d : bf16
  %true = arith.cmpf true, %c, %d : bf16
  // CHECK-NEXT: %select = llvm.select %eq, %a, %b : i1, i16
  %select = arith.select %eq, %a, %b : i16
  // CHECK-NEXT: "test.op"(%eq, %slt, %uge, %false, %ueq, %true, %select) : (i1, i1, i1, i1, i1, i1, i16) -> ()
  "test.op"(%eq, %slt, %uge, %false, %ueq, %true, %select) : (i1, i1, i1, i1, i1, i1, i16) -> ()
  func.return
}

// CHECK-LABEL: func.func @casts
func.func @casts(%a : i32, %b : f32) {
  // CHECK-NEXT: %extsi = llvm.sext %a : i32 to i64
  %extsi = arith.extsi %a : i32 to i64
  // CHECK-NEXT: %extui = llvm.zext %a : i32 to i64
  %extui = arith.extui %a : i32 to i64
  // CHECK-NEXT: %trunci = llvm.trunc %a : i32 to i8
  %trunci = arith.trunci %a : i32 to i8
  // CHECK-NEXT: %extf = llvm.fpext %b : f32 to f64
  %extf = arith.extf %b : f32 to f64
  // CHECK-NEXT: %truncf = llvm.fptrunc %b : f32 to f16
  %truncf = arith.truncf %b : f32 to f16
  // CHECK-NEXT: %sitofp = llvm.sitofp %a : i32 to f64
  %sitofp = arith.sitofp %a : i32 to f64
  // CHECK-NEXT: %uitofp = llvm.uitofp %a : i32 to f16
  %uitofp = arith.uitofp %a : i32 to f16
  // CHECK-NEXT: %fptosi = llvm.fptosi %b : f32 to i64
  %fptosi = arith.fptosi %b : f32 to i64
  // CHECK-NEXT: %fptoui = llvm.fptoui %b : f32 to i8
  %fptoui = arith.fptoui %b : f32 to i8
  // CHECK-NEXT: %bitcast = llvm.bitcast %b : f32 to i32
  %bitcast = arith.bitcast %b : f32 to i32
  // CHECK-NEXT: "test.op"(%extsi, %extui, %trunci, %extf, %truncf, %sitofp, %uitofp, %fptosi, %fptoui, %bitcast) : (i64, i64, i8, f64, f16, f64, f16, i64, i8, i32) -> ()
  "test.op"(%extsi, %extui, %trunci, %extf, %truncf, %sitofp, %uitofp, %fptosi, %fptoui, %bitcast) : (i64, i64, i8, f64, f16, f64, f16, i64, i8, i32) -> ()
  func.return
}

// CHECK-LABEL: func.func @constants
func.func @constants() {
  // CHECK-NEXT: %int = llvm.mlir.constant(42 : i32) : i32
  %int = arith.constant 42 : i32
  // CHECK-NEXT: %i64 = llvm.mlir.constant(-1) : i64
  %i64 = arith.constant -1 : i64
  // CHECK-NEXT: %bool = llvm.mlir.constant(true) : i1
  %bool = arith.constant true
  // CHECK-NEXT: %float = llvm.mlir.constant(1.500000e+00 : f32) : f32
  %float = arith.constant 1.5 : f32
  // CHECK-NEXT: %dense_int = llvm.mlir.constant(dense<[1, 2, 3, 4]> : vector<4xi32>) : vector<4xi32>
  %dense_int = arith.constant dense<[1, 2, 3, 4]> : vector<4xi32>
  // CHECK-NEXT: %dense_float = llvm.mlir.constant(dense<2.000000e+00> : vector<[2]xf64>) : vector<[2]xf64>
  %dense_float = arith.constant dense<2.0> : vector<[2]xf64>
  // CHECK-NEXT: "test.op"(%int, %i64, %bool, %float, %dense_int, %dense_float) : (i32, i64, i1, f32, vector<4xi32>, vector<[2]xf64>) -> ()
  "test.op"(%int, %i64, %bool, %float, %dense_int, %dense_float) : (i32, i64, i1, f32, vector<4xi32>, vector<[2]xf64>) -> ()
  func.return
}

// CHECK-LABEL: func.func @vector
func.func @vector(%a : vector<4xi32>, %b : vector<[2]xf64>, %c : i1) {
  // CHECK-NEXT: %addi = llvm.add %a, %a overflow<nuw> : vector<4xi32>
  %addi = arith.addi %a, %a overflow<nuw> : vector<4xi32>
  // CHECK-NEXT: %maxui = llvm.intr.umax(%a, %a) : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
  %maxui = arith.maxui %a, %a : vector<4xi32>
  // CHECK-NEXT: %mulf = llvm.fmul %b, %b {fastmathFlags = #llvm.fastmath<contract>} : vector<[2]xf64>
  %mulf = arith.mulf %b, %b fastmath<contract> : vector<[2]xf64>
  // CHECK-NEXT: %select = llvm.select %c, %a, %a : i1, vector<4xi32>
  %select = arith.select %c, %a, %a : vector<4xi32>
  // CHECK-NEXT: "test.op"(%addi, %maxui, %mulf, %select) : (vector<4xi32>, vector<4xi32>, vector<[2]xf64>, vector<4xi32>) -> ()
  "test.op"(%addi, %maxui, %mulf, %select) : (vector<4xi32>, vector<4xi32>, vector<[2]xf64>, vector<4xi32>) -> ()
  func.return
}

// Discardable attributes are forwarded to the converted operations.
// The custom format of `llvm.mlir.constant` does not print them, so they are checked in
// the generic format.

// CHECK-LABEL: func.func @discardable_attributes
func.func @discardable_attributes(%a : i32, %b : f32) {
  // CHECK-NEXT: %addi = llvm.add %a, %a {test.attr} : i32
  %addi = arith.addi %a, %a {test.attr} : i32
  // CHECK-NEXT: %addf = llvm.fadd %b, %b {test.attr, fastmathFlags = #llvm.fastmath<fast>} : f32
  %addf = arith.addf %b, %b fastmath<fast> {test.attr} : f32
  // CHECK-NEXT: %constant = llvm.mlir.constant(0 : i32) : i32
  // GENERIC: %constant = "llvm.mlir.constant"() <{value = 0 : i32}> {test.attr} : () -> i32
  %constant = arith.constant {test.attr} 0 : i32
  // CHECK-NEXT: "test.op"(%addi, %addf, %constant) : (i32, f32, i32) -> ()
  "test.op"(%addi, %addf, %constant) : (i32, f32, i32) -> ()
  func.return
}

// Operations without an LLVM lowering, or on types that are not LLVM-compatible, are
// left unchanged.

// CHECK-LABEL: func.func @unsupported
func.func @unsupported(%a : index, %b : i32, %c : tensor<4xi32>, %d : vector<2x4xf32>, %e : si32, %f : f8E4M3FN) {
  // CHECK-NEXT: %addi_index = arith.addi %a, %a : index
  %addi_index = arith.addi %a, %a : index
  // CHECK-NEXT: %index_cast = arith.index_cast %b : i32 to index
  %index_cast = arith.index_cast %b : i32 to index
  // CHECK-NEXT: %constant_index = arith.constant 0 : index
  %constant_index = arith.constant 0 : index
  // CHECK-NEXT: %addi_tensor = arith.addi %c, %c : tensor<4xi32>
  %addi_tensor = arith.addi %c, %c : tensor<4xi32>
  // CHECK-NEXT: %constant_tensor = arith.constant dense<0> : tensor<4xi32>
  %constant_tensor = arith.constant dense<0> : tensor<4xi32>
  // CHECK-NEXT: %addf_2d = arith.addf %d, %d : vector<2x4xf32>
  %addf_2d = arith.addf %d, %d : vector<2x4xf32>
  // CHECK-NEXT: %addi_signed = arith.addi %e, %e : si32
  %addi_signed = arith.addi %e, %e : si32
  // CHECK-NEXT: %addf_f8 = arith.addf %f, %f : f8E4M3FN
  %addf_f8 = arith.addf %f, %f : f8E4M3FN
  // CHECK-NEXT: %ceildivsi = arith.ceildivsi %b, %b : i32
  %ceildivsi = arith.ceildivsi %b, %b : i32
  // CHECK-NEXT: %sum, %overflow = arith.addui_extended %b, %b : i32, i1
  %sum, %overflow = arith.addui_extended %b, %b : i32, i1
  // CHECK-NEXT: "test.op"(%addi_index, %index_cast, %constant_index, %addi_tensor, %constant_tensor, %addf_2d, %addi_signed, %addf_f8, %ceildivsi, %sum, %overflow) : (index, index, index, tensor<4xi32>, tensor<4xi32>, vector<2x4xf32>, si32, f8E4M3FN, i32, i32, i1) -> ()
  "test.op"(%addi_index, %index_cast, %constant_index, %addi_tensor, %constant_tensor, %addf_2d, %addi_signed, %addf_f8, %ceildivsi, %sum, %overflow) : (index, index, index, tensor<4xi32>, tensor<4xi32>, vector<2x4xf32>, si32, f8E4M3FN, i32, i32, i1) -> ()
  func.return
}
