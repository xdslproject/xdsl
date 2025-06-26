// RUN: xdsl-opt %s  --split-input-file --verify-diagnostics | filecheck %s

"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  "scf.index_switch"(%0) <{cases = array<i32: 0>}> ({
  // CHECK: Expected attribute i64 but got i32
    "scf.yield"() : () -> ()
  }, {
    "scf.yield"() : () -> ()
  }) : (index) -> ()
}) : () -> ()

// -----

"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  "scf.index_switch"(%0) <{cases = array<i64: 0, 1>}> ({
  // CHECK: has 1 case regions but 2 case values
    "scf.yield"() : () -> ()
  }, {
    "scf.yield"() : () -> ()
  }) : (index) -> ()
}) : () -> ()

// -----

"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  "scf.index_switch"(%0) <{cases = array<i64: 0, 1>}> ({
  // CHECK: 'scf.index_switch' terminates with operation test.termop instead of scf.yield
    "test.termop"() : () -> ()
  }, {
    "scf.yield"() : () -> ()
  }) : (index) -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 0 : index}> : () -> index
  "scf.index_switch"(%0) <{cases = array<i64: 0>}> ({
    %1 = "arith.constant"() <{value = 0 : i64}> : () -> i64
    "scf.yield"(%1) : (i64) -> ()
  }, {
    %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
    "scf.yield"(%2) : (i32) -> ()
    // CHECK: region 0 returns values of types (i32) but expected (i64)
  }) : (index) -> (i64)
}) : () -> ()
