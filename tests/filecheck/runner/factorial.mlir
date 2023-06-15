// RUN: xdsl-run %s; echo $? | filecheck %s

builtin.module {

  func.func @factorial(%i : i64) -> i64 {
    %zero = "arith.constant"() {"value" = 0} : () -> i64
    %one = "arith.constant"() {"value" = 1} : () -> i64
    %eq0 = "arith.cmpi"(%i, %zero) {"predicate" = 0} : (i64, i64) -> i1
    %ret = "scf.if"(%eq0) ({
      "scf.yield"(%one) : (i64) -> ()
    },{
      %im1 = "arith.subi"(%i, %one) : (i64, i64) -> i64
      %facrec = "func.call"(%im1) {"callee" = @factorial} : (i64) -> i64
      "scf.yield"(%facrec) : (i64) -> ()
    }) : (i1) -> i64
    "func.return"(%ret) : (i64) -> ()
  }
  func.func @main() -> index {
    %zero = "arith.constant"() {"value" = 0} : () -> index
    %i = "arith.constant"() {"value" = 3} : () -> i64
    %fac = "func.call"(%i) {"callee" = @factorial} : (i64) -> i64
    print.println "factorial({})={}", %i : i64, %fac : i64
    "func.return"(%zero) : (index) -> ()
  }
}

// CHECK: 12
