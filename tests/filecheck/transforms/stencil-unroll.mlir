func.func @cse_recursive_effects_failure() -> (i32, i32, i32) {
    %104 = "test.op_with_memread"() : () -> i32
    %105 = arith.constant true
    %106 = "scf.if"(%105) ({
      "test.op_with_memwrite"() : () -> ()
      %107 = arith.constant 42 : i32
      scf.yield %107 : i32
    }, {
      %108 = arith.constant 24 : i32
      scf.yield %108 : i32
    }) : (i1) -> i32
    %109 = "test.op_with_memread"() : () -> i32
    func.return %104, %109, %106 : i32, i32, i32
  }
