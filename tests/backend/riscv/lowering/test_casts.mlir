builtin.module {
    func.func @bifurcation(%arg0: i64) -> i64 {
        %c0 = arith.constant 0 : i32
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
        %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i1
        %2 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i64
        %3 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i32
        "cf.br"(%c0)[^bb1] : (i32) -> ()
        ^bb1(%33: i32):  // pred: ^bb0
        %4 = "builtin.unrealized_conversion_cast"(%3) : (i32) -> i64
        %5 = arith.addi %2, %4 : i64
        func.return %5 : i64
    }
}
