// RUN: xdsl-opt %s -p reconcile-unrealized-casts | filecheck %s

builtin.module {
    // CHECK:  builtin.module {

    func.func @unused_cast(%arg0: i64) -> i64 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i16
        "func.return"(%arg0) : (i64) -> ()
    }

    // CHECK-NEXT:  func.func @unused_cast(%arg0 : i64) -> i64 {
    // CHECK-NEXT:    func.return %arg0 : i64
    // CHECK-NEXT:  }

    // -----

    func.func @simple_cycle(%arg0: i64) -> i64 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
        %1 = "builtin.unrealized_conversion_cast"(%0) : (i32) -> i64
        "func.return"(%1) : (i64) -> ()
    }

    // CHECK-NEXT:  func.func @simple_cycle(%arg0_1 : i64) -> i64 {
    // CHECK-NEXT:    func.return %arg0_1 : i64
    // CHECK-NEXT:  }

    // -----

    func.func @cycle_singleblock(%arg0: i64) -> i64 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i16
        %1 = "builtin.unrealized_conversion_cast"(%0) : (i16) -> i1
        %2 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i64
        %3 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i16
        %4 = "builtin.unrealized_conversion_cast"(%3) : (i16) -> i64
        %5 = "test.op"(%2, %4) : (i64, i64) -> i64
        func.return %5 : i64
    }

    // CHECK-NEXT:    func.func @cycle_singleblock(%arg0_2 : i64) -> i64 {
    // CHECK-NEXT:      %0 = "test.op"(%arg0_2, %arg0_2) : (i64, i64) -> i64
    // CHECK-NEXT:      func.return %0 : i64
    // CHECK-NEXT:    }

    // -----

    func.func @cycle_multiblock(%arg0: i64) -> i64 {
        %c0 = "test.op"() : () -> i32
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i16
        %1 = "builtin.unrealized_conversion_cast"(%0) : (i16) -> i1
        %2 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i64
        %3 = "builtin.unrealized_conversion_cast"(%1) : (i1) -> i16
        "cf.br"(%c0)[^bb1] : (i32) -> ()
        ^bb1(%33: i32):  // pred: ^bb0
        %4 = "builtin.unrealized_conversion_cast"(%3) : (i16) -> i64
        %5 = "test.op"(%2, %4) : (i64, i64) -> i64
        func.return %5 : i64
    }

    // CHECK-NEXT:    func.func @cycle_multiblock(%arg0_3 : i64) -> i64 {
    // CHECK-NEXT:      %c0 = "test.op"() : () -> i32
    // CHECK-NEXT:      "cf.br"(%c0) [^0] : (i32) -> ()
    // CHECK-NEXT:    ^0(%1 : i32):
    // CHECK-NEXT:      %2 = "test.op"(%arg0_3, %arg0_3) : (i64, i64) -> i64
    // CHECK-NEXT:      func.return %2 : i64
    // CHECK-NEXT:    }

    // -----

    func.func @failure_simple_cast(%arg0: i64) -> i32 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i32
        func.return %0 : i32
    }

    // CHECK-NEXT:    func.func @failure_simple_cast(%arg0_4 : i64) -> i32 {
    // CHECK-NEXT:      %3 = "builtin.unrealized_conversion_cast"(%arg0_4) : (i64) -> i32
    // CHECK-NEXT:      func.return %3 : i32
    // CHECK-NEXT:    }

    // -----

    func.func @failure_chain(%arg0: i64) -> i32 {
        %0 = "builtin.unrealized_conversion_cast"(%arg0) : (i64) -> i1
        %1 = "builtin.unrealized_conversion_cast"(%0) : (i1) -> i32
        func.return %1 : i32
    }

    // CHECK-NEXT:    func.func @failure_chain(%arg0_5 : i64) -> i32 {
    // CHECK-NEXT:      %4 = "builtin.unrealized_conversion_cast"(%arg0_5) : (i64) -> i1
    // CHECK-NEXT:      %5 = "builtin.unrealized_conversion_cast"(%4) : (i1) -> i32
    // CHECK-NEXT:      func.return %5 : i32
    // CHECK-NEXT:    }
}
// CHECK-NEXT:  }
