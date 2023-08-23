// RUN: XDSL_AUTO_ROUNDTRIP

builtin.module {
  func.func @builtin() {
    %0 = arith.constant 0 : i64
    %1 = builtin.unrealized_conversion_cast %0 : i64 to f32
    %2 = builtin.unrealized_conversion_cast %0 : i64 to i32
    %3 = builtin.unrealized_conversion_cast %0 : i64 to i64
    %4 = builtin.unrealized_conversion_cast to i64 {"comment" = "test"}
    %5 = builtin.unrealized_conversion_cast %0, %0 : i64, i64 to f32
    %6, %7 = builtin.unrealized_conversion_cast %5 : f32 to i64, i64
    func.return
  }
}
